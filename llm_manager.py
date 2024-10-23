import random as rd
from typing import Optional, Union, Dict, Tuple, List, Callable, Set
from abc import ABC, abstractmethod
import os
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import datasets
from datasets import DatasetDict
import numpy as np
import pandas as pd
import evaluate
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import is_datasets_available
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import (
    mean_over_attention_ranges,
    mean_over_attention_ranges_python_slow,
    mean_over_hidden_states_python_slow,
    mean_over_hidden_states,
    replace_ranges,
    get_combinations,
    find_non_existing_source_targets,
    row_to_vanilla_datapoint,
    row_to_prompt_datapoint,
    row_to_input_embeds_replace_datapoint,
)

ID2LABEL = {0: "FALSE", 1: "TRUE"}
LABEL2ID = {"FALSE": 0, "TRUE": 1}

SPLIT_EPOCH_ENDING = f"/split_{{}}_pos_{{}}_com_{{}}.npy"
TOKENS_ENDING = "/tokens.csv"

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
ALL_SEMANTIC_TOKENS = ["cls", "user", "sep1", "title", "sep2", "genres", "sep3"]
EMBEDDINGS_BASED_SEMANTIC_TOKENS = ALL_SEMANTIC_TOKENS + [
    "user embedding",
    "sep4",
    "movie embedding",
    "sep5",
]


def get_semantic_positional_encoding(
    input_ids: torch.Tensor, sep_token_id: int
) -> torch.Tensor:
    mask = input_ids == sep_token_id
    positions = mask.nonzero(as_tuple=True)
    cols = positions[1]

    # Step 3: Determine the number of True values per row
    num_trues_per_row = mask.sum(dim=1)
    max_trues_per_row = num_trues_per_row.max().item()
    # Step 4: Create an empty tensor to hold the result
    semantic_positional_encoding = -torch.ones(
        (mask.size(0), max_trues_per_row),  # type: ignore
        dtype=torch.long,
    )

    # Step 5: Use scatter to place column indices in the semantic_positional_encoding tensor
    # Create an index tensor that assigns each column index to the correct position in semantic_positional_encoding tensor
    row_indices = torch.arange(mask.size(0)).repeat_interleave(num_trues_per_row)
    column_indices = torch.cat([torch.arange(n) for n in num_trues_per_row])  # type: ignore

    semantic_positional_encoding[row_indices, column_indices] = cols
    semantic_positional_encoding = torch.stack(
        [
            semantic_positional_encoding[:, :-1] + 1,
            semantic_positional_encoding[:, 1:],
        ],
        dim=2,
    )
    # Create a tensor of zeros to represent the starting points
    second_points = torch.ones(
        semantic_positional_encoding.size(0),
        1,
        2,
        dtype=semantic_positional_encoding.dtype,
    )
    # Set the second column to be the first element of the first range
    second_points[:, 0, 1] = semantic_positional_encoding[:, 0, 0] - 1
    # Concatenate the start_points tensor with the original semantic_positional_encoding tensor
    semantic_positional_encoding = torch.cat(
        (second_points, semantic_positional_encoding), dim=1
    )
    return semantic_positional_encoding


def sort_ranges(semantic_positional_encoding: torch.Tensor):
    # Extract the second element (end of the current ranges excluded the starting cps token)
    end_elements = semantic_positional_encoding[:, :, 1]
    # Create the new ranges by adding 1 to the end elements
    new_ranges = torch.stack([end_elements, end_elements + 1], dim=-1)
    # add the cls positions to it
    cls_positions = torch.tensor([0, 1])
    cls_positions = cls_positions.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 2)
    cls_positions = cls_positions.expand(
        new_ranges.size(0), 1, -1
    )  # Shape (batch_size, 1, 2)
    new_ranges = torch.cat((new_ranges, cls_positions), dim=1)
    # Concatenate the original ranges with the new ranges
    semantic_positional_encoding = torch.cat(
        (semantic_positional_encoding, new_ranges), dim=1
    )
    # Step 1: Extract the last value of dimension 2
    last_values = semantic_positional_encoding[
        :, :, -1
    ]  # Shape (batch_size, num_elements)

    # Step 2: Sort the indices based on these last values
    # 'values' gives the sorted values (optional), 'indices' gives the indices to sort along dim 1
    _, indices = torch.sort(last_values, dim=1, descending=False)

    # Step 3: Apply the sorting indices to the original tensor
    semantic_positional_encoding = torch.gather(
        semantic_positional_encoding,
        1,
        indices.unsqueeze(-1).expand(-1, -1, semantic_positional_encoding.size(2)),
    )
    return semantic_positional_encoding


class DataCollatorBase(DataCollatorForLanguageModeling, ABC):
    """
    The Data Collators are used to generate non-existing edges on the fly. The false ratio allows to decide the ratio,
    in existing edges are replaced with non-existing edges.
    """

    def __init__(
        self, tokenizer, df, source_df, target_df, device="cpu", false_ratio=0.5
    ):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.false_ratio = false_ratio
        self.tokenizer = tokenizer
        self.source_ids, self.target_ids, self.edges = self.__df_to_nodes(df)
        self.source_df = source_df
        self.target_df = target_df
        self.device = device

    def __df_to_nodes(
        self, df: pd.DataFrame
    ) -> Tuple[List[int], List[int], Set[Tuple[int, int]]]:
        source_ids: List[int] = list(df["source_id"].unique())
        target_ids: List[int] = list(df["target_id"].unique())
        edges: Set[Tuple[int, int]] = set(
            df[["source_id", "target_id"]].itertuples(index=False, name=None)
        )
        return source_ids, target_ids, edges

    def __call__(self, features):
        if self.false_ratio != -1:
            total_features = len(features)
            new_feature_amount = int(self.false_ratio * total_features)
            with torch.no_grad():
                new_features = self._generate_false_examples(k=new_feature_amount)
            chosen_indices = rd.choices(range(total_features), k=new_feature_amount)
            for chosen_index, feature in zip(chosen_indices, new_features):
                features[chosen_index] = feature
        # Convert features into batches
        return self._convert_features_into_batches(features)

    @abstractmethod
    def _generate_false_examples(self, k: int) -> Dict:
        pass

    @abstractmethod
    def _convert_features_into_batches(self, features: List[Dict]) -> Dict:
        pass


class TextBasedDataCollator(DataCollatorBase, ABC):
    def __init__(self, tokenizer, df, source_df, target_df, false_ratio=0.5):
        super().__init__(tokenizer, df, source_df, target_df, false_ratio=false_ratio)

    def _convert_features_into_batches(self, features: List[Dict]) -> Dict:
        input_ids = []
        attention_mask = []
        labels = []
        semantic_positional_encoding = []

        for f in features:
            input_ids.append(f["input_ids"])
            attention_mask.append(f["attention_mask"])
            labels.append(f["labels"])
            semantic_positional_encoding.append(f["semantic_positional_encoding"])
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        semantic_positional_encoding = torch.tensor(
            semantic_positional_encoding, dtype=torch.long
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "semantic_positional_encoding": semantic_positional_encoding,
        }


class EmbeddingBasedDataCollator(DataCollatorBase):
    def __init__(
        self,
        tokenizer,
        device,
        df,
        source_df,
        target_df,
        data,
        get_embeddings_cb,
        false_ratio=0.5,
    ):
        super().__init__(
            tokenizer, df, source_df, target_df, false_ratio=false_ratio, device=device
        )
        self.data = data
        self.get_embeddings_cb = get_embeddings_cb

    def _convert_features_into_batches(self, features: List[Dict]) -> Dict:
        input_ids = []
        attention_mask = []
        labels = []
        semantic_positional_encoding = []
        source_ids = []
        target_ids = []
        graph_embeddings = []
        for f in features:
            input_ids.append(f["input_ids"])
            attention_mask.append(f["attention_mask"])
            labels.append(f["labels"])
            semantic_positional_encoding.append(f["semantic_positional_encoding"])
            source_ids.append(f["source_id"])
            target_ids.append(f["target_id"])
            if "graph_embeddings" in f:
                graph_embeddings.append(f["graph_embeddings"])
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        semantic_positional_encoding = torch.tensor(
            semantic_positional_encoding, dtype=torch.long
        )
        if len(graph_embeddings) == 0:
            source_embeddings, target_embeddings = self.get_embeddings_cb(
                self.data, source_ids, target_ids
            )
            graph_embeddings = torch.stack(
                [source_embeddings, target_embeddings], dim=1
            )
            del source_embeddings, target_embeddings
        else:
            graph_embeddings = torch.tensor(graph_embeddings)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "graph_embeddings": graph_embeddings,
            "semantic_positional_encoding": semantic_positional_encoding,
        }

    def _generate_false_examples(self, k: int) -> List[Dict]:
        node_pairs = find_non_existing_source_targets(
            self.edges, self.source_ids, self.target_ids, k=k
        )
        df = pd.DataFrame(
            {"source_id": node_pairs[:, 0], "target_id": node_pairs[:, 1]}
        )
        df = (
            df.merge(self.source_df, left_on="source_id", right_on="id")
            .reset_index(drop=True)
            .merge(self.target_df, left_on="target_id", right_on="id")
            .reset_index(drop=True)
        )
        df["prompt"] = df.apply(
            lambda row: row_to_input_embeds_replace_datapoint(
                row, self.tokenizer.sep_token, self.tokenizer.pad_token
            ),
            axis=1,
        )
        prompts = df["prompt"].tolist()
        del df
        tokenized = self.tokenizer(prompts, padding="max_length", truncation=True)
        del prompts
        sep_token_id = self.tokenizer.sep_token_id
        assert sep_token_id
        input_ids = torch.tensor(tokenized["input_ids"])
        semantic_positional_encoding = get_semantic_positional_encoding(
            input_ids,
            sep_token_id,
        )
        semantic_positional_encoding = sort_ranges(semantic_positional_encoding)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        # source_embeddings, target_embeddings = self.get_embeddings_cb(
        #     self.data, node_pairs[:, 0].tolist(), node_pairs[:, 1].tolist()
        # )
        # graph_embeddings = torch.stack([source_embeddings, target_embeddings], dim=1)
        result_dict = [
            {
                "input_ids": input_ids_,
                "attention_mask": attention_mask_,
                "labels": 0,
                "semantic_positional_encoding": semantic_positional_encoding_.to("cpu")
                .detach()
                .tolist(),
                "source_id": node_pair[0],
                "target_id": node_pair[1],
            }
            for semantic_positional_encoding_, input_ids_, attention_mask_, node_pair in zip(
                semantic_positional_encoding,
                input_ids,
                attention_mask,
                node_pairs,
            )
        ]
        return result_dict


class PromptEmbeddingDataCollator(TextBasedDataCollator):
    """
    The Prompt Data Collator also adds embeddings to the prompt on the fly.
    """

    def __init__(
        self,
        tokenizer,
        df,
        source_df,
        target_df,
        data,
        get_embeddings_cb,
        false_ratio=0.5,
    ):
        super().__init__(tokenizer, df, source_df, target_df, false_ratio=false_ratio)
        self.data = data
        self.get_embeddings_cb = get_embeddings_cb

    def _generate_false_examples(self, k: int) -> List[Dict]:
        node_pairs = find_non_existing_source_targets(
            self.edges, self.source_ids, self.target_ids, k=k
        )
        df = pd.DataFrame(
            {"source_id": node_pairs[:, 0], "target_id": node_pairs[:, 1]}
        )
        df = (
            df.merge(self.source_df, left_on="source_id", right_on="id")
            .reset_index(drop=True)
            .merge(self.target_df, left_on="target_id", right_on="id")
            .reset_index(drop=True)
        )
        source_embeddings, target_embeddings = self.get_embeddings_cb(
            self.data, node_pairs[:, 0].tolist(), node_pairs[:, 1].tolist()
        )
        df["prompt_source_embedding"] = source_embeddings.to("cpu").detach().tolist()
        df["prompt_target_embedding"] = target_embeddings.to("cpu").detach().tolist()

        df["prompt"] = df.apply(
            lambda row: row_to_prompt_datapoint(row, self.tokenizer.sep_token),
            axis=1,
        )
        prompts = df["prompt"].tolist()
        tokenized = self.tokenizer(prompts, padding="max_length", truncation=True)
        sep_token_id = self.tokenizer.sep_token_id
        assert sep_token_id
        input_ids = torch.tensor(tokenized["input_ids"])
        semantic_positional_encoding = get_semantic_positional_encoding(
            input_ids,
            sep_token_id,
        )
        semantic_positional_encoding = sort_ranges(semantic_positional_encoding)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        result_dict = [
            {
                "input_ids": input_ids_,
                "attention_mask": attention_mask_,
                "labels": 0,
                "semantic_positional_encoding": semantic_positional_encoding_.to("cpu")
                .detach()
                .tolist(),
            }
            for semantic_positional_encoding_, input_ids_, attention_mask_ in zip(
                semantic_positional_encoding,
                input_ids,
                attention_mask,
            )
        ]
        return result_dict

    def _transform_to_false_example(self) -> Dict:
        label = 0
        source_id, target_id = find_non_existing_source_targets(self.df)

        random_source: pd.DataFrame = (
            self.source_df[self.source_df["id"] == source_id]
            .sample(1)
            .rename(columns={"id": "source_id"})
            .reset_index(drop=True)
        )
        random_target: pd.DataFrame = (
            self.target_df[self.target_df["id"] == target_id]
            .sample(1)
            .rename(columns={"id": "target_id"})
            .reset_index(drop=True)
        )
        random_row = pd.concat([random_source, random_target], axis=1).iloc[0]
        source_embedding, target_embedding = self.get_embeddings_cb(
            self.data, source_id, target_id
        )
        random_row["prompt_source_embedding"] = source_embedding.detach().to("cpu")
        random_row["prompt_target_embedding"] = target_embedding.detach().to("cpu")
        random_row["prompt"] = row_to_prompt_datapoint(
            random_row, sep_token=self.tokenizer.sep_token
        )
        tokenized = self.tokenizer(
            random_row["prompt"], padding="max_length", truncation=True
        )
        sep_token_id = self.tokenizer.sep_token_id
        assert sep_token_id
        semantic_positional_encoding = get_semantic_positional_encoding(
            torch.tensor(tokenized["input_ids"]).unsqueeze(0),
            sep_token_id,
        )
        semantic_positional_encoding = sort_ranges(semantic_positional_encoding)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label,
            "semantic_positional_encoding": semantic_positional_encoding.squeeze(0)
            .to("cpu")
            .detach()
            .tolist(),
        }


class VanillaEmbeddingDataCollator(TextBasedDataCollator):
    """
    The vanilla data collator does only generate false edges without KGEs.
    """

    def __init__(self, tokenizer, df, source_df, target_df, false_ratio=0.5):
        super().__init__(tokenizer, df, source_df, target_df, false_ratio=false_ratio)

    def _generate_false_examples(self, k: int) -> List[Dict]:
        node_pairs = find_non_existing_source_targets(
            self.edges, self.source_ids, self.target_ids, k=k
        )
        df = pd.DataFrame(
            {"source_id": node_pairs[:, 0], "target_id": node_pairs[:, 1]}
        )
        df = (
            df.merge(self.source_df, left_on="source_id", right_on="id")
            .reset_index(drop=True)
            .merge(self.target_df, left_on="target_id", right_on="id")
            .reset_index(drop=True)
        )
        df["prompt"] = df.apply(
            lambda row: row_to_vanilla_datapoint(row, self.tokenizer.sep_token), axis=1
        )
        prompts = df["prompt"].tolist()
        tokenized = self.tokenizer(prompts, padding="max_length", truncation=True)
        sep_token_id = self.tokenizer.sep_token_id
        assert sep_token_id
        input_ids = torch.tensor(tokenized["input_ids"])
        semantic_positional_encoding = get_semantic_positional_encoding(
            input_ids,
            sep_token_id,
        )
        semantic_positional_encoding = sort_ranges(semantic_positional_encoding)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        result_dict = [
            {
                "input_ids": input_ids_,
                "attention_mask": attention_mask_,
                "labels": 0,
                "semantic_positional_encoding": semantic_positional_encoding_.to("cpu")
                .detach()
                .tolist(),
            }
            for semantic_positional_encoding_, input_ids_, attention_mask_ in zip(
                semantic_positional_encoding, input_ids, attention_mask
            )
        ]
        return result_dict

    def _transform_to_false_example(self) -> Dict:
        label = 0
        edge = find_non_existing_source_targets(
            self.edges, self.source_ids, self.target_ids
        )
        source_id = edge[0, 0].item()
        target_id = edge[0, 1].item()
        random_source: pd.DataFrame = (
            self.source_df[self.source_df["id"] == source_id]
            .sample(1)
            .rename(columns={"id": "source_id"})
            .reset_index(drop=True)
        )
        random_target: pd.DataFrame = (
            self.target_df[self.target_df["id"] == target_id]
            .sample(1)
            .rename(columns={"id": "target_id"})
            .reset_index(drop=True)
        )
        random_row = pd.concat([random_source, random_target], axis=1).iloc[0]
        random_row["prompt"] = row_to_vanilla_datapoint(
            random_row, self.tokenizer.sep_token
        )
        tokenized = self.tokenizer(
            random_row["prompt"], padding="max_length", truncation=True
        )
        sep_token_id = self.tokenizer.sep_token_id
        assert sep_token_id
        semantic_positional_encoding = get_semantic_positional_encoding(
            torch.tensor(tokenized["input_ids"]).unsqueeze(0),
            sep_token_id,
        )
        semantic_positional_encoding = sort_ranges(semantic_positional_encoding)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label,
            "semantic_positional_encoding": semantic_positional_encoding.squeeze(0)
            .to("cpu")
            .detach()
            .tolist(),
        }


METRIC = evaluate.load("accuracy")


class CustomTrainer(Trainer):
    """
    This custom trainer is needed, so we can have different data collators while training and evaluating.
    For that we adjust the get_eval_dataloader method.
    """

    def __init__(self, *args, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_data_collator = eval_data_collator

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.val_dataset[eval_dataset]` as the evaluation dataset.
                If a `Dataset`, will override `self.val_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`],
                columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])
        eval_dataset = (
            self.eval_dataset[eval_dataset]  # type: ignore
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )  # type: ignore
        data_collator = self.test_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )  # type: ignore
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator,  # type: ignore
                description="evaluation",
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):  # type: ignore
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)  # type: ignore
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)  # type: ignore
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}
        return self.accelerator.prepare(eval_dataloader)


class SequenceClassifierOutputOverRanges(SequenceClassifierOutput):
    def __init__(
        self,
        logits: torch.FloatTensor,
        loss: Optional[torch.FloatTensor] = None,
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None,
        attentions: Optional[Tuple[torch.FloatTensor, ...]] = None,
        semantic_positional_encoding: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions
        )  # type: ignore
        self.semantic_positional_encoding = semantic_positional_encoding

    def to_tuple(self):
        # Ensure that your custom field is included when converting to a tuple
        return tuple(
            v
            for v in (
                self.loss,
                self.logits,
                self.hidden_states,
                self.attentions,
                self.semantic_positional_encoding,
            )
            if v is not None
        )


class BertForSequenceClassificationRanges(BertForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        semantic_positional_encoding: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputOverRanges]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputOverRanges(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            semantic_positional_encoding=semantic_positional_encoding,
        )


class InputEmbedsReplaceBertEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    """
    Addition by this works context author: We add the semantic positional encodings and graph embeddings, so that we can replace
    the padding placeholders with the KGEs, before adding the positional or type encodings.
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        graph_embeddings: Optional[torch.FloatTensor] = None,
        semantic_positional_encoding: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            if (
                semantic_positional_encoding is not None
                and graph_embeddings is not None
                and inputs_embeds is not None
            ):
                batch_indices = torch.arange(len(inputs_embeds)).unsqueeze(1)
                inputs_embeds[
                    batch_indices, semantic_positional_encoding[:, [-4, -2], 0]
                ] = graph_embeddings  # replace the input embeds at the place holder positions with the KGEs.
                # inputs_embeds = inputs_embeds.scatter(
                #     1, mask, graph_embeddings
                # )
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class InputEmbedsReplaceBertModel(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    """
    Addition by this works context author: We add the semantic positional encodings and graph embeddings, so that we can replace
    the padding placeholders with the KGEs, before adding the positional or type encodings.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = InputEmbedsReplaceBertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()


class InputEmbedsReplaceBertForSequenceClassification(BertForSequenceClassification):
    """The bert base model has been adjusted so it also takes SPEs and KGEs."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = InputEmbedsReplaceBertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        graph_embeddings: Optional[torch.Tensor] = None,
        semantic_positional_encoding: Optional[torch.Tensor] = None,
        source_id: Optional[torch.LongTensor] = None,
        target_id: Optional[
            torch.LongTensor
        ] = None,  # we don't need source and target ids here anymore but only so that the collator picks them up on the way
        # and we can transform them to graph embeddings.
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputOverRanges]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if inputs_embeds is None:
            inputs_embeds = self.bert.embeddings(
                input_ids,
                semantic_positional_encoding=semantic_positional_encoding,
                graph_embeddings=graph_embeddings,
            )  # we replaced the placeholder padding tokens in the embedding generation with the KGEs
            assert isinstance(inputs_embeds, torch.Tensor)
        outputs = self.bert(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # feed forward the input embeds to the attention model

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputOverRanges(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            semantic_positional_encoding=semantic_positional_encoding,
        )


class ClassifierBase(ABC):
    def __init__(
        self,
        df,
        model,
        tokenizer,
        train_data_collator,
        test_data_collator,
        val_data_collator,
        model_max_length,
        semantic_datapoints,
        root_path,
        force_recompute=False,
    ) -> None:
        self.predictions = None
        self.df = df
        self.model = model
        self.tokenizer = tokenizer
        self.train_data_collator = train_data_collator
        self.test_data_collator = test_data_collator
        self.val_data_collator = val_data_collator
        self.semantic_datapoints = semantic_datapoints
        self.attentions_path = f"{root_path}/attentions.npy"
        self.logits_path = f"{root_path}/logits.npy"
        self.hidden_states_path = f"{root_path}/hidden_states_{{}}.npy"
        self.tokens_path = f"{root_path}/tokens.csv"
        self.training_path = f"{root_path}/training"
        self.best_model_path = f"{self.training_path}/best"
        self.log_path = f"{self.training_path}/logs"
        self.sub_attentions_dir_path = f"{root_path}/attentions"
        self.sub_logits_dir_path = f"{root_path}/logits"
        self.sub_hidden_states_dir_path = f"{root_path}/hidden_states"
        self.sub_attentions_path = f"{self.sub_attentions_dir_path}{SPLIT_EPOCH_ENDING}"
        self.sub_logits_path = f"{self.sub_logits_dir_path}/split_{{}}_com_{{}}.npy"
        self.sub_hidden_states_path = (
            f"{self.sub_hidden_states_dir_path}{SPLIT_EPOCH_ENDING}"
        )
        self.sub_tokens_dir_path = f"{root_path}/tokens"
        self.sub_tokens_path = f"{self.sub_tokens_dir_path}{TOKENS_ENDING}"
        self.force_recompute = force_recompute
        self.tokenizer = BertTokenizer.from_pretrained(
            MODEL_NAME, model_max_length=model_max_length
        )

    def _get_data_collator(self, split) -> DataCollatorForLanguageModeling:
        return (
            self.test_data_collator
            if split == "test"
            else self.val_data_collator
            if split == "val"
            else self.train_data_collator
        )

    def plot_confusion_matrix(
        self,
        split,
        dataset,
        tokenize=False,
        batch_size: int = 64,
        force_recompute=False,
    ):
        if split == "test":
            trainer = self._get_trainer(
                dataset, tokenize=tokenize, batch_size=batch_size
            )
            dataset = dataset["test"]
        else:
            trainer = self._get_trainer(
                dataset,
                tokenize=tokenize,
                eval_data_collator=self.val_data_collator,
                batch_size=batch_size,
            )
            dataset = dataset["val"]
        if not self.predictions or force_recompute:
            # Generate predictions
            predictions = trainer.predict(dataset)
            self.predictions = predictions
        # Get predicted labels and true labels
        preds = np.argmax(self.predictions.predictions, axis=-1)
        labels = self.predictions.label_ids
        # Compute confusion matrix
        cm = confusion_matrix(labels, preds)  # type: ignore

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Negative", "Positive"]
        )
        disp.plot(cmap=plt.cm.Blues)  # type: ignore
        plt.show()

    def _get_trainer(
        self,
        dataset,
        tokenize=False,
        eval_data_collator=None,
        epochs=3,
        batch_size: int = 64,
    ):
        if tokenize:
            tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        else:
            tokenized_dataset = dataset
        training_args = TrainingArguments(
            output_dir=self.training_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=self.log_path,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
        )
        if not eval_data_collator:
            eval_data_collator = self.test_data_collator
        # Initialize the Trainer
        return CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=self.train_data_collator,
            eval_data_collator=eval_data_collator,
            compute_metrics=self._compute_metrics,
        )

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return METRIC.compute(predictions=predictions, references=labels)

    @abstractmethod
    def tokenize_function(self, example, return_pt=False):
        pass

    def train_model_on_data(self, dataset, epochs=3, batch_size: int = 64):
        trainer = self._get_trainer(dataset, epochs=epochs, batch_size=batch_size)

        # Train the model
        trainer.train()

        trainer.model.to(device="cpu").save_pretrained(self.best_model_path)  # type: ignore
        trainer.model.to(device=self.device)  # type: ignore

    @staticmethod
    def _plot_training_loss_and_accuracy(model_type: str, root: str = "./data/llm"):
        training_state_path = (
            f"{root}/{model_type}/training/checkpoint-4420/trainer_state.json"
        )
        with open(training_state_path, "r") as f:
            trainer_state = json.load(f)
            # Extract loss values and corresponding steps
        losses = []
        steps = []

        for log in trainer_state["log_history"]:
            if "loss" in log:
                losses.append(log["loss"])
                steps.append(log["step"])

        # Extract accuracy values and corresponding epochs
        accuracies = []
        epochs = []

        for log in trainer_state["log_history"]:
            if "eval_accuracy" in log:
                accuracies.append(log["eval_accuracy"])
                epochs.append(log["epoch"])

        # Find the minimum loss and its corresponding step
        min_loss = min(losses)
        min_loss_step = steps[losses.index(min_loss)]

        # Find the maximum accuracy and its corresponding epoch
        max_accuracy = max(accuracies)
        max_accuracy_epoch = epochs[accuracies.index(max_accuracy)]

        # Plot loss development over steps
        plt.figure(figsize=(12, 6))
        plt.plot(steps, losses, label="Loss")
        plt.scatter(min_loss_step, min_loss, color="red")  # Mark the minimum loss
        plt.text(
            min_loss_step,
            min_loss,
            f"Min Loss: {min_loss:.4f}",
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
        )
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"Loss Development over Steps of {model_type} Model")
        plt.legend()
        plt.show()

        # Plot accuracy development over epochs
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, accuracies, label="Accuracy", color="green")
        plt.scatter(
            max_accuracy_epoch, max_accuracy, color="red"
        )  # Mark the maximum accuracy
        plt.text(
            max_accuracy_epoch,
            max_accuracy,
            f"Max Accuracy: {max_accuracy:.4f}",
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Development over Epochs of {model_type} Model")
        plt.legend()
        plt.show()

    def _means_over_ranges_cross(
        self,
        all_semantic_positional_encoding: torch.Tensor,
        attentions: torch.Tensor,
    ) -> torch.Tensor:
        attentions = mean_over_attention_ranges(
            attentions,
            all_semantic_positional_encoding[:, :, 0],
            all_semantic_positional_encoding[:, :, 1],
        )
        return attentions

    def forward_dataset_and_save_outputs(
        self,
        dataset: datasets.Dataset | datasets.DatasetDict,
        get_tokens_as_df_cb: Callable,
        splits: List[str] = ["train", "test", "val"],
        batch_size: int = 64,
        save_step_size: int = 1,
        load_fields: List[str] = ["attentions", "hidden_states", "logits"],
        is_test: bool = False,
        force_recompute: bool = False,
        combination_boundaries: Optional[Tuple[int, int]] = None,
    ) -> None:
        add_hidden_states = "hidden_states" in load_fields

        add_attentions = "attentions" in load_fields
        add_logits = "logits" in load_fields
        hidden_states_exist = True
        if is_test:
            dataset = DatasetDict(
                {
                    split: dataset.select(range(batch_size * 2))
                    for split, dataset in dataset.items()
                }
            )
        if (
            force_recompute
            or not os.path.exists(self.attentions_path)
            or not hidden_states_exist
            or not os.path.exists(self.tokens_path)
        ):
            assert isinstance(self.model, BertForSequenceClassification)
            self.model.eval()
            Path(self.sub_attentions_dir_path).mkdir(parents=True, exist_ok=True)
            Path(self.sub_logits_dir_path).mkdir(parents=True, exist_ok=True)
            Path(self.sub_hidden_states_dir_path).mkdir(parents=True, exist_ok=True)
            Path(self.sub_tokens_dir_path).mkdir(parents=True, exist_ok=True)
            semantic_positional_encodings = {
                "cls": [0],
                "user_id": [1],
                "movie_id": [3],
                "title": [5],
                "genres": [7],
                "seps": [2, 4, 6, 8],
            }
            sep_list = ["cls", "user_id", "movie_id", "title", "genres", "seps"]
            if len(dataset[splits[0]][0]["semantic_positional_encoding"]) > 9:
                semantic_positional_encodings["seps"].extend([10, 12])
                semantic_positional_encodings["kges"] = [9, 11]
                sep_list.append("seps")
            tokens_collected = False
            all_tokens = []
            key_combinations = get_combinations(sep_list)
            if combination_boundaries:
                assert (
                    combination_boundaries[0] < len(key_combinations)
                ), f"Expected boundaries to be smaller then the amount of combinations, but got {combination_boundaries[0]} at position 0 for {len(key_combinations)}"
                assert (
                    combination_boundaries[1] < len(key_combinations)
                ), f"Expected boundaries to be smaller then the amount of combinations, but got {combination_boundaries[1]} at position 1 for {len(key_combinations)}"
                assert (
                    combination_boundaries[0] < combination_boundaries[1]
                ), f"Expected boundaries at position 0 to be smaller then boundary at position 1, but got {combination_boundaries}"
                key_combinations = key_combinations[
                    combination_boundaries[0] : combination_boundaries[1]
                ]
            with torch.no_grad():
                for split in splits:
                    data_collator = self._get_data_collator(split)
                    data_loader = DataLoader(
                        dataset=dataset[split],  # type: ignore
                        batch_size=batch_size,
                        collate_fn=data_collator,
                    )
                    for combination in key_combinations:
                        combination_string = str(list(combination))
                        if len(combination) == 0:
                            print(
                                "forwarding without masking",
                                self.sub_attentions_path.format(
                                    split, 0, combination_string
                                ),
                            )
                        logits_of_combination = []
                        attentions_collected = []
                        hidden_states_collected = []
                        for idx, batch in enumerate(data_loader):
                            print(f"batch {idx}, combination {combination_string}")
                            # idx = 0
                            # if True:
                            batch = next(iter(data_loader))
                            splits_ = [split] * len(batch["input_ids"])
                            for key in combination:
                                for pos in semantic_positional_encodings[key]:
                                    batch["attention_mask"] = replace_ranges(
                                        batch["attention_mask"],
                                        batch["semantic_positional_encoding"][:, pos],
                                    )
                            outputs = self.model(
                                **batch,
                                output_hidden_states=add_hidden_states,
                                output_attentions=add_attentions,
                            )
                            semantic_positional_encoding = batch[
                                "semantic_positional_encoding"
                            ]
                            if add_attentions:
                                attentions = outputs.attentions
                                attentions = [
                                    torch.sum(layer, dim=1) for layer in attentions
                                ]
                                attentions = torch.stack(attentions).permute(1, 2, 3, 0)
                                if is_test:
                                    attentions_test = (
                                        mean_over_attention_ranges_python_slow(
                                            attentions,
                                            semantic_positional_encoding[:, :, 0],
                                            semantic_positional_encoding[:, :, 1],
                                        )
                                    )
                                attentions = mean_over_attention_ranges(
                                    attentions,
                                    semantic_positional_encoding[:, :, 0],
                                    semantic_positional_encoding[:, :, 1],
                                )
                                if not is_test:
                                    attentions_collected.append(attentions)
                                    if (idx + 1) % save_step_size == 0:
                                        np.save(
                                            self.sub_attentions_path.format(
                                                split,
                                                (idx + 1) / save_step_size,
                                                combination_string,
                                            ),
                                            np.concatenate(attentions_collected),
                                        )
                                        attentions_collected = []
                                else:
                                    assert torch.allclose(
                                        attentions_test, attentions, atol=1e-8
                                    ), f"Expected attentions to be same, but are not: {attentions[0,7,7,0]}, {attentions_test[0,7,7,0]}"
                            if add_logits:
                                logits = outputs.logits
                                logits_of_combination.append(logits.numpy())
                                del logits
                            if add_hidden_states:
                                hidden_states = torch.stack(
                                    outputs.hidden_states, dim=1
                                )
                                if is_test:
                                    test_hidden_states = (
                                        mean_over_hidden_states_python_slow(
                                            hidden_states,
                                            semantic_positional_encoding[:, :, 0],
                                            semantic_positional_encoding[:, :, 1],
                                        )
                                    )
                                hidden_states = mean_over_hidden_states(
                                    hidden_states,
                                    semantic_positional_encoding[:, :, 0],
                                    semantic_positional_encoding[:, :, 1],
                                )
                                if not is_test:
                                    hidden_states_collected.append(hidden_states)
                                    if (idx + 1) % save_step_size == 0:
                                        np.save(
                                            self.sub_hidden_states_path.format(
                                                split,
                                                (idx + 1) / save_step_size,
                                                combination_string,
                                            ),
                                            np.concatenate(hidden_states_collected),
                                        )
                                        hidden_states_collected = []
                                else:
                                    assert torch.allclose(
                                        test_hidden_states, hidden_states, atol=1e-8
                                    ), f"Expected hidden states to be same, but are not: {hidden_states[0,0,7,:7]}, {test_hidden_states[0,0,7,:7]}"
                            if not tokens_collected:
                                tokens = get_tokens_as_df_cb(
                                    batch["input_ids"],
                                    self.tokenizer,
                                    semantic_positional_encoding,
                                )
                                tokens["labels"] = batch["labels"].tolist()
                                tokens["split"] = splits_
                                all_tokens.append(tokens)
                        if not tokens_collected:
                            # Concatenate all hidden states across batches
                            all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                            if not is_test:
                                all_tokens.to_csv(
                                    self.sub_tokens_path,
                                    index=False,
                                )
                            del all_tokens
                            tokens_collected = True

                        if add_logits:
                            logits_of_combination = np.concatenate(
                                logits_of_combination
                            )
                            np.save(
                                self.sub_logits_path.format(split, combination_string),
                                logits_of_combination,
                            )

    @staticmethod
    def read_forward_dataset(root: str, splits: List[str] = ["train", "test", "val"]):
        tokens_path = f"{root}/tokens.csv"
        hidden_states_path = f"{root}/hidden_states.npy"
        attentions_path = f"{root}/attentions.npy"
        all_tokens = pd.read_csv(tokens_path)
        all_hidden_states = np.load(hidden_states_path)
        all_tokens["hidden_states"] = list(all_hidden_states)
        all_attentions = np.load(attentions_path)
        all_tokens["attentions"] = list(all_attentions)
        all_tokens = all_tokens[all_tokens["split"].isin(splits)]
        return all_tokens


class BertClassifierOriginalArchitectureBase(ClassifierBase):
    def __init__(
        self,
        df,
        tokenizer,
        train_data_collator,
        test_data_collator,
        val_data_collator,
        model_max_length,
        semantic_datapoints,
        root_path,
        model_name,
        force_recompute=False,
    ) -> None:
        best_model_path = f"{root_path}/training/best"
        # Initialize the model and tokenizer
        if os.path.exists(best_model_path) and not force_recompute:
            model = BertForSequenceClassificationRanges.from_pretrained(
                best_model_path, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
            )
        else:
            model = BertForSequenceClassificationRanges.from_pretrained(
                model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
            )
        super().__init__(
            df=df,
            model=model,
            tokenizer=tokenizer,
            train_data_collator=train_data_collator,
            test_data_collator=test_data_collator,
            val_data_collator=val_data_collator,
            model_max_length=model_max_length,
            semantic_datapoints=semantic_datapoints,
            root_path=root_path,
            force_recompute=force_recompute,
        )

    def tokenize_function(self, example, return_pt=False):
        tokenized = self.tokenizer(
            example["prompt"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        semantic_positional_encoding = get_semantic_positional_encoding(
            tokenized["input_ids"], self.tokenizer.sep_token_id
        )
        semantic_positional_encoding = sort_ranges(semantic_positional_encoding)
        if return_pt:
            result = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": example["labels"],
                "semantic_positional_encoding": semantic_positional_encoding,
            }
        else:
            result = {
                "input_ids": tokenized["input_ids"].detach().to("cpu").tolist(),
                "attention_mask": tokenized["attention_mask"]
                .detach()
                .to("cpu")
                .tolist(),
                "labels": example["labels"],
                "semantic_positional_encoding": semantic_positional_encoding.detach()
                .to("cpu")
                .tolist(),
            }
        return result


class EmbeddingBasedClassifier(ClassifierBase):
    def __init__(
        self,
        kge_manager,
        get_embeddings_cb,
        root_path: str,
        model: BertForSequenceClassification,
        model_name: str,
        model_max_length=256,
        false_ratio=0.5,
        force_recompute=False,
    ) -> None:
        tokenizer = BertTokenizer.from_pretrained(
            model_name, model_max_length=model_max_length
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device", self.device)
        train_data_collator = EmbeddingBasedDataCollator(
            tokenizer,
            self.device,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_train_data,
            get_embeddings_cb,
            false_ratio=false_ratio,
        )
        test_data_collator = EmbeddingBasedDataCollator(
            tokenizer,
            self.device,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_test_data,
            get_embeddings_cb,
            false_ratio=-1.0,
        )
        val_data_collator = EmbeddingBasedDataCollator(
            tokenizer,
            self.device,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_val_data,
            get_embeddings_cb,
            false_ratio=-1.0,
        )
        ClassifierBase.__init__(
            self,
            df=kge_manager.llm_df,
            model=model,
            tokenizer=tokenizer,
            train_data_collator=train_data_collator,
            test_data_collator=test_data_collator,
            val_data_collator=val_data_collator,
            model_max_length=model_max_length,
            semantic_datapoints=EMBEDDINGS_BASED_SEMANTIC_TOKENS,
            root_path=root_path,
            force_recompute=force_recompute,
        )

    def tokenize_function(self, example, return_pt=False):
        tokenized = self.tokenizer(
            example["prompt"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        semantic_positional_encoding = get_semantic_positional_encoding(
            tokenized["input_ids"], self.tokenizer.sep_token_id
        )
        semantic_positional_encoding = sort_ranges(semantic_positional_encoding)
        if return_pt:
            result = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": example["labels"],
                "semantic_positional_encoding": semantic_positional_encoding,
                "source_id": example["source_id"],
                "target_id": example["target_id"],
            }
        else:
            result = {
                "input_ids": tokenized["input_ids"].detach().to("cpu").tolist(),
                "attention_mask": tokenized["attention_mask"]
                .detach()
                .to("cpu")
                .tolist(),
                "labels": example["labels"],
                "semantic_positional_encoding": semantic_positional_encoding.detach()
                .to("cpu")
                .tolist(),
                "source_id": example["source_id"],
                "target_id": example["target_id"],
            }
        return result


class InputEmbedsReplaceClassifier(EmbeddingBasedClassifier):
    def __init__(
        self,
        kge_manager,
        get_embeddings_cb,
        root_path,
        model_name=MODEL_NAME,
        model_max_length=256,
        false_ratio: float = 0.5,
        force_recompute=False,
    ) -> None:
        training_path = f"{root_path}/training"
        model_path = f"{training_path}/best"

        if os.path.exists(model_path) and not force_recompute:
            model = InputEmbedsReplaceBertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
        else:
            model = InputEmbedsReplaceBertForSequenceClassification.from_pretrained(
                model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
            )
        assert isinstance(model, BertForSequenceClassification)
        super().__init__(
            kge_manager,
            get_embeddings_cb,
            root_path,
            model,
            model_name,
            model_max_length,
            false_ratio,
            force_recompute,
        )

    def plot_training_loss_and_accuracy(self):
        model_type = "Input Embeds Replace"
        self._plot_training_loss_and_accuracy(model_type)


class PromptBertClassifier(BertClassifierOriginalArchitectureBase):
    def __init__(
        self,
        kge_manager,
        get_embeddings_cb,
        root_path,
        model_name=MODEL_NAME,
        model_max_length=256,
        false_ratio=0.5,
        force_recompute=False,
    ) -> None:
        tokenizer = BertTokenizer.from_pretrained(
            model_name, model_max_length=model_max_length
        )
        train_data_collator = PromptEmbeddingDataCollator(
            tokenizer,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_train_data,
            get_embeddings_cb,
            false_ratio=false_ratio,
        )
        test_data_collator = PromptEmbeddingDataCollator(
            tokenizer,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_test_data,
            get_embeddings_cb,
            false_ratio=-1.0,
        )
        val_data_collator = PromptEmbeddingDataCollator(
            tokenizer,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_val_data,
            get_embeddings_cb,
            false_ratio=-1.0,
        )
        super().__init__(
            df=kge_manager.llm_df,
            tokenizer=tokenizer,
            train_data_collator=train_data_collator,
            test_data_collator=test_data_collator,
            val_data_collator=val_data_collator,
            model_max_length=model_max_length,
            semantic_datapoints=EMBEDDINGS_BASED_SEMANTIC_TOKENS,
            root_path=root_path,
            model_name=model_name,
            force_recompute=force_recompute,
        )

    def plot_training_loss_and_accuracy(self):
        model_type = "Prompt"
        self._plot_training_loss_and_accuracy(model_type)


class VanillaBertClassifier(BertClassifierOriginalArchitectureBase):
    def __init__(
        self,
        df,
        source_df,
        target_df,
        root_path,
        model_name=MODEL_NAME,
        model_max_length=256,
        false_ratio=0.5,
        force_recompute=False,
    ) -> None:
        tokenizer = BertTokenizer.from_pretrained(
            MODEL_NAME, model_max_length=model_max_length
        )
        train_data_collator = VanillaEmbeddingDataCollator(
            tokenizer, df, source_df, target_df, false_ratio=false_ratio
        )
        val_data_collator = VanillaEmbeddingDataCollator(
            tokenizer, df, source_df, target_df, false_ratio=-1.0
        )
        super().__init__(
            df=df,
            tokenizer=tokenizer,
            train_data_collator=train_data_collator,
            test_data_collator=val_data_collator,
            val_data_collator=val_data_collator,
            model_max_length=model_max_length,
            semantic_datapoints=ALL_SEMANTIC_TOKENS,
            root_path=root_path,
            model_name=model_name,
            force_recompute=force_recompute,
        )

    def plot_training_loss_and_accuracy(self):
        model_type = "Vanilla"
        self._plot_training_loss_and_accuracy(model_type)
