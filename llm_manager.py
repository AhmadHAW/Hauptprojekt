import random as rd
from typing import Optional, Union, Dict, Tuple, List, Callable
from abc import ABC, abstractmethod
import os
import json
from pathlib import Path

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import datasets
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
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import is_datasets_available
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataset_manager import (
    LLM_PROMPT_TRAINING_PATH,
    LLM_ATTENTION_TRAINING_PATH,
    LLM_VANILLA_TRAINING_PATH,
    LLM_PROMPT_BEST_MODEL_PATH,
    LLM_ATTENTION_BEST_MODEL_PATH,
    LLM_VANILLA_BEST_MODEL_PATH,
    LLM_VANILLA_PATH,
    LLM_PROMPT_PATH,
    LLM_ATTENTION_PATH,
)
from utils import (
    find_non_existing_source_target,
    mean_over_ranges,
    row_to_vanilla_datapoint,
    row_to_prompt_datapoint,
    row_to_attention_datapoint,
)

ID2LABEL = {0: "FALSE", 1: "TRUE"}
LABEL2ID = {"FALSE": 0, "TRUE": 1}
PROMPT_LOG_PATH = f"{LLM_PROMPT_TRAINING_PATH}/logs"
EMBEDDING_LOG_PATH = f"{LLM_ATTENTION_TRAINING_PATH}/logs"
VANILLA_LOG_PATH = f"{LLM_VANILLA_TRAINING_PATH}/logs"

PROMPT_TRAINING_STATE_PATH = (
    f"{LLM_PROMPT_TRAINING_PATH}/checkpoint-4420/trainer_state.json"
)
EMBEDDING_TRAINING_STATE_PATH = (
    f"{LLM_ATTENTION_TRAINING_PATH}/checkpoint-4420/trainer_state.json"
)
VANILLA_TRAINING_STATE_PATH = (
    f"{LLM_VANILLA_TRAINING_PATH}/checkpoint-4420/trainer_state.json"
)

VANILLA_ATTENTIONS_PATH = f"{LLM_VANILLA_PATH}/attentions.npy"
PROMPT_ATTENTIONS_PATH = f"{LLM_PROMPT_PATH}/attentions.npy"
EMBEDDING_ATTENTIONS_PATH = f"{LLM_ATTENTION_PATH}/attentions.npy"
VANILLA_HIDDEN_STATES_PATH = f"{LLM_VANILLA_PATH}/hidden_states.npy"
PROMPT_HIDDEN_STATES_PATH = f"{LLM_PROMPT_PATH}/hidden_states.npy"
EMBEDDING_HIDDEN_STATES_PATH = f"{LLM_ATTENTION_PATH}/hidden_states.npy"
PROMPT_GRAPH_EMBEDDINGS_PATH = f"{LLM_PROMPT_PATH}/graph_embeddings.npy"
EMBEDDING_GRAPH_EMBEDDINGS_PATH = f"{LLM_ATTENTION_PATH}/graph_embeddings.npy"
VANILLA_TOKENS_PATH = f"{LLM_VANILLA_PATH}/tokens.csv"
PROMPT_TOKENS_PATH = f"{LLM_PROMPT_PATH}/tokens.csv"
EMBEDDING_TOKENS_PATH = f"{LLM_ATTENTION_PATH}/tokens.csv"

SPLIT_EPOCH_ENDING = f"/split_{{}}_epoch_{{}}.npy"
TOKENS_ENDING = f"/tokens_{{}}_{{}}.csv"

VANILLA_HIDDEN_STATES_DIR_PATH = f"{LLM_VANILLA_PATH}/hidden_states"
HIDDEN_STATES_VANILLA_PATH = f"{VANILLA_HIDDEN_STATES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

VANILLA_RANGES_DIR_PATH = f"{LLM_VANILLA_PATH}/ranges"
RANGES_VANILLA_PATH = f"{VANILLA_RANGES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

VANILLA_ATTENTIONS_DIR_PATH = f"{LLM_VANILLA_PATH}/attentions"
ATTENTIONS_VANILLA_PATH = f"{VANILLA_ATTENTIONS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

VANILLA_INPUT_IDS_DIR_PATH = f"{LLM_VANILLA_PATH}/input_ids"
INPUT_IDS_VANILLA_PATH = f"{VANILLA_INPUT_IDS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

VANILLA_SUB_TOKENS_DIR_PATH = f"{LLM_VANILLA_PATH}/tokens"
VANILLA_SUB_TOKENS_PATH = f"{VANILLA_SUB_TOKENS_DIR_PATH}{TOKENS_ENDING}"

PROMPT_HIDDEN_STATES_DIR_PATH = f"{LLM_PROMPT_PATH}/hidden_states"
HIDDEN_STATES_PROMPT_PATH = f"{PROMPT_HIDDEN_STATES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

PROMPT_RANGES_DIR_PATH = f"{LLM_PROMPT_PATH}/ranges"
RANGES_PROMPT_PATH = f"{PROMPT_RANGES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

PROMPT_ATTENTIONS_DIR_PATH = f"{LLM_PROMPT_PATH}/attentions"
ATTENTIONS_PROMPT_PATH = f"{PROMPT_ATTENTIONS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

PROMPT_INPUT_IDS_DIR_PATH = f"{LLM_PROMPT_PATH}/input_ids"
INPUT_IDS_PROMPT_PATH = f"{PROMPT_INPUT_IDS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

PROMPT_GRAPH_EMBEDDINGS_DIR_PATH = f"{LLM_PROMPT_PATH}/graph_embeddings"
GRAPH_EMBEDDINGS_PROMPT_PATH = f"{PROMPT_GRAPH_EMBEDDINGS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

PROMPT_SUB_TOKENS_DIR_PATH = f"{LLM_PROMPT_PATH}/tokens"
PROMPT_SUB_TOKENS_PATH = f"{PROMPT_SUB_TOKENS_DIR_PATH}{TOKENS_ENDING}"

EMBEDDING_HIDDEN_STATES_DIR_PATH = f"{LLM_ATTENTION_PATH}/hidden_states"
HIDDEN_STATES_EMBEDDING_PATH = f"{EMBEDDING_HIDDEN_STATES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

EMBEDDING_RANGES_DIR_PATH = f"{LLM_ATTENTION_PATH}/ranges"
RANGES_EMBEDDING_PATH = f"{EMBEDDING_RANGES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

EMBEDDING_ATTENTIONS_DIR_PATH = f"{LLM_ATTENTION_PATH}/attentions"
ATTENTIONS_EMBEDDING_PATH = f"{EMBEDDING_ATTENTIONS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

EMBEDDING_INPUT_IDS_DIR_PATH = f"{LLM_ATTENTION_PATH}/input_ids"
INPUT_IDS_EMBEDDING_PATH = f"{EMBEDDING_INPUT_IDS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

EMBEDDING_GRAPH_EMBEDDINGS_DIR_PATH = f"{LLM_ATTENTION_PATH}/graph_embeddings"
GRAPH_EMBEDDINGS_EMBEDDING_PATH = (
    f"{EMBEDDING_GRAPH_EMBEDDINGS_DIR_PATH}{SPLIT_EPOCH_ENDING}"
)

EMBEDDING_SUB_TOKENS_DIR_PATH = f"{LLM_ATTENTION_PATH}/tokens"
EMBEDDING_SUB_TOKENS_PATH = f"{EMBEDDING_SUB_TOKENS_DIR_PATH}{TOKENS_ENDING}"

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
ALL_SEMANTIC_TOKENS = ["cls", "user", "sep1", "title", "sep2", "genres", "sep3"]
EMBEDDING_BASED_SEMANTIC_TOKENS = ALL_SEMANTIC_TOKENS + [
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

    def __init__(self, tokenizer, df, source_df, target_df, false_ratio=2.0):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.false_ratio = false_ratio
        self.tokenizer = tokenizer
        self.df = df
        self.source_df = source_df
        self.target_df = target_df

    def __call__(self, features):
        new_features = []
        for feature in features:
            # Every datapoint has a chance to be replaced by a negative datapoint, based on the false_ratio.
            # The _transform_to_false_exmample methods have to be implemented by the inheriting class.
            # For the prompt classifier, every new datapoint also contains embeddings of the nodes.
            # If the false ratio is -1 this step will be skipped (for validation)
            if self.false_ratio != -1 and rd.uniform(0, 1) >= (
                1 / (self.false_ratio + 1)
            ):
                new_feature = self._transform_to_false_example()
                new_features.append(new_feature)
            else:
                new_features.append(feature)
        # Convert features into batches
        return self._convert_features_into_batches(new_features)

    @abstractmethod
    def _transform_to_false_example(self) -> Dict:
        pass

    @abstractmethod
    def _convert_features_into_batches(self, features: List[Dict]) -> Dict:
        pass


class TextBasedDataCollator(DataCollatorBase, ABC):
    def __init__(self, tokenizer, df, source_df, target_df, false_ratio=2.0):
        super().__init__(tokenizer, df, source_df, target_df, false_ratio=false_ratio)

    def _convert_features_into_batches(self, features: List[Dict]) -> Dict:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor(
            [f["attention_mask"] for f in features], dtype=torch.long
        )
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        semantic_positional_encoding = torch.tensor(
            [f["semantic_positional_encoding"] for f in features], dtype=torch.long
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
        get_embedding_cb,
        false_ratio=2.0,
    ):
        super().__init__(tokenizer, df, source_df, target_df, false_ratio=false_ratio)
        self.device = device
        self.data = data
        self.get_embedding_cb = get_embedding_cb

    def _convert_features_into_batches(self, features: List[Dict]) -> Dict:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor(
            [f["attention_mask"] for f in features], dtype=torch.long
        )
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        semantic_positional_encoding = torch.tensor(
            [f["semantic_positional_encoding"] for f in features], dtype=torch.long
        )
        for f in features:
            if isinstance(f["graph_embeddings"], list):
                f["graph_embeddings"] = torch.tensor(f["graph_embeddings"])
            else:
                f["graph_embeddings"] = f["graph_embeddings"]
        graph_embeddings = torch.stack(
            [f["graph_embeddings"].detach().to("cpu") for f in features]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "graph_embeddings": graph_embeddings,
            "semantic_positional_encoding": semantic_positional_encoding,
        }

    def _transform_to_false_example(self) -> Dict:
        label = 0
        source_id, target_id = find_non_existing_source_target(self.df)

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

        source_embedding, target_embedding = self.get_embedding_cb(
            self.data, source_id, target_id
        )
        random_row["prompt"] = row_to_attention_datapoint(
            random_row, self.tokenizer.sep_token, self.tokenizer.pad_token
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
        semantic_positional_encoding = (
            sort_ranges(semantic_positional_encoding)
            .squeeze(0)
            .to("cpu")
            .detach()
            .tolist()
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label,
            "graph_embeddings": torch.stack([source_embedding, target_embedding]),
            "semantic_positional_encoding": semantic_positional_encoding,
        }


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
        get_embedding_cb,
        false_ratio=2.0,
    ):
        super().__init__(tokenizer, df, source_df, target_df, false_ratio=false_ratio)
        self.data = data
        self.get_embedding_cb = get_embedding_cb

    def _transform_to_false_example(self) -> Dict:
        label = 0
        source_id, target_id = find_non_existing_source_target(self.df)

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
        source_embedding, target_embedding = self.get_embedding_cb(
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

    def __init__(self, tokenizer, df, source_df, target_df, false_ratio=2.0):
        super().__init__(tokenizer, df, source_df, target_df, false_ratio=false_ratio)

    def _transform_to_false_example(self) -> Dict:
        label = 0
        source_id, target_id = find_non_existing_source_target(self.df)
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
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
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


class EmbeddingBertForSequenceClassification(BertForSequenceClassification):
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
            inputs_embeds = self.bert.embeddings(input_ids)
            assert isinstance(inputs_embeds, torch.Tensor)
        if graph_embeddings is not None and len(graph_embeddings) > 0:
            if attention_mask is not None:
                mask = (
                    (
                        (attention_mask.to(self.device).sum(dim=1) - 1)
                        .unsqueeze(1)
                        .repeat((1, 2))
                        - torch.tensor([3, 1], device=self.device)
                    )
                    .unsqueeze(2)
                    .repeat((1, 1, self.config.hidden_size))
                )
                inputs_embeds = inputs_embeds.to(self.device).scatter(
                    1, mask.to(self.device), graph_embeddings.to(self.device)
                )
        outputs = self.bert(
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


class ClassifierBase(ABC):
    def __init__(
        self,
        df,
        semantic_datapoints,
        best_model_path,
        attentions_path,
        hidden_states_path,
        tokens_path,
        force_recompute=False,
    ) -> None:
        self.predictions = None
        self.df = df
        self.semantic_datapoints = semantic_datapoints
        self.best_model_path = best_model_path
        self.attentions_path = attentions_path
        self.hidden_states_path = hidden_states_path
        self.tokens_path = tokens_path
        self.force_recompute = force_recompute
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return METRIC.compute(predictions=predictions, references=labels)

    @abstractmethod
    def _get_trainer(
        self,
        dataset,
        tokenize=False,
        eval_data_collator=None,
        epochs=3,
        batch_size: int = 64,
    ) -> Trainer:
        pass

    def train_model_on_data(self, dataset, epochs=3, batch_size: int = 64):
        trainer = self._get_trainer(dataset, epochs=epochs, batch_size=batch_size)

        # Train the model
        trainer.train()

        trainer.model.to(device="cpu").save_pretrained(self.best_model_path)  # type: ignore
        trainer.model.to(device=self.device)  # type: ignore

    def _plot_training_loss_and_accuracy(self, path_to_trainer_state, model_type):
        with open(path_to_trainer_state, "r") as f:
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

    def _avg_over_hidden_states(
        self,
        all_semantic_positional_encoding: torch.Tensor,
        last_hidden_states: torch.Tensor,
    ):
        averaged_hidden_states = []
        for position in range(all_semantic_positional_encoding.shape[1]):
            semantic_positional_encoding = all_semantic_positional_encoding[:, position]
            starts = semantic_positional_encoding[:, 0]
            ends = semantic_positional_encoding[:, 1]
            averaged_hidden_state = mean_over_ranges(last_hidden_states, starts, ends)
            averaged_hidden_states.append(averaged_hidden_state)
        return torch.stack(averaged_hidden_states)

    def _means_over_ranges_cross(
        self,
        all_semantic_positional_encoding: torch.Tensor,
        all_attentions: torch.Tensor,
    ) -> torch.Tensor:
        all_attentios_avgs = torch.zeros(
            all_semantic_positional_encoding.shape[0],
            all_semantic_positional_encoding.shape[1],
            all_semantic_positional_encoding.shape[1],
            all_attentions.shape[-1],
        )
        for batch, (batch_range, batch_attention) in enumerate(
            zip(all_semantic_positional_encoding, all_attentions)
        ):
            for from_, delimiter_from in enumerate(batch_range):
                for to_, delimiter_to in enumerate(batch_range):
                    batch_attention_sliced = batch_attention[
                        delimiter_from[0] : delimiter_from[1]
                    ][:, delimiter_to[0] : delimiter_to[1]]
                    attention_avg = torch.mean(batch_attention_sliced, dim=(0, 1))
                    all_attentios_avgs[batch, from_, to_] = attention_avg
        return all_attentios_avgs


class BertClassifierOriginalArchitectureBase(ClassifierBase):
    def __init__(
        self,
        df,
        semantic_datapoints,
        best_model_path,
        attentions_path,
        hidden_states_path,
        tokens_path,
        model_max_length=256,
        force_recompute=False,
    ) -> None:
        super().__init__(
            df=df,
            semantic_datapoints=semantic_datapoints,
            best_model_path=best_model_path,
            attentions_path=attentions_path,
            hidden_states_path=hidden_states_path,
            tokens_path=tokens_path,
            force_recompute=force_recompute,
        )

        # Initialize the model and tokenizer
        if os.path.exists(self.best_model_path) and not self.force_recompute:
            self.model = BertForSequenceClassificationRanges.from_pretrained(
                self.best_model_path, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
            )
        else:
            self.model = BertForSequenceClassificationRanges.from_pretrained(
                MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
            )

        self.tokenizer = BertTokenizer.from_pretrained(
            MODEL_NAME, model_max_length=model_max_length
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


class AttentionBertClassifierBase(ClassifierBase):
    def __init__(
        self,
        kge_manager,
        get_embedding_cb,
        model_max_length=256,
        false_ratio=2.0,
        force_recompute=False,
    ) -> None:
        best_model_path = LLM_ATTENTION_BEST_MODEL_PATH
        attentions_path = EMBEDDING_ATTENTIONS_PATH
        hidden_states_path = EMBEDDING_HIDDEN_STATES_PATH
        self.graph_embeddings_path = EMBEDDING_GRAPH_EMBEDDINGS_PATH
        tokens_path = EMBEDDING_TOKENS_PATH
        super().__init__(
            df=kge_manager.llm_df,
            semantic_datapoints=EMBEDDING_BASED_SEMANTIC_TOKENS,
            best_model_path=best_model_path,
            attentions_path=attentions_path,
            hidden_states_path=hidden_states_path,
            tokens_path=tokens_path,
            force_recompute=force_recompute,
        )

        # Initialize the model and tokenizer
        if os.path.exists(self.best_model_path) and not self.force_recompute:
            self.model = EmbeddingBertForSequenceClassification.from_pretrained(
                self.best_model_path, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
            )
        else:
            self.model = EmbeddingBertForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
            )

        self.tokenizer = BertTokenizer.from_pretrained(
            MODEL_NAME, model_max_length=model_max_length
        )
        self.train_data_collator = EmbeddingBasedDataCollator(
            self.tokenizer,
            self.device,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_train_data,
            get_embedding_cb,
            false_ratio=false_ratio,
        )
        self.test_data_collator = EmbeddingBasedDataCollator(
            self.tokenizer,
            self.device,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_test_data,
            get_embedding_cb,
            false_ratio=-1.0,
        )
        self.eval_data_collator = EmbeddingBasedDataCollator(
            self.tokenizer,
            self.device,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_val_data,
            get_embedding_cb,
            false_ratio=-1.0,
        )

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
            output_dir=LLM_ATTENTION_TRAINING_PATH,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=EMBEDDING_LOG_PATH,
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
                "graph_embeddings": example["graph_embeddings"],
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
                "graph_embeddings": example["graph_embeddings"],
            }
        return result

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
                eval_data_collator=self.eval_data_collator,
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

    def _get_data_collator(self, split):
        return (
            self.test_data_collator
            if split == "test"
            else self.eval_data_collator
            if split == "val"
            else self.train_data_collator
        )

    def plot_training_loss_and_accuracy(self):
        model_type = "Embedding"
        self._plot_training_loss_and_accuracy(EMBEDDING_TRAINING_STATE_PATH, model_type)

    def forward_dataset_and_save_outputs(
        self,
        dataset: datasets.Dataset | datasets.DatasetDict,
        get_tokens_as_df_cb: Callable,
        splits: List[str] = ["train", "test", "val"],
        batch_size: int = 64,
        epochs: int = 1,
        load_fields: List[str] = ["attentions", "hidden_states"],
        force_recompute: bool = False,
    ):
        if (
            force_recompute
            or not os.path.exists(self.attentions_path)
            or not os.path.exists(self.hidden_states_path)
            or not os.path.exists(self.tokens_path)
            or not os.path.exists(self.graph_embeddings_path)
        ):
            assert isinstance(self.model, EmbeddingBertForSequenceClassification)
            self.model.eval()
            Path(EMBEDDING_ATTENTIONS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(EMBEDDING_HIDDEN_STATES_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(EMBEDDING_INPUT_IDS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(EMBEDDING_RANGES_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(EMBEDDING_SUB_TOKENS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(EMBEDDING_GRAPH_EMBEDDINGS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            add_hidden_states = "hidden_states" in load_fields
            add_attentions = "attentions" in load_fields
            add_graph_embeddings = "graph_embeddings" in load_fields
            with torch.no_grad():
                for split in splits:
                    data_collator = self._get_data_collator(split)
                    for epoch in range(epochs):
                        all_hidden_states = []
                        all_attentions = []
                        all_graph_embeddings = []
                        all_tokens = []
                        print(f"Adding {split} Forward Epoch {epoch + 1} from {epochs}")
                        data_loader = DataLoader(
                            dataset=dataset[split],
                            batch_size=batch_size,
                            collate_fn=data_collator,
                        )
                        for idx, batch in enumerate(data_loader):
                            # if True:
                            #    batch = next(iter(data_loader))
                            splits_ = [split] * len(batch["input_ids"])
                            outputs = self.model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                graph_embeddings=batch["graph_embeddings"],
                                semantic_positional_encoding=batch[
                                    "semantic_positional_encoding"
                                ],
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
                                attentions = self._means_over_ranges_cross(
                                    semantic_positional_encoding, attentions
                                ).numpy()
                                all_attentions.append(attentions)
                            if add_hidden_states:
                                hidden_states_on_each_layer = []
                                for hidden_states in outputs.hidden_states:
                                    hidden_states_on_layer = (
                                        self._avg_over_hidden_states(
                                            semantic_positional_encoding, hidden_states
                                        )
                                        .permute((1, 0, 2))
                                        .numpy()
                                    )
                                    hidden_states_on_each_layer.append(
                                        hidden_states_on_layer
                                    )
                                hidden_states_on_each_layer = np.stack(
                                    hidden_states_on_each_layer
                                )
                                all_hidden_states.append(hidden_states_on_each_layer)
                            tokens = get_tokens_as_df_cb(
                                batch["input_ids"],
                                self.tokenizer,
                                semantic_positional_encoding,
                            )
                            if add_graph_embeddings:
                                all_graph_embeddings.append(
                                    batch["graph_embeddings"].numpy()
                                )
                            tokens["labels"] = batch["labels"].tolist()
                            tokens["split"] = splits_
                            all_tokens.append(tokens)

                        # Concatenate all hidden states across batches
                        all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                        all_tokens.to_csv(
                            EMBEDDING_SUB_TOKENS_PATH.format(split, epoch),
                            index=False,
                        )
                        del all_tokens
                        if "attentions" in load_fields:
                            all_attentions = np.concatenate(all_attentions)
                            np.save(
                                ATTENTIONS_EMBEDDING_PATH.format(split, epoch),
                                all_attentions,
                            )
                            del all_attentions
                        if "hidden_states" in load_fields:
                            all_hidden_states = np.concatenate(
                                all_hidden_states, axis=1
                            )
                            np.save(
                                HIDDEN_STATES_EMBEDDING_PATH.format(split, epoch),
                                all_hidden_states,
                            )
                            del all_hidden_states
                        if "graph_embeddings" in load_fields:
                            all_graph_embeddings = np.concatenate(all_graph_embeddings)
                            np.save(
                                GRAPH_EMBEDDINGS_EMBEDDING_PATH.format(split, epoch),
                                all_graph_embeddings,
                            )
                            del all_graph_embeddings

                # all_tokens
                all_tokens = []
                for split in splits:
                    for epoch in range(epochs):
                        all_tokens.append(
                            pd.read_csv(EMBEDDING_SUB_TOKENS_PATH.format(split, epoch))
                        )
                all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                all_tokens.to_csv(self.tokens_path, index=False)

                # hidden states:
                if "hidden_states" in load_fields:
                    all_hidden_states = []
                    for split in splits:
                        for epoch in range(epochs):
                            all_hidden_states.append(
                                np.load(
                                    HIDDEN_STATES_EMBEDDING_PATH.format(split, epoch)
                                )
                            )
                    all_hidden_states = np.concatenate(all_hidden_states, axis=1)
                    np.save(self.hidden_states_path, all_hidden_states)
                    all_hidden_states = torch.from_numpy(all_hidden_states).permute(
                        (1, 0, 2, 3)
                    )
                    all_tokens["hidden_states"] = all_hidden_states.unbind()
                    del all_hidden_states

                # attentions:
                if "attentions" in load_fields:
                    attentions = []
                    for split in splits:
                        for epoch in range(epochs):
                            attentions.append(
                                np.load(ATTENTIONS_EMBEDDING_PATH.format(split, epoch))
                            )
                    attentions = np.concatenate(attentions)
                    np.save(self.attentions_path, attentions)
                    attentions = torch.from_numpy(attentions)
                    all_tokens["attentions"] = attentions.unbind()
                    del attentions

                # graph embeddings:
                if "graph_embeddings" in load_fields:
                    graph_embeddings = []
                    for split in splits:
                        for epoch in range(epochs):
                            graph_embeddings.append(
                                np.load(
                                    GRAPH_EMBEDDINGS_EMBEDDING_PATH.format(split, epoch)
                                )
                            )
                    graph_embeddings = np.concatenate(graph_embeddings)
                    np.save(self.graph_embeddings_path, graph_embeddings)
                    graph_embeddings = torch.from_numpy(graph_embeddings)
                    all_tokens["graph_embeddings"] = graph_embeddings.unbind()
                    del graph_embeddings
        else:
            all_tokens = pd.read_csv(self.tokens_path)
            if "hidden_states" in load_fields:
                averaged_hidden_states = torch.from_numpy(
                    np.load(self.hidden_states_path)
                )
                all_tokens["hidden_states"] = torch.unbind(averaged_hidden_states)
                del averaged_hidden_states
            if "attentions" in load_fields:
                averaged_attentions = torch.from_numpy(np.load(self.attentions_path))
                all_tokens["attentions"] = torch.unbind(averaged_attentions)
                del averaged_attentions
            if "graph_embeddings" in load_fields:
                graph_embeddings = torch.from_numpy(np.load(self.graph_embeddings_path))
                all_tokens["graph_embeddings"] = torch.unbind(graph_embeddings)
                del graph_embeddings
        all_tokens[all_tokens["split"].isin(splits)]
        return all_tokens


class PromptBertClassifier(BertClassifierOriginalArchitectureBase):
    def __init__(
        self,
        kge_manager,
        get_embedding_cb,
        model_max_length=256,
        false_ratio=2.0,
        force_recompute=False,
    ) -> None:
        best_model_path = LLM_PROMPT_BEST_MODEL_PATH
        attentions_path = PROMPT_ATTENTIONS_PATH
        hidden_states_path = PROMPT_HIDDEN_STATES_PATH
        self.graph_embeddings_path = PROMPT_GRAPH_EMBEDDINGS_PATH
        tokens_path = PROMPT_TOKENS_PATH
        super().__init__(
            df=kge_manager.llm_df,
            semantic_datapoints=EMBEDDING_BASED_SEMANTIC_TOKENS,
            best_model_path=best_model_path,
            attentions_path=attentions_path,
            hidden_states_path=hidden_states_path,
            tokens_path=tokens_path,
            force_recompute=force_recompute,
            model_max_length=model_max_length,
        )
        self.train_data_collator = PromptEmbeddingDataCollator(
            self.tokenizer,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_train_data,
            get_embedding_cb,
            false_ratio=false_ratio,
        )
        self.test_data_collator = PromptEmbeddingDataCollator(
            self.tokenizer,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_test_data,
            get_embedding_cb,
            false_ratio=-1.0,
        )
        self.eval_data_collator = PromptEmbeddingDataCollator(
            self.tokenizer,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            kge_manager.gnn_val_data,
            get_embedding_cb,
            false_ratio=-1.0,
        )

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
            output_dir=LLM_PROMPT_TRAINING_PATH,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=PROMPT_LOG_PATH,
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
                eval_data_collator=self.eval_data_collator,
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

    def plot_training_loss_and_accuracy(self):
        model_type = "Prompt"
        self._plot_training_loss_and_accuracy(PROMPT_TRAINING_STATE_PATH, model_type)

    def forward_dataset_and_save_outputs(
        self,
        dataset: datasets.Dataset | datasets.DatasetDict,
        get_tokens_as_df_cb: Callable,
        splits: List[str] = ["train", "test", "val"],
        batch_size: int = 64,
        epochs: int = 1,
        load_fields: List[str] = ["attentions", "hidden_states"],
        force_recompute: bool = False,
    ):
        if (
            force_recompute
            or not os.path.exists(self.attentions_path)
            or not os.path.exists(self.hidden_states_path)
            or not os.path.exists(self.tokens_path)
            or not os.path.exists(self.graph_embeddings_path)
        ):
            assert isinstance(self.model, BertForSequenceClassificationRanges)
            self.model.eval()
            Path(PROMPT_ATTENTIONS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_HIDDEN_STATES_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_INPUT_IDS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_RANGES_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_SUB_TOKENS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_GRAPH_EMBEDDINGS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            add_hidden_states = "hidden_states" in load_fields
            add_attentions = "attentions" in load_fields
            add_graph_embeddings = "graph_embeddings" in load_fields
            with torch.no_grad():
                for split in splits:
                    data_collator = self._get_data_collator(split)
                    for epoch in range(epochs):
                        all_hidden_states = []
                        all_attentions = []
                        all_graph_embeddings = []
                        all_tokens = []
                        print(f"Prompt {split} Forward Epoch {epoch + 1} from {epochs}")
                        data_loader = DataLoader(
                            dataset=dataset[split],
                            batch_size=batch_size,
                            collate_fn=data_collator,
                        )
                        for idx, batch in enumerate(data_loader):
                            # if True:
                            #    batch = next(iter(data_loader))
                            splits_ = [split] * len(batch["input_ids"])
                            outputs = self.model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                semantic_positional_encoding=batch[
                                    "semantic_positional_encoding"
                                ],
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
                                attentions = self._means_over_ranges_cross(
                                    semantic_positional_encoding, attentions
                                ).numpy()
                                all_attentions.append(attentions)
                            if add_hidden_states:
                                hidden_states_on_each_layer = []
                                for hidden_states in outputs.hidden_states:
                                    hidden_states_on_layer = (
                                        self._avg_over_hidden_states(
                                            semantic_positional_encoding, hidden_states
                                        )
                                        .permute((1, 0, 2))
                                        .numpy()
                                    )
                                    hidden_states_on_each_layer.append(
                                        hidden_states_on_layer
                                    )
                                hidden_states_on_each_layer = np.stack(
                                    hidden_states_on_each_layer
                                )
                                all_hidden_states.append(hidden_states_on_each_layer)
                            tokens, graph_embeddings = get_tokens_as_df_cb(
                                batch["input_ids"],
                                self.tokenizer,
                                semantic_positional_encoding,
                            )
                            if add_graph_embeddings:
                                all_graph_embeddings.append(graph_embeddings.numpy())
                            del graph_embeddings
                            tokens["labels"] = batch["labels"].tolist()
                            tokens["split"] = splits_
                            all_tokens.append(tokens)

                        # Concatenate all hidden states across batches
                        all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                        all_tokens.to_csv(
                            PROMPT_SUB_TOKENS_PATH.format(split, epoch),
                            index=False,
                        )
                        del all_tokens
                        if "attentions" in load_fields:
                            all_attentions = np.concatenate(all_attentions)
                            np.save(
                                ATTENTIONS_PROMPT_PATH.format(split, epoch),
                                all_attentions,
                            )
                            del all_attentions
                        if "hidden_states" in load_fields:
                            all_hidden_states = np.concatenate(
                                all_hidden_states, axis=1
                            )
                            np.save(
                                HIDDEN_STATES_PROMPT_PATH.format(split, epoch),
                                all_hidden_states,
                            )
                            del all_hidden_states
                        if "graph_embeddings" in load_fields:
                            all_graph_embeddings = np.concatenate(all_graph_embeddings)
                            np.save(
                                GRAPH_EMBEDDINGS_PROMPT_PATH.format(split, epoch),
                                all_graph_embeddings,
                            )
                            del all_graph_embeddings

                # all_tokens
                all_tokens = []
                for split in splits:
                    for epoch in range(epochs):
                        all_tokens.append(
                            pd.read_csv(PROMPT_SUB_TOKENS_PATH.format(split, epoch))
                        )
                all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                all_tokens.to_csv(self.tokens_path, index=False)

                # hidden states:
                if "hidden_states" in load_fields:
                    all_hidden_states = []
                    for split in splits:
                        for epoch in range(epochs):
                            all_hidden_states.append(
                                np.load(HIDDEN_STATES_PROMPT_PATH.format(split, epoch))
                            )
                    all_hidden_states = np.concatenate(all_hidden_states, axis=1)
                    np.save(self.hidden_states_path, all_hidden_states)
                    all_hidden_states = torch.from_numpy(all_hidden_states).permute(
                        (1, 0, 2, 3)
                    )
                    all_tokens["hidden_states"] = all_hidden_states.unbind()
                    del all_hidden_states

                # attentions:
                if "attentions" in load_fields:
                    attentions = []
                    for split in splits:
                        for epoch in range(epochs):
                            attentions.append(
                                np.load(ATTENTIONS_PROMPT_PATH.format(split, epoch))
                            )
                    attentions = np.concatenate(attentions)
                    np.save(self.attentions_path, attentions)
                    attentions = torch.from_numpy(attentions)
                    all_tokens["attentions"] = attentions.unbind()
                    del attentions

                # graph embeddings:
                if "graph_embeddings" in load_fields:
                    graph_embeddings = []
                    for split in splits:
                        for epoch in range(epochs):
                            graph_embeddings.append(
                                np.load(
                                    GRAPH_EMBEDDINGS_PROMPT_PATH.format(split, epoch)
                                )
                            )
                    graph_embeddings = np.concatenate(graph_embeddings)
                    np.save(self.graph_embeddings_path, graph_embeddings)
                    graph_embeddings = torch.from_numpy(graph_embeddings)
                    all_tokens["graph_embeddings"] = graph_embeddings.unbind()
                    del graph_embeddings
        else:
            all_tokens = pd.read_csv(self.tokens_path)
            if "hidden_states" in load_fields:
                averaged_hidden_states = torch.from_numpy(
                    np.load(self.hidden_states_path)
                )
                all_tokens["hidden_states"] = torch.unbind(averaged_hidden_states)
                del averaged_hidden_states
            if "attentions" in load_fields:
                averaged_attentions = torch.from_numpy(np.load(self.attentions_path))
                all_tokens["attentions"] = torch.unbind(averaged_attentions)
                del averaged_attentions
            if "graph_embeddings" in load_fields:
                graph_embeddings = torch.from_numpy(np.load(self.graph_embeddings_path))
                all_tokens["graph_embeddings"] = torch.unbind(graph_embeddings)
                del graph_embeddings
        all_tokens[all_tokens["split"].isin(splits)]
        return all_tokens

    def _get_data_collator(self, split):
        return (
            self.test_data_collator
            if split == "test"
            else self.eval_data_collator
            if split == "val"
            else self.train_data_collator
        )


class VanillaBertClassifier(BertClassifierOriginalArchitectureBase):
    def __init__(
        self,
        df,
        source_df,
        target_df,
        model_max_length=256,
        false_ratio=2.0,
        force_recompute=False,
    ) -> None:
        super().__init__(
            df=df,
            semantic_datapoints=ALL_SEMANTIC_TOKENS,
            best_model_path=LLM_VANILLA_BEST_MODEL_PATH,
            attentions_path=VANILLA_ATTENTIONS_PATH,
            hidden_states_path=VANILLA_HIDDEN_STATES_PATH,
            tokens_path=VANILLA_TOKENS_PATH,
            model_max_length=model_max_length,
            force_recompute=force_recompute,
        )
        self.train_data_collator = VanillaEmbeddingDataCollator(
            self.tokenizer, df, source_df, target_df, false_ratio=false_ratio
        )
        self.eval_data_collator = VanillaEmbeddingDataCollator(
            self.tokenizer, df, source_df, target_df, false_ratio=-1.0
        )

    def _get_trainer(
        self, dataset, tokenize: bool = False, epochs: int = 3, batch_size: int = 64
    ) -> Trainer:
        if tokenize:
            tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        else:
            tokenized_dataset = dataset
        training_args = TrainingArguments(
            output_dir=LLM_VANILLA_TRAINING_PATH,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=VANILLA_LOG_PATH,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
        )
        assert isinstance(self.model, BertForSequenceClassificationRanges)
        # Initialize the Trainer
        return CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=self.train_data_collator,
            eval_data_collator=self.eval_data_collator,
            compute_metrics=self._compute_metrics,  # type: ignore
        )

    def plot_confusion_matrix(
        self, split, dataset, tokenize=False, batch_size=64, force_recompute=False
    ):
        trainer = self._get_trainer(dataset, tokenize=tokenize, batch_size=batch_size)
        if split == "test":
            dataset = dataset["test"]
        else:
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

    def plot_training_loss_and_accuracy(self):
        model_type = "Vanilla"
        self._plot_training_loss_and_accuracy(VANILLA_TRAINING_STATE_PATH, model_type)

    def forward_dataset_and_save_outputs(
        self,
        dataset: datasets.Dataset | datasets.DatasetDict,
        get_tokens_as_df_cb: Callable,
        splits: List[str] = ["train", "test", "val"],
        batch_size: int = 64,
        epochs: int = 1,
        load_fields: List[str] = ["attentions", "hidden_states"],
        force_recompute: bool = False,
    ):
        if (
            force_recompute
            or not os.path.exists(self.attentions_path)
            or not os.path.exists(self.hidden_states_path)
            or not os.path.exists(self.tokens_path)
        ):
            assert isinstance(self.model, BertForSequenceClassificationRanges)
            self.model.eval()
            Path(VANILLA_ATTENTIONS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(VANILLA_HIDDEN_STATES_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(VANILLA_INPUT_IDS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(VANILLA_RANGES_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(VANILLA_SUB_TOKENS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            add_hidden_states = "hidden_states" in load_fields
            add_attentions = "attentions" in load_fields
            with torch.no_grad():
                for split in splits:
                    data_collator = self._get_data_collator(split)
                    for epoch in range(epochs):
                        all_hidden_states = []
                        all_attentions = []
                        all_tokens = []
                        print(
                            f"Vanilla {split} Forward Epoch {epoch + 1} from {epochs}"
                        )
                        data_loader = DataLoader(
                            dataset=dataset[split],
                            batch_size=batch_size,
                            collate_fn=data_collator,
                        )
                        for idx, batch in enumerate(data_loader):
                            # if True:
                            #    batch = next(iter(data_loader))
                            input_ids = batch["input_ids"]
                            splits_ = [split] * len(input_ids)
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=batch["attention_mask"],
                                semantic_positional_encoding=batch[
                                    "semantic_positional_encoding"
                                ],
                                output_hidden_states=add_hidden_states,
                                output_attentions=add_attentions,
                            )
                            semantic_positional_encoding = batch[
                                "semantic_positional_encoding"
                            ]
                            if "attentions" in load_fields:
                                attentions = outputs.attentions
                                attentions = [
                                    torch.sum(layer, dim=1) for layer in attentions
                                ]
                                attentions = torch.stack(attentions).permute(1, 2, 3, 0)
                                attentions = self._means_over_ranges_cross(
                                    semantic_positional_encoding, attentions
                                )
                                all_attentions.append(attentions.numpy())
                                del attentions
                            if "hidden_states" in load_fields:
                                hidden_states_on_each_layer = []
                                for hidden_states in outputs.hidden_states:
                                    hidden_states_on_layer = (
                                        self._avg_over_hidden_states(
                                            semantic_positional_encoding, hidden_states
                                        ).permute((1, 0, 2))
                                    )
                                    hidden_states_on_each_layer.append(
                                        hidden_states_on_layer.numpy()
                                    )
                                    del hidden_states
                                hidden_states_on_each_layer = np.stack(
                                    hidden_states_on_each_layer
                                )
                                all_hidden_states.append(hidden_states_on_each_layer)
                            tokens = get_tokens_as_df_cb(
                                self.tokenizer,
                                input_ids,
                                semantic_positional_encoding[:, [1, 3, 5]],
                            )
                            tokens["labels"] = batch["labels"].tolist()
                            tokens["split"] = splits_
                            all_tokens.append(tokens)

                        # Concatenate all hidden states across batches
                        all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                        all_tokens.to_csv(
                            VANILLA_SUB_TOKENS_PATH.format(split, epoch), index=False
                        )
                        del all_tokens
                        if "attentions" in load_fields:
                            all_attentions = np.concatenate(all_attentions)
                            np.save(
                                ATTENTIONS_VANILLA_PATH.format(split, epoch),
                                all_attentions,
                            )
                            del all_attentions
                        if "hidden_states" in load_fields:
                            all_hidden_states = np.concatenate(
                                all_hidden_states, axis=1
                            )
                            np.save(
                                HIDDEN_STATES_VANILLA_PATH.format(split, epoch),
                                all_hidden_states,
                            )
                            del all_hidden_states

                # all_tokens
                all_tokens = []
                for split in splits:
                    for epoch in range(epochs):
                        all_tokens.append(
                            pd.read_csv(VANILLA_SUB_TOKENS_PATH.format(split, epoch))
                        )
                all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                all_tokens.to_csv(self.tokens_path, index=False)

                # hidden states:
                if "hidden_states" in load_fields:
                    all_hidden_states = []
                    for split in splits:
                        for epoch in range(epochs):
                            all_hidden_states.append(
                                np.load(HIDDEN_STATES_VANILLA_PATH.format(split, epoch))
                            )
                    all_hidden_states = np.concatenate(all_hidden_states, axis=1)
                    np.save(self.hidden_states_path, all_hidden_states)
                    all_hidden_states = torch.from_numpy(all_hidden_states).permute(
                        (1, 0, 2, 3)
                    )
                    all_tokens["hidden_states"] = all_hidden_states.unbind()
                    del all_hidden_states

                # attentions:
                if "attentions" in load_fields:
                    attentions = []
                    for split in splits:
                        for epoch in range(epochs):
                            attentions.append(
                                np.load(ATTENTIONS_VANILLA_PATH.format(split, epoch))
                            )
                    attentions = np.concatenate(attentions)
                    np.save(self.attentions_path, attentions)
                    attentions = torch.from_numpy(attentions)
                    all_tokens["attentions"] = attentions.unbind()
                    del attentions

        else:
            all_tokens = pd.read_csv(self.tokens_path)
            if "hidden_states" in load_fields:
                averaged_hidden_states = torch.from_numpy(
                    np.load(self.hidden_states_path)
                )
                all_tokens["hidden_states"] = torch.unbind(averaged_hidden_states)
                del averaged_hidden_states
            if "attentions" in load_fields:
                averaged_attentions = torch.from_numpy(np.load(self.attentions_path))
                all_tokens["attentions"] = torch.unbind(averaged_attentions)
                del averaged_attentions
        all_tokens[all_tokens["split"].isin(splits)]
        return all_tokens

    def _get_data_collator(self, split: str = "train") -> VanillaEmbeddingDataCollator:
        if split == "train":
            return self.train_data_collator
        else:
            return self.eval_data_collator
