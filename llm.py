import random as rd
from typing import Optional, Union, Dict, Tuple, List
from abc import ABC, abstractmethod
import os
import json
import ast
from pathlib import Path

from movie_lens_loader import row_to_prompt_datapoint, row_to_adding_embedding_datapoint, row_to_vanilla_datapoint, LLM_PROMPT_TRAINING_PATH, LLM_ADDING_TRAINING_PATH, LLM_VANILLA_TRAINING_PATH, LLM_PROMPT_BEST_MODEL_PATH, LLM_ADDING_BEST_MODEL_PATH, LLM_VANILLA_BEST_MODEL_PATH, PCA_PATH, LLM_VANILLA_PATH, LLM_PROMPT_PATH, LLM_ADDING_PATH

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import ModelOutput
import datasets
import numpy as np
import pandas as pd
import evaluate
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertModel, BertTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import is_datasets_available
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
import joblib

ID2LABEL = {0: "FALSE", 1: "TRUE"}
LABEL2ID = {"FALSE": 0, "TRUE": 1}
PROMPT_LOG_PATH = f"{LLM_PROMPT_TRAINING_PATH}/logs"
ADDING_LOG_PATH = f"{LLM_ADDING_TRAINING_PATH}/logs"
VANILLA_LOG_PATH = f"{LLM_VANILLA_TRAINING_PATH}/logs"

PROMPT_TRAINING_STATE_PATH = f"{LLM_PROMPT_TRAINING_PATH}/checkpoint-4420/trainer_state.json"
ADDING_TRAINING_STATE_PATH = f"{LLM_ADDING_TRAINING_PATH}/checkpoint-4420/trainer_state.json"
VANILLA_TRAINING_STATE_PATH = f"{LLM_VANILLA_TRAINING_PATH}/checkpoint-4420/trainer_state.json"

PCA_VANILLA_MODEL_PATH = f"{PCA_PATH}/vanilla_{{}}.pkl"
PCA_PROMPT_MODEL_PATH = f"{PCA_PATH}/prompt_{{}}_{{}}.pkl"
PCA_EMBEDDING_MODEL_PATH = f"{PCA_PATH}/embedding_{{}}_{{}}.pkl"

VANILLA_ATTENTIONS_PATH = f"{LLM_VANILLA_PATH}/attentions.npy"
PROMPT_ATTENTIONS_PATH = f"{LLM_PROMPT_PATH}/attentions.npy"
ADDING_ATTENTIONS_PATH = f"{LLM_ADDING_PATH}/attentions.npy"
VANILLA_HIDDEN_STATES_PATH = f"{LLM_VANILLA_PATH}/hidden_states.npy"
PROMPT_HIDDEN_STATES_PATH = f"{LLM_PROMPT_PATH}/hidden_states.npy"
ADDING_HIDDEN_STATES_PATH = f"{LLM_ADDING_PATH}/hidden_states.npy"
PROMPT_GRAPH_EMBEDDINGS_PATH = f"{LLM_PROMPT_PATH}/graph_embeddings.npy"
ADDING_GRAPH_EMBEDDINGS_PATH = f"{LLM_ADDING_PATH}/graph_embeddings.npy"
VANILLA_TOKENS_PATH = f"{LLM_VANILLA_PATH}/tokens.csv"
PROMPT_TOKENS_PATH = f"{LLM_PROMPT_PATH}/tokens.csv"
ADDING_TOKENS_PATH = f"{LLM_ADDING_PATH}/tokens.csv"

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

ADDING_HIDDEN_STATES_DIR_PATH = f"{LLM_ADDING_PATH}/hidden_states"
HIDDEN_STATES_ADDING_PATH = f"{ADDING_HIDDEN_STATES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

ADDING_RANGES_DIR_PATH = f"{LLM_ADDING_PATH}/ranges"
RANGES_ADDING_PATH = f"{ADDING_RANGES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

ADDING_ATTENTIONS_DIR_PATH = f"{LLM_ADDING_PATH}/attentions"
ATTENTIONS_ADDING_PATH = f"{ADDING_ATTENTIONS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

ADDING_INPUT_IDS_DIR_PATH = f"{LLM_ADDING_PATH}/input_ids"
INPUT_IDS_ADDING_PATH = f"{ADDING_INPUT_IDS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

ADDING_GRAPH_EMBEDDINGS_DIR_PATH = f"{LLM_ADDING_PATH}/graph_embeddings"
GRAPH_EMBEDDINGS_ADDING_PATH = f"{ADDING_GRAPH_EMBEDDINGS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

ADDING_SUB_TOKENS_DIR_PATH = f"{LLM_ADDING_PATH}/tokens"
ADDING_SUB_TOKENS_PATH = f"{ADDING_SUB_TOKENS_DIR_PATH}{TOKENS_ENDING}"




ALL_SEMANTIC_TOKENS = ["cls", "user", "sep1", "title", "sep2", "genres", "sep3"]
EMBEDDING_BASED_SEMANTIC_TOKENS = ALL_SEMANTIC_TOKENS + ["user embedding", "sep4", "movie embedding", "sep5"]


def mean_over_ranges(tens: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    '''
    This operation allows to produce a mean over ranges of different sizes in torch tensor manner only.
    We use padding positions to calculate an average and then remove the padded positions afterwards.
    This code was based on ChatGPT suggestions and works on the assumption:
    Let S be the sum of the padded list of numbers.
    Let n be the number of elements in the padded list.
    Let μ be the average of the padded list of numbers.
    Let r be the difference between the actual range lengths and the max range length 
    Let x be the number that is at the padding position.
    μ' = ((μ * n)-(r * x)) / (n - r)
    '''
    # input: # ends: torch.tensor([2, 5, 6]) starts: tensor([0, 2, 4])
    # Compute the maximum length of the ranges
    max_length = (ends - starts).max().item()
    range_diffs = (max_length - (ends-starts)).unsqueeze(1) # the amount of times, the range had to be padded 
    # Create a range tensor from 0 to max_length-1
    range_tensor = torch.arange(max_length).unsqueeze(0)

    # Compute the ranges using broadcasting and masking
    ranges = starts.unsqueeze(1) + range_tensor
    mask = ranges < ends.unsqueeze(1)

    # Apply the mask
    result = ranges * mask  # result: tensor([[0, 1, 0], [2, 3, 4], [4, 5, 0]]) here padding index is 0
                            #                        -                     -    positions were padded
    result = result.unsqueeze(dim = 2).repeat(1,1, tens.shape[2])
    gather = tens.gather(dim = 1, index = result)
    means = torch.mean(gather, dim = 1) # The mean was computed with the padding positions. We will remove the values from the mean now,
    values_to_remove = range_diffs * tens[:,0] # the summed value at padded position
    actual_means = (means * max_length - values_to_remove) / (max_length - range_diffs) # the actual mean without padding positions
    return actual_means

def means_over_ranges_cross(all_ranges_over_batch: torch.Tensor, all_attentions: torch.Tensor) -> torch.Tensor:
    all_attentios_avgs = torch.zeros(all_ranges_over_batch.shape[0], all_ranges_over_batch.shape[1], all_ranges_over_batch.shape[1], all_attentions.shape[-1])
    for batch, (batch_range, batch_attention) in enumerate(zip(all_ranges_over_batch, all_attentions)):
        for from_, delimiter_from in enumerate(batch_range):
            for to_, delimiter_to in enumerate(batch_range):
                batch_attention_sliced = batch_attention[delimiter_from[0]:delimiter_from[1]][:,delimiter_to[0]:delimiter_to[1]]
                attention_avg = torch.mean(batch_attention_sliced, dim = (0,1))
                all_attentios_avgs[batch, from_, to_] = attention_avg
    return all_attentios_avgs

def avg_over_hidden_states(all_ranges_over_batch, last_hidden_states):
    averaged_hidden_states = []
    for position in range(all_ranges_over_batch.shape[1]):
        ranges_over_batch = all_ranges_over_batch[:,position]
        starts = ranges_over_batch[:,0]
        ends = ranges_over_batch[:,1]
        averaged_hidden_state = mean_over_ranges(last_hidden_states, starts, ends)
        averaged_hidden_states.append(averaged_hidden_state)
    return torch.stack(averaged_hidden_states)

def sort_ranges(ranges_over_batch):
    # Extract the second element (end of the current ranges excluded the starting cps token)
    end_elements = ranges_over_batch[:, :, 1]
    # Create the new ranges by adding 1 to the end elements
    new_ranges = torch.stack([end_elements, end_elements + 1], dim=-1)
    #add the cls positions to it
    cls_positions = torch.tensor([0, 1])
    cls_positions = cls_positions.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 2)
    cls_positions = cls_positions.expand(new_ranges.size(0), 1, -1)  # Shape (batch_size, 1, 2)
    new_ranges = torch.cat((new_ranges, cls_positions), dim=1)
    # Concatenate the original ranges with the new ranges
    ranges_over_batch = torch.cat((ranges_over_batch, new_ranges), dim=1)
    # Step 1: Extract the last value of dimension 2
    last_values = ranges_over_batch[:, :, -1]  # Shape (batch_size, num_elements)

    # Step 2: Sort the indices based on these last values
    # 'values' gives the sorted values (optional), 'indices' gives the indices to sort along dim 1
    _, indices = torch.sort(last_values, dim=1, descending=False)

    # Step 3: Apply the sorting indices to the original tensor
    ranges_over_batch = torch.gather(ranges_over_batch, 1, indices.unsqueeze(-1).expand(-1, -1, ranges_over_batch.size(2)))
    return ranges_over_batch

def find_sep_in_tokens(tokens, sep_token_id):
    return tokens == sep_token_id

def get_ranges_over_batch(input_ids, sep_token_id):
    mask = find_sep_in_tokens(input_ids, sep_token_id)
    positions = mask.nonzero(as_tuple=True)
    cols = positions[1]

    # Step 3: Determine the number of True values per row
    num_trues_per_row = mask.sum(dim=1)
    max_trues_per_row = num_trues_per_row.max().item()

    # Step 4: Create an empty tensor to hold the result
    ranges_over_batch = -torch.ones((mask.size(0), max_trues_per_row), dtype=torch.long)

    # Step 5: Use scatter to place column indices in the ranges_over_batch tensor
    # Create an index tensor that assigns each column index to the correct position in ranges_over_batch tensor
    row_indices = torch.arange(mask.size(0)).repeat_interleave(num_trues_per_row)
    column_indices = torch.cat([torch.arange(n) for n in num_trues_per_row])

    ranges_over_batch[row_indices, column_indices] = cols
    ranges_over_batch = torch.stack([ranges_over_batch[:, :-1]+1, ranges_over_batch[:, 1:]], dim=2)
    # Create a tensor of zeros to represent the starting points
    second_points = torch.ones(ranges_over_batch.size(0), 1, 2, dtype=ranges_over_batch.dtype)

    # Set the second column to be the first element of the first range
    second_points[:, 0, 1] = ranges_over_batch[:, 0, 0]-1
    # Concatenate the start_points tensor with the original ranges_over_batch tensor
    ranges_over_batch = torch.cat((second_points, ranges_over_batch), dim=1)
    return ranges_over_batch

class DataCollatorBase(DataCollatorForLanguageModeling, ABC):
    '''
    The Data Collators are used to generate non-existing edges on the fly. The false ratio allows to decide the ratio,
    in existing edges are replaced with non-existing edges.
    '''
    def __init__(self, tokenizer, df, false_ratio = 2.0):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.false_ratio = false_ratio
        self.tokenizer = tokenizer
        self.df = df

    def __call__(self, features):
        new_features = []
        for feature in features:
            #Every datapoint has a chance to be replaced by a negative datapoint, based on the false_ratio.
            #The _transform_to_false_exmample methods have to be implemented by the inheriting class.
            #For the prompt classifier, every new datapoint also contains embeddings of the nodes.
            #If the false ratio is -1 this step will be skipped (for validation)
            if self.false_ratio != -1 and rd.uniform(0, 1) >=( 1 / (self.false_ratio + 1)):
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
    
    def _find_non_existing_user_movie(self):
        while True:
            user_id = np.random.choice(self.df["mappedUserId"].unique())
            movie_id = np.random.choice(self.df["mappedMovieId"].unique())
            if not ((self.df["mappedUserId"] == user_id) & (self.df["mappedMovieId"] == movie_id)).any():
                return user_id, movie_id

class TextBasedDataCollator(DataCollatorBase, ABC):
    def __init__(self, tokenizer, df, false_ratio = 2.0):
        super().__init__(tokenizer=tokenizer,false_ratio = false_ratio, df = df)
    
    def _convert_features_into_batches(self, features: List[Dict]) -> Dict:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        ranges_over_batch = torch.tensor([f["ranges_over_batch"] for f in features], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ranges_over_batch": ranges_over_batch
        }


class EmbeddingBasedDataCollator(DataCollatorBase):
    def __init__(self, tokenizer, device, df, data, get_embedding_cb, kge_dimension = 128, false_ratio = 2.0):
        super().__init__(tokenizer=tokenizer,false_ratio = false_ratio, df = df)
        self.device = device
        self.data = data
        self.get_embedding_cb = get_embedding_cb
        self.kge_dimension = kge_dimension

    def _convert_features_into_batches(self, features: List[Dict]) -> Dict:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        ranges_over_batch = torch.tensor([f["ranges_over_batch"] for f in features], dtype=torch.long)
        for f in features:
            if isinstance(f["graph_embeddings"], list):
                f["graph_embeddings"] = torch.tensor(f["graph_embeddings"])
            else:
                f["graph_embeddings"] = f["graph_embeddings"]
        graph_embeddings = torch.stack([f["graph_embeddings"].detach().to("cpu") for f in features])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "graph_embeddings": graph_embeddings,
            "ranges_over_batch": ranges_over_batch
        }
    
    def _transform_to_false_example(self) -> Dict:
        label = 0
        user_id, movie_id = self._find_non_existing_user_movie()
        random_row = self.df[self.df["mappedMovieId"] == movie_id].sample(1).iloc[0]
        random_row["mappedUserId"] = user_id
        user_embedding, movie_embedding = self.get_embedding_cb(self.data, user_id, movie_id)
        random_row["prompt"] = row_to_adding_embedding_datapoint(random_row, self.tokenizer.sep_token, self.tokenizer.pad_token)
        tokenized = self.tokenizer(random_row["prompt"], padding="max_length", truncation=True)
        ranges_over_batch = get_ranges_over_batch(torch.tensor(tokenized["input_ids"]).unsqueeze(0), self.tokenizer.sep_token_id)
        ranges_over_batch = sort_ranges(ranges_over_batch).squeeze(0).to("cpu").detach().tolist()
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label,
            "graph_embeddings" : torch.stack([user_embedding, movie_embedding]),
            "ranges_over_batch": ranges_over_batch
            }
    

class PromptEmbeddingDataCollator(TextBasedDataCollator):
    '''
    The Prompt Data Collator also adds embeddings to the prompt on the fly.
    '''
    def __init__(self, tokenizer, df, data, get_embedding_cb, kge_dimension = 4, false_ratio = 2.0):
        super().__init__(tokenizer=tokenizer,false_ratio = false_ratio, df = df)
        self.data = data
        self.get_embedding_cb = get_embedding_cb
        self.kge_dimension = kge_dimension


    def _transform_to_false_example(self) -> Dict:
        label = 0
        user_id, movie_id = self._find_non_existing_user_movie()
        random_row = self.df[self.df["mappedMovieId"] == movie_id].sample(1).iloc[0]
        random_row["mappedUserId"] = user_id
        user_embedding, movie_embedding = self.get_embedding_cb(self.data, user_id, movie_id)
        random_row["user_embedding"] = user_embedding.to("cpu").detach().tolist()
        random_row["movie_embedding"] = movie_embedding.to("cpu").detach().tolist()
        random_row["prompt"] = row_to_prompt_datapoint(random_row, self.kge_dimension, sep_token=self.tokenizer.sep_token)
        tokenized = self.tokenizer(random_row["prompt"], padding="max_length", truncation=True)
        ranges_over_batch = get_ranges_over_batch(torch.tensor(tokenized["input_ids"]).unsqueeze(0), self.tokenizer.sep_token_id)
        ranges_over_batch = sort_ranges(ranges_over_batch)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label,
            "ranges_over_batch": ranges_over_batch.squeeze(0).to("cpu").detach().tolist()
            }
    
class VanillaEmbeddingDataCollator(TextBasedDataCollator):
    '''
    The vanilla data collator does only generate false edges with the prompt, title, user_id and genres.
    '''
    def __init__(self, tokenizer, df, false_ratio = 2.0):
        super().__init__(tokenizer=tokenizer,false_ratio = false_ratio, df = df)


    def _transform_to_false_example(self) -> Dict:
        label = 0
        user_id, movie_id = self._find_non_existing_user_movie()
        random_row = self.df[self.df["mappedMovieId"] == movie_id].sample(1).iloc[0]
        random_row["mappedUserId"] = user_id
        random_row["prompt"] = row_to_vanilla_datapoint(random_row, self.tokenizer.sep_token)
        tokenized = self.tokenizer(random_row["prompt"], padding="max_length", truncation=True)
        ranges_over_batch = get_ranges_over_batch(torch.tensor(tokenized["input_ids"]).unsqueeze(0), self.tokenizer.sep_token_id)
        ranges_over_batch = sort_ranges(ranges_over_batch)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label,
            "ranges_over_batch": ranges_over_batch.squeeze(0).to("cpu").detach().tolist()
            }
    

METRIC = evaluate.load('accuracy')

class CustomTrainer(Trainer):
    '''
    This custom trainer is needed, so we can have different data collators while training and evaluating.
    For that we adjust the get_eval_dataloader method.
    '''
    def __init__(self, *args, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_data_collator = eval_data_collator
        

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
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
            self.eval_dataset[eval_dataset] # type: ignore
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        ) # type: ignore
        data_collator = self.test_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation") # type: ignore
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation") # type: ignore

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset): # type: ignore
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset) # type: ignore
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params) # type: ignore
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
    
class SequenceClassifierOutputOverRanges(SequenceClassifierOutput):
    def __init__(self, loss=None, logits=None, hidden_states=None, attentions=None, ranges_over_batch=None):
        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions) # type: ignore
        self.ranges_over_batch = ranges_over_batch

    def to_tuple(self):
        # Ensure that your custom field is included when converting to a tuple
        return tuple(v for v in (self.loss, self.logits, self.hidden_states, self.attentions, self.ranges_over_batch) if v is not None)

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
        ranges_over_batch: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputOverRanges]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
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
            ranges_over_batch = ranges_over_batch
        )

class AddingEmbeddingBertForSequenceClassification(BertForSequenceClassification):
    
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
        ranges_over_batch: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputOverRanges]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.bert.embeddings(input_ids)
            assert isinstance(inputs_embeds, torch.Tensor)
        if graph_embeddings is not None and len(graph_embeddings) > 0:
            
            if attention_mask is not None:
                mask = ((attention_mask.to(self.device).sum(dim = 1) -1).unsqueeze(1).repeat((1,2))-torch.tensor([3,1], device=self.device)).unsqueeze(2).repeat((1,1,self.config.hidden_size))
                inputs_embeds = inputs_embeds.to(self.device).scatter(1, mask.to(self.device), graph_embeddings.to(self.device))
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
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
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
            ranges_over_batch = ranges_over_batch,
        )

class ClassifierBase(ABC):
    def __init__(self, df, semantic_datapoints, best_model_path, attentions_path, hidden_states_path, tokens_path, batch_size = 64, force_recompute = False) -> None:
        self.predictions = None
        self.df = df
        self.batch_size = batch_size
        self.semantic_datapoints = semantic_datapoints
        self.best_model_path = best_model_path
        self.attentions_path = attentions_path
        self.hidden_states_path = hidden_states_path
        self.tokens_path = tokens_path
        self.force_recompute = force_recompute
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return METRIC.compute(predictions=predictions, references=labels)
    
    @abstractmethod
    def _set_up_trainer(self, dataset, tokenize = False, eval_data_collator = None, epochs = 3) -> Trainer:
        pass

    
    def train_model_on_data(self, dataset, epochs = 3):
        trainer = self._set_up_trainer(dataset, epochs = epochs)

        # Train the model
        trainer.train()

        trainer.model.to(device = "cpu").save_pretrained(self.best_model_path) # type: ignore
        trainer.model.to(device = self.device) # type: ignore

    def plot_attentions(self, attentions, title):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(attentions, xticklabels=self.semantic_datapoints, yticklabels=self.semantic_datapoints, cmap='viridis', ax=ax)
        plt.title(title)
        plt.xlabel('Tokens')
        plt.ylabel('Tokens')
        plt.show()

    def plot_attention_graph(self, attentions, title, weight_coef = 5, fig_dpi = 100, fig_size = (8,8)):
        plt.rcParams['figure.figsize'] = fig_size
        plt.rcParams['figure.dpi'] = fig_dpi # 200 e.g. is really fine, but slower
        plt.figure(figsize=fig_size, dpi=fig_dpi)
        # Create an undirected graph
        G = nx.Graph()
        labels = {}
        attentions = torch.mean(attentions, dim = 0).permute((2,0,1))
        for layer, attentions_ in enumerate(attentions):
            for from_, inner in enumerate(attentions_):
                from_name = f"{self.semantic_datapoints[from_]}_{layer}"
                if layer == 0:
                    labels[from_name] = self.semantic_datapoints[from_]
                G.add_node(from_name, name = from_name, layer = layer)
                for to_, weight in enumerate(inner):
                    to_name = f"{self.semantic_datapoints[to_]}_{layer+1}"
                    G.add_node(to_name, name = to_name, layer = layer+1)
                    G.add_edge(from_name, to_name, weight = weight)

        pos = nx.multipartite_layout(G, subset_key="layer")
        semantic_datapoints = self.semantic_datapoints.copy()
        semantic_datapoints.reverse()
        for node, (x, y) in pos.items():
            y = semantic_datapoints.index(node.split("_")[0]) 
            layer = G.nodes[node]['layer']
            pos[node] = (layer, y)  # type: ignore # Fixing the x-coordinate to be the layer and y-coordinate can be customized

        nx.draw(G, pos=pos, with_labels = False)

        edge_weights = nx.get_edge_attributes(G, 'weight')

        # Create a list of edge thicknesses based on weights
        # Normalize the weights to get thickness values
        max_weight = max(edge_weights.values())
        edge_thickness = [edge_weights[edge] / max_weight * weight_coef for edge in G.edges()]# Draw edges with varying thicknesses
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=edge_thickness) # type: ignore
        shift_to_left = 0.4
        label_pos = {node: (x - shift_to_left, y) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(G,label_pos,labels)
        # Get the current axis limits
        x_min, x_max = plt.xlim()

        # Adjust x-axis limits to add more whitespace on the left
        plt.xlim(x_min - (shift_to_left+0.1), x_max)  # Increase the left limit by 0.1
        plt.title(title)
        plt.show()

    @abstractmethod
    def save_pca(self, pca, label):
        pass

    @abstractmethod
    def pcas_exists(self):
        pass
    
    def load_pcas(self):
        pass

    def init_pca(self, hidden_states, force_recompute = False):
        if self.pcas_exists() and not force_recompute:
            return self.load_pcas()
        else:
            
            hidden_states = hidden_states.numpy()
            pcas = []
            for idx, label in enumerate(self.semantic_datapoints):
                pca = PCA(n_components=2)  # Adjust number of components as needed
                pcas.append(pca)
                position_hidden_states = hidden_states[:,idx]
                pca.fit_transform(position_hidden_states)
                self.save_pca(pca, label = label)
            return pcas
    
    def _plot_training_loss_and_accuracy(self, path_to_trainer_state, model_type):
        with open(path_to_trainer_state, 'r') as f:
            trainer_state = json.load(f)
            # Extract loss values and corresponding steps
        losses = []
        steps = []

        for log in trainer_state['log_history']:
            if 'loss' in log:
                losses.append(log['loss'])
                steps.append(log['step'])

        # Extract accuracy values and corresponding epochs
        accuracies = []
        epochs = []

        for log in trainer_state['log_history']:
            if 'eval_accuracy' in log:
                accuracies.append(log['eval_accuracy'])
                epochs.append(log['epoch'])

        # Find the minimum loss and its corresponding step
        min_loss = min(losses)
        min_loss_step = steps[losses.index(min_loss)]

        # Find the maximum accuracy and its corresponding epoch
        max_accuracy = max(accuracies)
        max_accuracy_epoch = epochs[accuracies.index(max_accuracy)]

        # Plot loss development over steps
        plt.figure(figsize=(12, 6))
        plt.plot(steps, losses, label='Loss')
        plt.scatter(min_loss_step, min_loss, color='red')  # Mark the minimum loss
        plt.text(min_loss_step, min_loss, f'Min Loss: {min_loss:.4f}', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title(f'Loss Development over Steps of {model_type} Model')
        plt.legend()
        plt.show()

        # Plot accuracy development over epochs
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, accuracies, label='Accuracy', color='green')
        plt.scatter(max_accuracy_epoch, max_accuracy, color='red')  # Mark the maximum accuracy
        plt.text(max_accuracy_epoch, max_accuracy, f'Max Accuracy: {max_accuracy:.4f}', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Development over Epochs of {model_type} Model')
        plt.legend()
        plt.show()
        

class ClassifierOriginalArchitectureBase(ClassifierBase):
    def __init__(self, df, semantic_datapoints, best_model_path, attentions_path, hidden_states_path, tokens_path, model_name = "google/bert_uncased_L-2_H-128_A-2", batch_size = 64, model_max_length = 256, force_recompute = False) -> None:
        super().__init__( df = df, semantic_datapoints = semantic_datapoints, best_model_path = best_model_path, attentions_path = attentions_path, hidden_states_path = hidden_states_path, tokens_path = tokens_path, batch_size = batch_size, force_recompute = force_recompute)
        self.model_name = model_name
        
        # Initialize the model and tokenizer
        if os.path.exists(self.best_model_path) and not self.force_recompute:
            self.model = BertForSequenceClassificationRanges.from_pretrained(self.best_model_path, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
        else:
            self.model = BertForSequenceClassificationRanges.from_pretrained(self.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, model_max_length=model_max_length)

    def tokenize_function(self, example, return_pt = False):
        tokenized =  self.tokenizer(example["prompt"], padding="max_length", truncation=True, return_tensors = "pt")
        ranges_over_batch = get_ranges_over_batch(tokenized["input_ids"], self.tokenizer.sep_token_id)
        ranges_over_batch = sort_ranges(ranges_over_batch)
        if return_pt:
            result = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": example["labels"],
                "ranges_over_batch": ranges_over_batch}
        else:
            result = {
                "input_ids": tokenized["input_ids"].detach().to("cpu").tolist(),
                "attention_mask": tokenized["attention_mask"].detach().to("cpu").tolist(),
                "labels": example["labels"],
                "ranges_over_batch": ranges_over_batch.detach().to("cpu").tolist()}
        return result
        
        
class AddingEmbeddingsBertClassifierBase(ClassifierBase):
    def __init__(self, movie_lens_loader, get_embedding_cb, model_name = "google/bert_uncased_L-2_H-128_A-2", kge_dimension = 128, batch_size = 64,model_max_length = 256, false_ratio = 2.0, force_recompute = False) -> None:
        self.kge_dimension = kge_dimension
        best_model_path = LLM_ADDING_BEST_MODEL_PATH.format(self.kge_dimension)
        self.model_name = model_name
        attentions_path = ADDING_ATTENTIONS_PATH.format(self.kge_dimension)
        hidden_states_path = ADDING_HIDDEN_STATES_PATH.format(self.kge_dimension)
        self.graph_embeddings_path = ADDING_GRAPH_EMBEDDINGS_PATH.format(self.kge_dimension)
        tokens_path = ADDING_TOKENS_PATH.format(self.kge_dimension)
        super().__init__(df = movie_lens_loader.llm_df, semantic_datapoints = EMBEDDING_BASED_SEMANTIC_TOKENS, best_model_path = best_model_path, attentions_path = attentions_path, hidden_states_path = hidden_states_path, tokens_path = tokens_path, batch_size = batch_size, force_recompute = force_recompute)
        
        # Initialize the model and tokenizer
        if os.path.exists(self.best_model_path) and not self.force_recompute:
            self.model = AddingEmbeddingBertForSequenceClassification.from_pretrained(self.best_model_path, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
        else:
            self.model = AddingEmbeddingBertForSequenceClassification.from_pretrained(self.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, model_max_length=model_max_length)
        self.train_data_collator = EmbeddingBasedDataCollator(self.tokenizer, self.device, movie_lens_loader.llm_df, movie_lens_loader.gnn_train_data, get_embedding_cb, kge_dimension=self.kge_dimension, false_ratio = false_ratio)
        self.test_data_collator = EmbeddingBasedDataCollator(self.tokenizer, self.device, movie_lens_loader.llm_df, movie_lens_loader.gnn_test_data, get_embedding_cb, kge_dimension=self.kge_dimension, false_ratio = false_ratio)
        self.eval_data_collator = EmbeddingBasedDataCollator(self.tokenizer, self.device, movie_lens_loader.llm_df, movie_lens_loader.gnn_val_data, get_embedding_cb, kge_dimension=self.kge_dimension, false_ratio = false_ratio)

    def _set_up_trainer(self, dataset, tokenize = False, eval_data_collator = None, epochs = 3):
        if tokenize:
            tokenized_dataset = dataset.map(self.tokenize_function, batched = True)
        else:
            tokenized_dataset = dataset
        training_args = TrainingArguments(
            output_dir=LLM_ADDING_TRAINING_PATH.format(self.kge_dimension),
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=ADDING_LOG_PATH.format(self.kge_dimension),
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
    
    def tokenize_function(self, example, return_pt = False):
        tokenized =  self.tokenizer(example["prompt"], padding="max_length", truncation=True, return_tensors = "pt")
        ranges_over_batch = get_ranges_over_batch(tokenized["input_ids"], self.tokenizer.sep_token_id)
        ranges_over_batch = sort_ranges(ranges_over_batch)
        if return_pt:
            result = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": example["labels"],
                "ranges_over_batch": ranges_over_batch,
                "graph_embeddings": example["graph_embeddings"]}
        else:
            result = {
                "input_ids": tokenized["input_ids"].detach().to("cpu").tolist(),
                "attention_mask": tokenized["attention_mask"].detach().to("cpu").tolist(),
                "labels": example["labels"],
                "ranges_over_batch": ranges_over_batch.detach().to("cpu").tolist(),
                "graph_embeddings": example["graph_embeddings"]}
        return result
    def plot_confusion_matrix(self, split, dataset, tokenize = False, force_recompute = False):
        if split == "test":
            trainer = self._set_up_trainer(dataset, tokenize = tokenize)
            dataset = dataset["test"]
        else:
            trainer = self._set_up_trainer(dataset, tokenize = tokenize, eval_data_collator = self.eval_data_collator)
            dataset = dataset["val"]
        if not self.predictions or force_recompute: 
        # Generate predictions
            predictions = trainer.predict(dataset)
            self.predictions = predictions
        # Get predicted labels and true labels
        preds = np.argmax(self.predictions.predictions, axis=-1)
        labels = self.predictions.label_ids
        # Compute confusion matrix
        cm = confusion_matrix(labels, preds) # type: ignore

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues) # type: ignore
        plt.show()

    def _get_data_collator(self, split):
        return self.test_data_collator if split == "test" else self.eval_data_collator if split == "val" else self.train_data_collator
    
    def plot_training_loss_and_accuracy(self, kge_dimension = 128):
        model_type = "Embedding"
        self._plot_training_loss_and_accuracy(ADDING_TRAINING_STATE_PATH.format(kge_dimension), model_type)

    def forward_dataset_and_save_outputs(self, dataset, splits = ["train", "test","val"], batch_size = 64, epochs = 3, load_fields = ["attentions", "hidden_states", "graph_embeddings"], force_recompute = False):
        if force_recompute or not os.path.exists(self.attentions_path) or not os.path.exists(self.hidden_states_path) or not os.path.exists(self.tokens_path) or not os.path.exists(self.graph_embeddings_path):
            assert isinstance(self.model, AddingEmbeddingBertForSequenceClassification)
            self.model.eval()
            Path(ADDING_ATTENTIONS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(ADDING_HIDDEN_STATES_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(ADDING_INPUT_IDS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(ADDING_RANGES_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(ADDING_SUB_TOKENS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(ADDING_GRAPH_EMBEDDINGS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
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
                        data_loader = DataLoader(dataset=dataset[split], batch_size= batch_size, collate_fn = data_collator)
                        for idx, batch in enumerate(data_loader):
                        #if True:
                        #    batch = next(iter(data_loader))
                            splits_ = [split] *  len(batch["input_ids"])
                            outputs = self.model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], graph_embeddings = batch["graph_embeddings"], ranges_over_batch = batch["ranges_over_batch"], output_hidden_states=add_hidden_states, output_attentions = add_attentions)
                            ranges_over_batch = batch["ranges_over_batch"]
                            if add_attentions:
                                attentions = outputs.attentions
                                attentions = [torch.sum(layer, dim=1) for layer in attentions]
                                attentions = torch.stack(attentions).permute(1,2,3,0)
                                attentions = means_over_ranges_cross(ranges_over_batch, attentions).numpy()
                                all_attentions.append(attentions)
                            if add_hidden_states:
                                hidden_states_on_each_layer = []
                                for hidden_states in outputs.hidden_states:
                                    hidden_states_on_layer = avg_over_hidden_states(ranges_over_batch, hidden_states).permute((1,0,2)).numpy()
                                    hidden_states_on_each_layer.append(hidden_states_on_layer)
                                hidden_states_on_each_layer = np.stack(hidden_states_on_each_layer)
                                all_hidden_states.append(hidden_states_on_each_layer)
                            tokens = self.get_tokens_as_df(batch["input_ids"], ranges_over_batch[:, [1,3,5]])
                            if add_graph_embeddings:
                                all_graph_embeddings.append(batch["graph_embeddings"].numpy())
                            tokens["labels"] = batch["labels"].tolist()
                            tokens["split"] = splits_
                            all_tokens.append(tokens)



                        # Concatenate all hidden states across batches
                        all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                        all_tokens.to_csv(ADDING_SUB_TOKENS_PATH.format(self.kge_dimension, split, epoch), index = False)
                        del all_tokens             
                        if "attentions" in load_fields:
                            all_attentions = np.concatenate(all_attentions)
                            np.save(ATTENTIONS_ADDING_PATH.format(self.kge_dimension, split, epoch), all_attentions)
                            del all_attentions
                        if "hidden_states" in load_fields:
                            all_hidden_states = np.concatenate(all_hidden_states)
                            np.save(HIDDEN_STATES_ADDING_PATH.format(self.kge_dimension, split, epoch), all_hidden_states)
                            del all_hidden_states
                        if "graph_embeddings" in load_fields:
                            all_graph_embeddings = np.concatenate(all_graph_embeddings)
                            np.save(GRAPH_EMBEDDINGS_ADDING_PATH.format(self.kge_dimension, split, epoch), all_graph_embeddings)
                            del all_graph_embeddings

                # all_tokens
                all_tokens = []
                for split in splits:
                    for epoch in range(epochs):
                        all_tokens.append(pd.read_csv(ADDING_SUB_TOKENS_PATH.format(self.kge_dimension, split, epoch)))
                all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                all_tokens.to_csv(self.tokens_path, index = False)

                # hidden states:
                if "hidden_states" in load_fields:
                    all_hidden_states = []
                    for split in splits:
                        for epoch in range(epochs):
                            all_hidden_states.append(np.load(HIDDEN_STATES_ADDING_PATH.format(split, epoch)))
                    all_hidden_states = np.concatenate(all_hidden_states)
                    np.save(self.hidden_states_path, all_hidden_states)
                    all_hidden_states = torch.from_numpy(all_hidden_states)
                    all_tokens["hidden_states"] = all_hidden_states.unbind()
                    del all_hidden_states

                # attentions:
                if "attentions" in load_fields:
                    attentions = []
                    for split in splits:
                        for epoch in range(epochs):
                            attentions.append(np.load(ATTENTIONS_ADDING_PATH.format(split, epoch)))
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
                            graph_embeddings.append(np.load(GRAPH_EMBEDDINGS_ADDING_PATH.format(self.kge_dimension, split, epoch)))
                    graph_embeddings = np.concatenate(graph_embeddings)
                    np.save(self.graph_embeddings_path, graph_embeddings)
                    graph_embeddings = torch.from_numpy(graph_embeddings)
                    all_tokens["graph_embeddings"] = graph_embeddings.unbind()
                    del graph_embeddings
        else:
            all_tokens = pd.read_csv(self.tokens_path)
            if "hidden_states" in load_fields:
                averaged_hidden_states = torch.from_numpy(np.load(self.hidden_states_path))
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
    
    def get_tokens_as_df(self, input_ids, all_ranges_over_batch) -> pd.DataFrame:
        user_ids = []
        titles = []
        genres = []
        all_semantic_tokens = [user_ids, titles, genres]
        ends = all_ranges_over_batch[:,:,1]
        starts = all_ranges_over_batch[:,:,0]
        # input: # ends: torch.tensor([2, 5, 6]) starts: tensor([0, 2, 4])
        # Compute the maximum length of the ranges
        max_length = (ends - starts).max()
        # Create a range tensor from 0 to max_length-1
        range_tensor = torch.arange(max_length).unsqueeze(0)
        for pos, semantic_tokens in enumerate(all_semantic_tokens):
            # Compute the ranges using broadcasting and masking
            ranges =  starts[:,pos].unsqueeze(1) + range_tensor
            mask = ranges < ends[:,pos].unsqueeze(1)

            # Apply the mask
            result = ranges * mask  # result: tensor([[0, 1, 0], [2, 3, 4], [4, 5, 0]]) here padding index is 0
                                    #                        -                     -    positions were padded
            #result = result.unsqueeze(dim = 2).repeat(1,1, input_ids.shape[2])
            gather = input_ids.gather(dim = 1, index = result)
            decoded = self.tokenizer.batch_decode(gather, skip_special_tokens = False)
            semantic_tokens.extend([decode.replace(" [CLS]", "") for decode in decoded])
        all_semantic_tokens[0] = [int(id) for id in all_semantic_tokens[0]]
        all_semantic_tokens[2] = [ast.literal_eval(string_list) for string_list in all_semantic_tokens[2]]
        data = {"user_id": all_semantic_tokens[0], "title": all_semantic_tokens[1], "genres": all_semantic_tokens[2]}
        df = pd.DataFrame(data)
        return df

    
    def save_pca(self, pca, label):
        pca_path = PCA_EMBEDDING_MODEL_PATH.format(self.kge_dimension, label)
        joblib.dump(pca, pca_path)

    def pcas_exists(self):
        for label in self.semantic_datapoints:
            pca_path = PCA_EMBEDDING_MODEL_PATH.format(self.kge_dimension, label)
            if not os.path.exists(pca_path):
                return False
        return True
    
    def load_pcas(self):
        pcas = []
        for label in self.semantic_datapoints:
            pca_path = PCA_EMBEDDING_MODEL_PATH.format(self.kge_dimension, label)
            pcas.append(joblib.load(pca_path))
        return pcas
        
    
    
class PromptBertClassifier(ClassifierOriginalArchitectureBase):
    def __init__(self, movie_lens_loader, get_embedding_cb, model_name = "google/bert_uncased_L-2_H-128_A-2", kge_dimension = 4, batch_size = 64,model_max_length = 256, false_ratio = 2.0, force_recompute = False) -> None:
        assert kge_dimension <= 16
        self.kge_dimension = kge_dimension
        best_model_path = LLM_PROMPT_BEST_MODEL_PATH.format(self.kge_dimension)
        attentions_path = PROMPT_ATTENTIONS_PATH.format(self.kge_dimension)
        hidden_states_path = PROMPT_HIDDEN_STATES_PATH.format(self.kge_dimension)
        self.graph_embeddings_path = PROMPT_GRAPH_EMBEDDINGS_PATH.format(self.kge_dimension)
        tokens_path = PROMPT_TOKENS_PATH.format(self.kge_dimension)
        super().__init__(df = movie_lens_loader.llm_df, semantic_datapoints=EMBEDDING_BASED_SEMANTIC_TOKENS, model_name=model_name, best_model_path = best_model_path, attentions_path = attentions_path, hidden_states_path = hidden_states_path, tokens_path = tokens_path, force_recompute=force_recompute, batch_size = batch_size,model_max_length = model_max_length)
        self.train_data_collator = PromptEmbeddingDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_train_data, get_embedding_cb, kge_dimension = kge_dimension, false_ratio = false_ratio)
        self.test_data_collator = PromptEmbeddingDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_test_data, get_embedding_cb, kge_dimension = kge_dimension, false_ratio = false_ratio)
        self.eval_data_collator = PromptEmbeddingDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_val_data, get_embedding_cb, kge_dimension = kge_dimension, false_ratio = false_ratio)

    
    
    def _set_up_trainer(self, dataset, tokenize = False, eval_data_collator = None, epochs = 3):
        if tokenize:
            tokenized_dataset = dataset.map(self.tokenize_function, batched = True)
        else:
            tokenized_dataset = dataset
        training_args = TrainingArguments(
            output_dir=LLM_PROMPT_TRAINING_PATH.format(self.kge_dimension),
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=PROMPT_LOG_PATH.format(self.kge_dimension),
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True
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
    
    def plot_confusion_matrix(self, split, dataset, tokenize = False, force_recompute = False):
        if split == "test":
            trainer = self._set_up_trainer(dataset, tokenize = tokenize)
            dataset = dataset["test"]
        else:
            trainer = self._set_up_trainer(dataset, tokenize = tokenize, eval_data_collator = self.eval_data_collator)
            dataset = dataset["val"]
        if not self.predictions or force_recompute: 
        # Generate predictions
            predictions = trainer.predict(dataset)
            self.predictions = predictions
        # Get predicted labels and true labels
        preds = np.argmax(self.predictions.predictions, axis=-1)
        labels = self.predictions.label_ids
        # Compute confusion matrix
        cm = confusion_matrix(labels, preds) # type: ignore

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues) # type: ignore
        plt.show()

    def plot_training_loss_and_accuracy(self, kge_dimension = 4):
        model_type = "Prompt"
        self._plot_training_loss_and_accuracy(PROMPT_TRAINING_STATE_PATH.format(kge_dimension), model_type)

    def forward_dataset_and_save_outputs(self, dataset, splits = ["train", "test","val"], batch_size = 64, epochs = 3, load_fields = ["attentions", "hidden_states", "graph_embeddings"], force_recompute = False):
        if force_recompute or not os.path.exists(self.attentions_path) or not os.path.exists(self.hidden_states_path) or not os.path.exists(self.tokens_path) or not os.path.exists(self.graph_embeddings_path):
            assert isinstance(self.model, BertForSequenceClassificationRanges)
            self.model.eval()
            Path(PROMPT_ATTENTIONS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_HIDDEN_STATES_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_INPUT_IDS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_RANGES_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_SUB_TOKENS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_GRAPH_EMBEDDINGS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
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
                        data_loader = DataLoader(dataset=dataset[split], batch_size= batch_size, collate_fn = data_collator)
                        for idx, batch in enumerate(data_loader):
                        #if True:
                        #    batch = next(iter(data_loader))
                            splits_ = [split] *  len(batch["input_ids"])
                            outputs = self.model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], ranges_over_batch = batch["ranges_over_batch"], output_hidden_states=add_hidden_states, output_attentions = add_attentions)
                            ranges_over_batch = batch["ranges_over_batch"]
                            if add_attentions:
                                attentions = outputs.attentions
                                attentions = [torch.sum(layer, dim=1) for layer in attentions]
                                attentions = torch.stack(attentions).permute(1,2,3,0)
                                attentions = means_over_ranges_cross(ranges_over_batch, attentions).numpy()
                                all_attentions.append(attentions)
                            if add_hidden_states:
                                hidden_states_on_each_layer = []
                                for hidden_states in outputs.hidden_states:
                                    hidden_states_on_layer = avg_over_hidden_states(ranges_over_batch, hidden_states).permute((1,0,2)).numpy()
                                    hidden_states_on_each_layer.append(hidden_states_on_layer)
                                hidden_states_on_each_layer = np.stack(hidden_states_on_each_layer)
                                all_hidden_states.append(hidden_states_on_each_layer)
                            tokens, graph_embeddings = self.get_tokens_as_df(batch["input_ids"], ranges_over_batch[:,[1,3,5,7,9]])
                            if add_graph_embeddings:
                                all_graph_embeddings.append(graph_embeddings.numpy())
                            del graph_embeddings
                            tokens["labels"] = batch["labels"].tolist()
                            tokens["split"] = splits_
                            all_tokens.append(tokens)



                        # Concatenate all hidden states across batches
                        all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                        all_tokens.to_csv(PROMPT_SUB_TOKENS_PATH.format(self.kge_dimension, split, epoch), index = False)
                        del all_tokens             
                        if "attentions" in load_fields:
                            all_attentions = np.concatenate(all_attentions)
                            np.save(ATTENTIONS_PROMPT_PATH.format(self.kge_dimension, split, epoch), all_attentions)
                            del all_attentions
                        if "hidden_states" in load_fields:
                            all_hidden_states = np.concatenate(all_hidden_states)
                            np.save(HIDDEN_STATES_PROMPT_PATH.format(self.kge_dimension, split, epoch), all_hidden_states)
                            del all_hidden_states
                        if "graph_embeddings" in load_fields:
                            all_graph_embeddings = np.concatenate(all_graph_embeddings)
                            np.save(GRAPH_EMBEDDINGS_PROMPT_PATH.format(self.kge_dimension, split, epoch), all_graph_embeddings)
                            del all_graph_embeddings

                # all_tokens
                all_tokens = []
                for split in splits:
                    for epoch in range(epochs):
                        all_tokens.append(pd.read_csv(PROMPT_SUB_TOKENS_PATH.format(self.kge_dimension, split, epoch)))
                all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                all_tokens.to_csv(self.tokens_path, index = False)

                # hidden states:
                if "hidden_states" in load_fields:
                    all_hidden_states = []
                    for split in splits:
                        for epoch in range(epochs):
                            all_hidden_states.append(np.load(HIDDEN_STATES_PROMPT_PATH.format(split, epoch)))
                    all_hidden_states = np.concatenate(all_hidden_states)
                    np.save(self.hidden_states_path, all_hidden_states)
                    all_hidden_states = torch.from_numpy(all_hidden_states)
                    all_tokens["hidden_states"] = all_hidden_states.unbind()
                    del all_hidden_states

                # attentions:
                if "attentions" in load_fields:
                    attentions = []
                    for split in splits:
                        for epoch in range(epochs):
                            attentions.append(np.load(ATTENTIONS_PROMPT_PATH.format(split, epoch)))
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
                            graph_embeddings.append(np.load(GRAPH_EMBEDDINGS_PROMPT_PATH.format(self.kge_dimension, split, epoch)))
                    graph_embeddings = np.concatenate(graph_embeddings)
                    np.save(self.graph_embeddings_path, graph_embeddings)
                    graph_embeddings = torch.from_numpy(graph_embeddings)
                    all_tokens["graph_embeddings"] = graph_embeddings.unbind()
                    del graph_embeddings
        else:
            all_tokens = pd.read_csv(self.tokens_path)
            if "hidden_states" in load_fields:
                averaged_hidden_states = torch.from_numpy(np.load(self.hidden_states_path))
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

    def get_tokens_as_df(self, input_ids, all_ranges_over_batch) -> Tuple[pd.DataFrame, torch.Tensor]:
        all_semantic_tokens = [[], [], [], [], []]
        ends = all_ranges_over_batch[:,:,1]
        starts = all_ranges_over_batch[:,:,0]
        # input: # ends: torch.tensor([2, 5, 6]) starts: tensor([0, 2, 4])
        # Compute the maximum length of the ranges
        max_length = (ends - starts).max()
        # Create a range tensor from 0 to max_length-1
        range_tensor = torch.arange(max_length).unsqueeze(0)
        for pos, semantic_tokens in enumerate(all_semantic_tokens):
            # Compute the ranges using broadcasting and masking
            ranges =  starts[:,pos].unsqueeze(1) + range_tensor
            mask = ranges < ends[:,pos].unsqueeze(1)

            # Apply the mask
            result = ranges * mask  # result: tensor([[0, 1, 0], [2, 3, 4], [4, 5, 0]]) here padding index is 0
                                    #                        -                     -    positions were padded
            #result = result.unsqueeze(dim = 2).repeat(1,1, input_ids.shape[2])
            gather = input_ids.gather(dim = 1, index = result)
            decoded = self.tokenizer.batch_decode(gather, skip_special_tokens = False)
            semantic_tokens.extend([decode.replace(" [CLS]", "") for decode in decoded])
        all_semantic_tokens[0] = [int(id) for id in all_semantic_tokens[0]]
        all_semantic_tokens[2] = [ast.literal_eval(string_list) for string_list in all_semantic_tokens[2]]
        all_semantic_tokens[3] = [[float(str_float) for str_float in ast.literal_eval(string_list.replace(" ", ""))] for string_list in all_semantic_tokens[3]]
        all_semantic_tokens[4] = [[float(str_float) for str_float in ast.literal_eval(string_list.replace(" ", ""))] for string_list in all_semantic_tokens[4]]
        user_embeddings = torch.tensor(all_semantic_tokens[3])
        movie_embeddings = torch.tensor(all_semantic_tokens[4])
        graph_embeddings = torch.stack([user_embeddings, movie_embeddings]).permute((1,0,2))
        data = {"user_id": all_semantic_tokens[0], "title": all_semantic_tokens[1], "genres": all_semantic_tokens[2]}
        df = pd.DataFrame(data)
        return df, graph_embeddings

    def _get_data_collator(self, split):
        return self.test_data_collator if split == "test" else self.eval_data_collator if split == "val" else self.train_data_collator
    
    def save_pca(self, pca, label):
        pca_path = PCA_PROMPT_MODEL_PATH.format(self.kge_dimension, label)
        joblib.dump(pca, pca_path)

    def pcas_exists(self):
        for label in self.semantic_datapoints:
            pca_path = PCA_PROMPT_MODEL_PATH.format(self.kge_dimension, label)
            if not os.path.exists(pca_path):
                return False
        return True

    def load_pcas(self):
        pcas = []
        for label in self.semantic_datapoints:
            pca_path = PCA_PROMPT_MODEL_PATH.format(self.kge_dimension, label)
            pcas.append(joblib.load(pca_path))
        return pcas
    


class VanillaBertClassifier(ClassifierOriginalArchitectureBase):
    def __init__(self, df, model_name = "google/bert_uncased_L-2_H-128_A-2", batch_size = 64,model_max_length = 256, false_ratio = 2.0, force_recompute = False) -> None:
        super().__init__(df = df, semantic_datapoints=ALL_SEMANTIC_TOKENS, model_name=model_name, best_model_path = LLM_VANILLA_BEST_MODEL_PATH, attentions_path = VANILLA_ATTENTIONS_PATH, hidden_states_path = VANILLA_HIDDEN_STATES_PATH, tokens_path = VANILLA_TOKENS_PATH, batch_size = batch_size, model_max_length = model_max_length, force_recompute=force_recompute)
        self.data_collator = VanillaEmbeddingDataCollator(self.tokenizer, df, false_ratio = false_ratio)
    
    def _set_up_trainer(self, dataset, tokenize = False, epochs = 3):
        if tokenize:
            tokenized_dataset = dataset.map(self.tokenize_function, batched = True)
        else:
            tokenized_dataset = dataset
        training_args = TrainingArguments(
            output_dir=LLM_VANILLA_TRAINING_PATH,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=VANILLA_LOG_PATH,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True
        )
        assert isinstance(self.model, BertForSequenceClassificationRanges)
        # Initialize the Trainer
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,  # type: ignore
        )
    
    def plot_confusion_matrix(self, split, dataset, tokenize = False, force_recompute = False):
        trainer = self._set_up_trainer(dataset, tokenize = tokenize)
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
        cm = confusion_matrix(labels, preds) # type: ignore

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues) # type: ignore
        plt.show()

    def plot_training_loss_and_accuracy(self):
        model_type = "Vanilla"
        self._plot_training_loss_and_accuracy(VANILLA_TRAINING_STATE_PATH, model_type)

    
    def save_pca(self, pca, label):
        pca_path = PCA_VANILLA_MODEL_PATH.format(label)
        joblib.dump(pca, pca_path)

    def pcas_exists(self):
        for label in self.semantic_datapoints:
            pca_path = PCA_VANILLA_MODEL_PATH.format(label)
            if not os.path.exists(pca_path):
                return False
        return True
    
    def load_pcas(self):
        pcas = []
        for label in self.semantic_datapoints:
            pca_path = PCA_VANILLA_MODEL_PATH.format(label)
            pcas.append(joblib.load(pca_path))
        return pcas
    
    def forward_dataset_and_save_outputs(self, dataset, splits = ["train", "test","val"], batch_size = 64, epochs = 3, load_fields = ["attentions", "hidden_states"], force_recompute = False):
        if force_recompute or not os.path.exists(self.attentions_path) or not os.path.exists(self.hidden_states_path) or not os.path.exists(self.tokens_path):
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
                        print(f"Vanilla {split} Forward Epoch {epoch + 1} from {epochs}")
                        data_loader = DataLoader(dataset=dataset[split], batch_size= batch_size, collate_fn = data_collator)
                        for idx, batch in enumerate(data_loader):
                        #if True:
                        #    batch = next(iter(data_loader))
                            input_ids = batch["input_ids"]
                            splits_ = [split] *  len(input_ids)
                            outputs = self.model(input_ids = input_ids, attention_mask = batch["attention_mask"], ranges_over_batch = batch["ranges_over_batch"], output_hidden_states=add_hidden_states, output_attentions = add_attentions)
                            ranges_over_batch = batch["ranges_over_batch"]
                            if "attentions" in load_fields:
                                attentions = outputs.attentions
                                attentions = [torch.sum(layer, dim=1) for layer in attentions]
                                attentions = torch.stack(attentions).permute(1,2,3,0)
                                attentions = means_over_ranges_cross(ranges_over_batch, attentions)
                                all_attentions.append(attentions.numpy())
                                del attentions
                            if "hidden_states" in load_fields:
                                hidden_states_on_each_layer = []
                                for hidden_states in outputs.hidden_states:
                                    hidden_states_on_layer = avg_over_hidden_states(ranges_over_batch, hidden_states).permute((1,0,2))
                                    hidden_states_on_each_layer.append(hidden_states_on_layer.numpy())
                                    del hidden_states
                                hidden_states_on_each_layer = np.stack(hidden_states_on_each_layer)
                                all_hidden_states.append(hidden_states_on_each_layer)
                            tokens = self.get_tokens_as_df(input_ids, ranges_over_batch[:, [1,3,5]])
                            tokens["labels"] = batch["labels"].tolist()
                            tokens["split"] = splits_
                            all_tokens.append(tokens)



                        # Concatenate all hidden states across batches
                        all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                        all_tokens.to_csv(VANILLA_SUB_TOKENS_PATH.format(split, epoch), index = False)
                        del all_tokens
                        if "attentions" in load_fields:
                            all_attentions = np.concatenate(all_attentions)
                            np.save(ATTENTIONS_VANILLA_PATH.format(split, epoch), all_attentions)
                            del all_attentions
                        if "hidden_states" in load_fields:
                            all_hidden_states = np.concatenate(all_hidden_states)
                            np.save(HIDDEN_STATES_VANILLA_PATH.format(split, epoch), all_hidden_states)
                            del all_hidden_states

                # all_tokens
                all_tokens = []
                for split in splits:
                    for epoch in range(epochs):
                        all_tokens.append(pd.read_csv(VANILLA_SUB_TOKENS_PATH.format(split, epoch)))
                all_tokens = pd.concat(all_tokens).reset_index(drop=True)
                all_tokens.to_csv(self.tokens_path, index = False)

                # hidden states:
                if "hidden_states" in load_fields:
                    all_hidden_states = []
                    for split in splits:
                        for epoch in range(epochs):
                            all_hidden_states.append(np.load(HIDDEN_STATES_VANILLA_PATH.format(split, epoch)))
                    all_hidden_states = np.concatenate(all_hidden_states)
                    np.save(self.hidden_states_path, all_hidden_states)
                    all_hidden_states = torch.from_numpy(all_hidden_states)
                    all_tokens["hidden_states"] = all_hidden_states.unbind()
                    del all_hidden_states

                # attentions:
                if "attentions" in load_fields:
                    attentions = []
                    for split in splits:
                        for epoch in range(epochs):
                            attentions.append(np.load(ATTENTIONS_VANILLA_PATH.format(split, epoch)))
                    attentions = np.concatenate(attentions)
                    np.save(self.attentions_path, attentions)
                    attentions = torch.from_numpy(attentions)
                    all_tokens["attentions"] = attentions.unbind()
                    del attentions

        else:
            all_tokens = pd.read_csv(self.tokens_path)
            if "hidden_states" in load_fields:
                averaged_hidden_states = torch.from_numpy(np.load(self.hidden_states_path))
                all_tokens["hidden_states"] = torch.unbind(averaged_hidden_states)
                del averaged_hidden_states
            if "attentions" in load_fields:
                averaged_attentions = torch.from_numpy(np.load(self.attentions_path))
                all_tokens["attentions"] = torch.unbind(averaged_attentions)
                del averaged_attentions
        all_tokens[all_tokens["split"].isin(splits)]
        return all_tokens
    
    def get_tokens_as_df(self, input_ids, all_ranges_over_batch) -> pd.DataFrame:
        user_ids = []
        titles = []
        genres = []
        all_semantic_tokens = [user_ids, titles, genres]
        ends = all_ranges_over_batch[:,:,1]
        starts = all_ranges_over_batch[:,:,0]
        # input: # ends: torch.tensor([2, 5, 6]) starts: tensor([0, 2, 4])
        # Compute the maximum length of the ranges
        max_length = (ends - starts).max()
        # Create a range tensor from 0 to max_length-1
        range_tensor = torch.arange(max_length).unsqueeze(0)
        for pos, semantic_tokens in enumerate(all_semantic_tokens):
            # Compute the ranges using broadcasting and masking
            ranges =  starts[:,pos].unsqueeze(1) + range_tensor
            mask = ranges < ends[:,pos].unsqueeze(1)

            # Apply the mask
            result = ranges * mask  # result: tensor([[0, 1, 0], [2, 3, 4], [4, 5, 0]]) here padding index is 0
                                    #                        -                     -    positions were padded
            #result = result.unsqueeze(dim = 2).repeat(1,1, input_ids.shape[2])
            gather = input_ids.gather(dim = 1, index = result)
            decoded = self.tokenizer.batch_decode(gather, skip_special_tokens = False)
            semantic_tokens.extend([decode.replace(" [CLS]", "") for decode in decoded])
        all_semantic_tokens[0] = [int(id) for id in all_semantic_tokens[0]]
        all_semantic_tokens[2] = [ast.literal_eval(string_list) for string_list in all_semantic_tokens[2]]
        data = {"user_id": all_semantic_tokens[0], "title": all_semantic_tokens[1], "genres": all_semantic_tokens[2]}
        df = pd.DataFrame(data)
        return df

    def _get_data_collator(self, split = None):
        return self.data_collator
        