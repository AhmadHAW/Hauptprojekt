import random as rd
from typing import Optional, Union, Dict, Tuple, List
from abc import ABC
import os
import json
import ast
from pathlib import Path

from movie_lens_loader import row_to_prompt_datapoint, row_to_adding_embedding_datapoint, row_to_vanilla_datapoint, LLM_PROMPT_TRAINING_PATH, LLM_ADDING_TRAINING_PATH, LLM_VANILLA_TRAINING_PATH, LLM_PROMPT_BEST_MODEL_PATH, LLM_ADDING_BEST_MODEL_PATH, LLM_VANILLA_BEST_MODEL_PATH, PCA_PATH, LLM_VANILLA_PATH, LLM_PROMPT_PATH, LLM_ADDING_PATH

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
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

VANILLA_ATTENTIONS_PATH = f"{LLM_VANILLA_PATH}/attentions.pt"
PROMPT_ATTENTIONS_PATH = f"{LLM_PROMPT_PATH}/attentions.pt"
ADDING_ATTENTIONS_PATH = f"{LLM_ADDING_PATH}/attentions.pt"
VANILLA_HIDDEN_STATES_PATH = f"{LLM_VANILLA_PATH}/hidden_states.pt"
PROMPT_HIDDEN_STATES_PATH = f"{LLM_PROMPT_PATH}/hidden_states.pt"
ADDING_HIDDEN_STATES_PATH = f"{LLM_ADDING_PATH}/hidden_states.pt"
PROMPT_GRAPH_EMBEDDINGS_PATH = f"{LLM_PROMPT_PATH}/graph_embeddings.pt"
ADDING_GRAPH_EMBEDDINGS_PATH = f"{LLM_ADDING_PATH}/graph_embeddings.pt"
VANILLA_TOKENS_PATH = f"{LLM_VANILLA_PATH}/tokens.csv"
PROMPT_TOKENS_PATH = f"{LLM_PROMPT_PATH}/tokens.csv"
ADDING_TOKENS_PATH = f"{LLM_ADDING_PATH}/tokens.csv"

SPLIT_EPOCH_ENDING = f"/split_{{}}_epoch_{{}}.npy"
TOKENS_ENDING = f"/tokens_{{}}_{{}}.csv"

VANILLA_HIDDEN_STATES_DIR_PATH = f"{LLM_VANILLA_PATH}/hidden_states"
LAST_HIDDEN_STATES_VANILLA_PATH = f"{VANILLA_HIDDEN_STATES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

VANILLA_RANGES_DIR_PATH = f"{LLM_VANILLA_PATH}/ranges"
RANGES_VANILLA_PATH = f"{VANILLA_RANGES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

VANILLA_ATTENTIONS_DIR_PATH = f"{LLM_VANILLA_PATH}/attentions"
ATTENTIONS_VANILLA_PATH = f"{VANILLA_ATTENTIONS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

VANILLA_INPUT_IDS_DIR_PATH = f"{LLM_VANILLA_PATH}/input_ids"
INPUT_IDS_VANILLA_PATH = f"{VANILLA_INPUT_IDS_DIR_PATH}{SPLIT_EPOCH_ENDING}"

VANILLA_SUB_TOKENS_DIR_PATH = f"{LLM_VANILLA_PATH}/tokens"
VANILLA_SUB_TOKENS_PATH = f"{VANILLA_SUB_TOKENS_DIR_PATH}{TOKENS_ENDING}"

PROMPT_HIDDEN_STATES_DIR_PATH = f"{LLM_PROMPT_PATH}/hidden_states"
LAST_HIDDEN_STATES_PROMPT_PATH = f"{PROMPT_HIDDEN_STATES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

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
LAST_HIDDEN_STATES_ADDING_PATH = f"{ADDING_HIDDEN_STATES_DIR_PATH}{SPLIT_EPOCH_ENDING}"

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




ALL_SEMANTIC_TOKENS = ["user_id", "title", "genres"]
EMBEDDING_BASED_SEMANTIC_TOKENS = ALL_SEMANTIC_TOKENS + ["user embedding", "movie embedding"]


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
    max_length = (ends - starts).max()
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

def avg_over_last_hidden_states(all_ranges_over_batch, last_hidden_states):
    averaged_hidden_states = []
    for position in range(all_ranges_over_batch.shape[1]):
        ranges_over_batch = all_ranges_over_batch[:,position]
        starts = ranges_over_batch[:,0]
        ends = ranges_over_batch[:,1]
        averaged_hidden_state = mean_over_ranges(last_hidden_states, starts, ends)
        #print(starts.shape, averaged_hidden_state.shape)
        averaged_hidden_states.append(averaged_hidden_state)
    return torch.stack(averaged_hidden_states)

class DataCollatorBase(DataCollatorForLanguageModeling, ABC):
    '''
    The Data Collators are used to generate non-existing edges on the fly. The false ratio allows to decide the ratio,
    in existing edges are replaced with non-existing edges.
    '''
    def __init__(self, tokenizer, df, false_ratio = 2.0):
        super().__init__(tokenizer=tokenizer, mlm=False)
        assert false_ratio > 0
        self.false_ratio = false_ratio
        self.tokenizer = tokenizer
        self.df = df

    def __call__(self, features):
        new_features = []
        for feature in features:
            #Every datapoint has a chance to be replaced by a negative datapoint, based on the false_ratio.
            #The _transform_to_false_exmample methods have to be implemented by the inheriting class.
            #For the prompt classifier, every new datapoint also contains embeddings of the nodes.
            if rd.uniform(0, 1) >=( 1 / (self.false_ratio + 1)):
                new_feature = self._transform_to_false_example()
                new_features.append(new_feature)
            else:
                new_features.append(feature)
        # Convert features into batches
        return self._convert_features_into_batches(new_features)
    
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
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
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
        for f in features:
            if isinstance(f["graph_embeddings"], list):
                f["graph_embeddings"] = torch.stack([torch.tensor(f["graph_embeddings"][0]), torch.tensor(f["graph_embeddings"][1])])
            else:
                f["graph_embeddings"] = f["graph_embeddings"]
        graph_embeddings = torch.stack([f["graph_embeddings"].to(self.device) for f in features])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "graph_embeddings": graph_embeddings
        }
    
    def _transform_to_false_example(self):
        label = 0
        user_id, movie_id = self._find_non_existing_user_movie()
        random_row = self.df[self.df["mappedMovieId"] == movie_id].iloc[0]
        random_row["mappedUserId"] = user_id
        user_embedding, movie_embedding = self.get_embedding_cb(self.data, user_id, movie_id)
        random_row["prompt"] = row_to_adding_embedding_datapoint(random_row, self.tokenizer.sep_token, self.tokenizer.pad_token)
        tokenized = self.tokenizer(random_row["prompt"], padding="max_length", truncation=True)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label,
            "graph_embeddings" : torch.stack([user_embedding, movie_embedding]),
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


    def _transform_to_false_example(self):
        label = 0
        user_id, movie_id = self._find_non_existing_user_movie()
        random_row = self.df[self.df["mappedMovieId"] == movie_id].iloc[0]
        random_row["mappedUserId"] = user_id
        user_embedding, movie_embedding = self.get_embedding_cb(self.data, user_id, movie_id)
        random_row["user_embedding"] = user_embedding.to("cpu").detach().tolist()
        random_row["movie_embedding"] = movie_embedding.to("cpu").detach().tolist()
        random_row["prompt"] = row_to_prompt_datapoint(random_row, self.kge_dimension, sep_token=self.tokenizer.sep_token)
        tokenized = self.tokenizer(random_row["prompt"], padding="max_length", truncation=True)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label
            }
    
class VanillaEmbeddingDataCollator(TextBasedDataCollator):
    '''
    The vanilla data collator does only generate false edges with the prompt, title, user_id and genres.
    '''
    def __init__(self, tokenizer, df, false_ratio = 2.0):
        super().__init__(tokenizer=tokenizer,false_ratio = false_ratio, df = df)


    def _transform_to_false_example(self):
        label = 0
        user_id, movie_id = self._find_non_existing_user_movie()
        random_row = self.df[self.df["mappedMovieId"] == movie_id].iloc[0]
        random_row["mappedUserId"] = user_id
        random_row["prompt"] = row_to_vanilla_datapoint(random_row, self.tokenizer.sep_token)
        tokenized = self.tokenizer(random_row["prompt"], padding="max_length", truncation=True)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label
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
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.test_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
    
class InsertEmbeddingBertForSequenceClassification(BertForSequenceClassification):
    
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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.bert.embeddings(input_ids)
        if graph_embeddings is not None and len(graph_embeddings) > 0:
            
            if attention_mask is not None:
                mask = ((attention_mask.sum(dim = 1) -1).unsqueeze(1).repeat((1,2))-torch.tensor([3,1]).to(self.device)).unsqueeze(2).repeat((1,1,self.config.hidden_size))        
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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ClassifierBase():
    def __init__(self, df, semantic_datapoints, batch_size = 64, force_recompute = False) -> None:
        self.predictions = None
        self.df = df
        self.batch_size = batch_size
        self.semantic_datapoints = semantic_datapoints
        self.force_recompute = force_recompute
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return METRIC.compute(predictions=predictions, references=labels)
    
    def train_model_on_data(self, dataset, epochs = 3):
        trainer = self._set_up_trainer(dataset, epochs = epochs)

        # Train the model
        trainer.train()

        trainer.model.to(device = "cpu").save_pretrained(self.best_model_path)
        trainer.model.to(device = self.device)

    
    def evaluate_model_on_data(self, dataset, split):
        if split == "test":
            trainer = self._set_up_trainer(dataset["test"])
            test_results = trainer.evaluate(eval_dataset = dataset["test"])
        else:
            trainer = self._set_up_trainer(dataset["val"], self.eval_data_collator)
            test_results = trainer.evaluate(eval_dataset = dataset["val"])

        print(test_results)

    def plot_attentions(self, attentions, title):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(attentions, xticklabels=self.semantic_datapoints, yticklabels=self.semantic_datapoints, cmap='viridis', ax=ax)
        plt.title(title)
        plt.xlabel('Tokens')
        plt.ylabel('Tokens')
        plt.show()

    def plot_attention_graph(self, attentions, title, weight_coef = 15):
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
        nx.draw(G, pos=pos, with_labels = False)

        edge_weights = nx.get_edge_attributes(G, 'weight')

        # Create a list of edge thicknesses based on weights
        # Normalize the weights to get thickness values
        max_weight = max(edge_weights.values())
        edge_thickness = [edge_weights[edge] / max_weight * weight_coef for edge in G.edges()]# Draw edges with varying thicknesses
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=edge_thickness)
        shift_to_left = 0.21
        label_pos = {node: (x - shift_to_left, y) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(G,label_pos,labels)
        # Get the current axis limits
        x_min, x_max = plt.xlim()

        # Adjust x-axis limits to add more whitespace on the left
        plt.xlim(x_min - (shift_to_left+0.1), x_max)  # Increase the left limit by 0.1
        plt.title(title)
        plt.show()

    
    

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


    def _find_sep_in_tokens(self, tokens):
        return tokens == self.tokenizer.sep_token_id
    
    def _get_ranges_over_batch(self, input_ids):
        mask = self._find_sep_in_tokens(input_ids)
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
        ranges_over_batch = torch.stack([ranges_over_batch[:, :-1], ranges_over_batch[:, 1:]], dim=2)
        # Create a tensor of zeros to represent the starting points
        start_points = torch.zeros(ranges_over_batch.size(0), 1, 2, dtype=ranges_over_batch.dtype)

        # Set the second column to be the first element of the first range
        start_points[:, 0, 1] = ranges_over_batch[:, 0, 0]

        # Concatenate the start_points tensor with the original ranges_over_batch tensor
        ranges_over_batch = torch.cat((start_points, ranges_over_batch), dim=1)
        ranges_over_batch[:,:,0] += 1
        return ranges_over_batch
    
    def generate_semantic_attention_matrix(self, dataset, split = "val", batch_size = 64, epochs = 3, step_info_size = 10):
        self.model.eval()
        layers = self.model.config.num_hidden_layers
        data_collator = self._get_data_collator(split)
        result_matrix = torch.zeros((layers, len(self.semantic_datapoints), len(self.semantic_datapoints)))
        for epoch in range(epochs):
            data_loader = DataLoader(dataset=dataset[split], batch_size= batch_size, collate_fn = data_collator)
            #batch = next(iter(data_loader))
            for idx, batch in enumerate(data_loader):
                combined_attentions = self._generate_attentions_for_batch(batch)
                ranges_over_batch = self._get_ranges_over_batch()
                # Initialize the result matrix
                for layer in range(layers):
                    for batch_, ranges in enumerate(ranges_over_batch):
                        for idx, range_x in enumerate(ranges):
                            from_range = torch.arange(range_x[0],range_x[1])
                            for idy, range_y in enumerate(ranges):
                                to_range = torch.arange(range_y[0], range_y[1])
                                result_matrix[layer, idx, idy] += torch.sum(combined_attentions[layer, batch_, from_range][:, to_range])/(len(from_range)*len(to_range))
                if idx > 0 and idx % step_info_size ==0:
                    print(f"Epoch: {epoch}, Batch: {idx}/{len(data_loader)}")
        normalized_tensor = result_matrix / result_matrix.sum(dim = (1,2), keepdim=True)
        normalized_tensor_row = result_matrix / result_matrix.sum(dim=2, keepdim=True)
        return normalized_tensor, normalized_tensor_row
    
    def _generate_attentions_for_batch(self, batch):
        with torch.no_grad():
            outputs = self.forward_batch(batch, output_attentions = True)
            return torch.stack([torch.sum(attentions, dim=1).detach() for attentions in outputs.attentions])  # This will contain the attention weights for each layer and head
    
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
    def __init__(self, df, semantic_datapoints, model_name = "google/bert_uncased_L-2_H-128_A-2", batch_size = 64, model_max_length = 256, force_recompute = False) -> None:
        super().__init__( df = df, semantic_datapoints = semantic_datapoints, batch_size = batch_size, force_recompute = force_recompute)
        self.model_name = model_name
        
        # Initialize the model and tokenizer
        if os.path.exists(self.best_model_path) and not self.force_recompute:
            self.model = BertForSequenceClassification.from_pretrained(self.best_model_path, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, model_max_length=model_max_length)

    def tokenize_function(self, example, return_pt = False):
        if return_pt:
            tokenized =  self.tokenizer(example["prompt"], padding="max_length", truncation=True, return_tensors = "pt")
        else:
            tokenized =  self.tokenizer(example["prompt"], padding="max_length", truncation=True)
        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": example["labels"]}
        return result
        
    def forward_batch(self, batch, output_attentions = False, output_hidden_states = False):
        return self.model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], output_attentions=output_attentions, output_hidden_states = output_hidden_states)
        
class AddingEmbeddingsBertClassifierBase(ClassifierBase):
    def __init__(self, movie_lens_loader, get_embedding_cb, model_name = "google/bert_uncased_L-2_H-128_A-2", kge_dimension = 128, batch_size = 64,model_max_length = 256, force_recompute = False) -> None:
        super().__init__(df = movie_lens_loader.llm_df, semantic_datapoints = EMBEDDING_BASED_SEMANTIC_TOKENS, batch_size = batch_size, force_recompute = force_recompute)
        self.kge_dimension = kge_dimension
        self.best_model_path = LLM_ADDING_BEST_MODEL_PATH.format(self.kge_dimension)
        self.model_name = model_name
        self.attentions_path = ADDING_ATTENTIONS_PATH.format(self.kge_dimension)
        self.hidden_states_path = ADDING_HIDDEN_STATES_PATH.format(self.kge_dimension)
        self.graph_embeddings_path = ADDING_GRAPH_EMBEDDINGS_PATH.format(self.kge_dimension)
        self.tokens_path = ADDING_TOKENS_PATH.format(self.kge_dimension)
        
        # Initialize the model and tokenizer
        if os.path.exists(self.best_model_path) and not self.force_recompute:
            self.model = InsertEmbeddingBertForSequenceClassification.from_pretrained(self.best_model_path, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
        else:
            self.model = InsertEmbeddingBertForSequenceClassification.from_pretrained(self.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, model_max_length=model_max_length)
        self.train_data_collator = EmbeddingBasedDataCollator(self.tokenizer, self.device, movie_lens_loader.llm_df, movie_lens_loader.gnn_train_data, get_embedding_cb, kge_dimension=self.kge_dimension)
        self.test_data_collator = EmbeddingBasedDataCollator(self.tokenizer, self.device, movie_lens_loader.llm_df, movie_lens_loader.gnn_test_data, get_embedding_cb, kge_dimension=self.kge_dimension)
        self.eval_data_collator = EmbeddingBasedDataCollator(self.tokenizer, self.device, movie_lens_loader.llm_df, movie_lens_loader.gnn_val_data, get_embedding_cb, kge_dimension=self.kge_dimension)

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
        if return_pt:
            tokenized =  self.tokenizer(example["prompt"], padding="max_length", truncation=True, return_tensors = "pt")
        else:
            tokenized =  self.tokenizer(example["prompt"], padding="max_length", truncation=True)
        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": example["labels"],
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
        cm = confusion_matrix(labels, preds)

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    def _get_data_collator(self, split):
        return self.test_data_collator if split == "test" else self.eval_data_collator if split == "val" else self.train_data_collator
    
    def plot_training_loss_and_accuracy(self, kge_dimension = 128):
        model_type = "Embedding"
        self._plot_training_loss_and_accuracy(ADDING_TRAINING_STATE_PATH.format(kge_dimension), model_type)

    def forward_dataset_and_save_outputs(self, dataset, splits = ["train", "test","val"], batch_size = 64, epochs = 3, load_fields = ["attentions", "hidden_states", "graph_embeddings"], force_recompute = False):
        if force_recompute or not os.path.exists(self.attentions_path) or not os.path.exists(self.hidden_states_path) or not os.path.exists(self.tokens_path) or not os.path.exists(self.graph_embeddings_path):
            self.model.eval()
            Path(ADDING_ATTENTIONS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(ADDING_HIDDEN_STATES_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(ADDING_INPUT_IDS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(ADDING_RANGES_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(ADDING_SUB_TOKENS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(ADDING_GRAPH_EMBEDDINGS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                for split in splits:
                    data_collator = self._get_data_collator(split)
                    for epoch in range(epochs):
                        last_hidden_states = []
                        all_attentions = []
                        all_graph_embeddings = []
                        all_tokens = []
                        print(f"Adding {split} Forward Epoch {epoch + 1} from {epochs}")
                        data_loader = DataLoader(dataset=dataset[split], batch_size= batch_size, collate_fn = data_collator)
                        for idx, batch in enumerate(data_loader):
                        #if True:
                        #    batch = next(iter(data_loader))
                            splits_ = [split] *  len(batch["input_ids"])
                            outputs = self.model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], graph_embeddings = batch["graph_embeddings"], output_hidden_states=True, output_attentions = True)
                            ranges_over_batch = self._get_ranges_over_batch(batch["input_ids"])
                            if "attentions" in load_fields:
                                attentions = outputs.attentions
                                attentions = [torch.sum(layer, dim=1) for layer in attentions]
                                attentions = torch.stack(attentions).permute(1,2,3,0)
                                attentions = means_over_ranges_cross(ranges_over_batch, attentions)
                                all_attentions.append(attentions.numpy())
                                del attentions
                            if "hidden_states" in load_fields:
                                hidden_states = outputs.hidden_states[-1]
                                hidden_states = avg_over_last_hidden_states(ranges_over_batch, hidden_states).permute((1,0,2))
                                last_hidden_states.append(hidden_states.numpy())
                                del hidden_states
                            tokens = self.get_tokens_as_df(batch["input_ids"], ranges_over_batch)
                            if "graph_embeddings" in load_fields:
                                all_graph_embeddings.append(batch["graph_embeddings"].numpy())
                            tokens["labels"] = batch["labels"].tolist()
                            tokens["split"] = splits_
                            all_tokens.append(tokens)



                        # Concatenate all hidden states across batches
                        all_tokens = pd.concat(all_tokens).reset_index()
                        all_tokens.to_csv(ADDING_SUB_TOKENS_PATH.format(self.kge_dimension, split, epoch), index = False)
                        del all_tokens             
                        if "attentions" in load_fields:
                            all_attentions = np.concatenate(all_attentions)
                            np.save(ATTENTIONS_ADDING_PATH.format(self.kge_dimension, split, epoch), all_attentions)
                            del all_attentions
                        if "hidden_states" in load_fields:
                            last_hidden_states = np.concatenate(last_hidden_states)
                            np.save(LAST_HIDDEN_STATES_ADDING_PATH.format(self.kge_dimension, split, epoch), last_hidden_states)
                            del last_hidden_states
                        if "graph_embeddings" in load_fields:
                            all_graph_embeddings = np.concatenate(all_graph_embeddings)
                            np.save(GRAPH_EMBEDDINGS_ADDING_PATH.format(self.kge_dimension, split, epoch), all_graph_embeddings)
                            del all_graph_embeddings

                # all_tokens
                all_tokens = []
                for split in splits:
                    for epoch in range(epochs):
                        all_tokens.append(pd.read_csv(ADDING_SUB_TOKENS_PATH.format(self.kge_dimension, split, epoch)))
                all_tokens = pd.concat(all_tokens).reset_index()
                all_tokens.to_csv(self.tokens_path, index = False)

                # hidden states:
                if "hidden_states" in load_fields:
                    last_hidden_states = []
                    for split in splits:
                        for epoch in range(epochs):
                            last_hidden_states.append(torch.from_numpy(np.load(LAST_HIDDEN_STATES_ADDING_PATH.format(self.kge_dimension, split, epoch))))
                    last_hidden_states = torch.cat(last_hidden_states)
                    torch.save(last_hidden_states, self.hidden_states_path)
                    all_tokens["hidden_states"] = last_hidden_states.unbind()
                    del last_hidden_states

                # attentions:
                if "attentions" in load_fields:
                    attentions = []
                    for split in splits:
                        for epoch in range(epochs):
                            attentions_ = torch.from_numpy(np.load(ATTENTIONS_ADDING_PATH.format(self.kge_dimension, split, epoch)))
                            attentions.append(attentions_)
                    attentions = torch.cat(attentions)
                    torch.save(attentions, self.attentions_path)
                    all_tokens["attentions"] = attentions.unbind()
                    del attentions

                # graph embeddings:
                if "graph_embeddings" in load_fields:
                    graph_embeddings = []
                    for split in splits:
                        for epoch in range(epochs):
                            graph_embeddings.append(torch.from_numpy(np.load(GRAPH_EMBEDDINGS_ADDING_PATH.format(self.kge_dimension, split, epoch))))
                    graph_embeddings = torch.cat(graph_embeddings)
                    torch.save(graph_embeddings, self.graph_embeddings_path)
                    all_tokens["graph_embeddings"] = graph_embeddings.unbind()
                    del graph_embeddings
        else:
            all_tokens = pd.read_csv(self.tokens_path)
            if "hidden_states" in load_fields:
                averaged_hidden_states = torch.load(self.hidden_states_path)
                all_tokens["hidden_states"] = torch.unbind(averaged_hidden_states)
                del averaged_hidden_states
            if "attentions" in load_fields:
                averaged_attentions = torch.load(self.attentions_path)
                all_tokens["attentions"] = torch.unbind(averaged_attentions)
                del averaged_attentions
            if "graph_embeddings" in load_fields:
                graph_embeddings = torch.load(self.graph_embeddings_path)
                all_tokens["graph_embeddings"] = torch.unbind(graph_embeddings)
                del graph_embeddings
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
            decoded = self.tokenizer.batch_decode(gather, skip_special_tokens = True)
            if pos == 0:
                semantic_tokens.extend([decode[len("user : "):] for decode in decoded])
            if pos == 1:
                semantic_tokens.extend([decode[len("title : "):] for decode in decoded])
            if pos == 2:
                semantic_tokens.extend([decode[len("genres : "):] for decode in decoded])
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
        
    def forward_batch(self, batch, output_attentions = False, output_hidden_states = False):
        return self.model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], graph_embeddings = batch["graph_embeddings"], output_attentions=output_attentions, output_hidden_states = output_hidden_states)


    
    
class PromptBertClassifier(ClassifierOriginalArchitectureBase):
    def __init__(self, movie_lens_loader, get_embedding_cb, model_name = "google/bert_uncased_L-2_H-128_A-2", kge_dimension = 4, batch_size = 64,model_max_length = 256, force_recompute = False) -> None:
        assert kge_dimension <= 16
        self.kge_dimension = kge_dimension
        self.best_model_path = LLM_PROMPT_BEST_MODEL_PATH.format(self.kge_dimension)
        self.attentions_path = PROMPT_ATTENTIONS_PATH.format(self.kge_dimension)
        self.hidden_states_path = PROMPT_HIDDEN_STATES_PATH.format(self.kge_dimension)
        self.graph_embeddings_path = PROMPT_GRAPH_EMBEDDINGS_PATH.format(self.kge_dimension)
        self.tokens_path = PROMPT_TOKENS_PATH.format(self.kge_dimension)
        super().__init__(df = movie_lens_loader.llm_df, semantic_datapoints=EMBEDDING_BASED_SEMANTIC_TOKENS, model_name=model_name, force_recompute=force_recompute, batch_size = batch_size,model_max_length = model_max_length)
        self.train_data_collator = PromptEmbeddingDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_train_data, get_embedding_cb, kge_dimension = kge_dimension)
        self.test_data_collator = PromptEmbeddingDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_test_data, get_embedding_cb, kge_dimension = kge_dimension)
        self.eval_data_collator = PromptEmbeddingDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_val_data, get_embedding_cb, kge_dimension = kge_dimension)

    
    
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
        cm = confusion_matrix(labels, preds)

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    def plot_training_loss_and_accuracy(self, kge_dimension = 4):
        model_type = "Prompt"
        self._plot_training_loss_and_accuracy(PROMPT_TRAINING_STATE_PATH.format(kge_dimension), model_type)

    def forward_dataset_and_save_outputs(self, dataset, splits = ["train", "test","val"], batch_size = 64, epochs = 3, load_fields = ["attentions", "hidden_states", "graph_embeddings"], force_recompute = False):
        if force_recompute or not os.path.exists(self.attentions_path) or not os.path.exists(self.hidden_states_path) or not os.path.exists(self.tokens_path) or not os.path.exists(self.graph_embeddings_path):
            self.model.eval()
            Path(PROMPT_ATTENTIONS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_HIDDEN_STATES_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_INPUT_IDS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_RANGES_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_SUB_TOKENS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            Path(PROMPT_GRAPH_EMBEDDINGS_DIR_PATH.format(self.kge_dimension)).mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                for split in splits:
                    data_collator = self._get_data_collator(split)
                    for epoch in range(epochs):
                        last_hidden_states = []
                        all_attentions = []
                        all_graph_embeddings = []
                        all_tokens = []
                        print(f"Prompt {split} Forward Epoch {epoch + 1} from {epochs}")
                        data_loader = DataLoader(dataset=dataset[split], batch_size= batch_size, collate_fn = data_collator)
                        for idx, batch in enumerate(data_loader):
                        #if True:
                        #    batch = next(iter(data_loader))
                            splits_ = [split] *  len(batch["input_ids"])
                            outputs = self.model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], output_hidden_states=True, output_attentions = True)
                            ranges_over_batch = self._get_ranges_over_batch(batch["input_ids"])
                            if "attentions" in load_fields:
                                attentions = outputs.attentions
                                attentions = [torch.sum(layer, dim=1) for layer in attentions]
                                attentions = torch.stack(attentions).permute(1,2,3,0)
                                attentions = means_over_ranges_cross(ranges_over_batch, attentions)
                                all_attentions.append(attentions.numpy())
                                del attentions
                            if "hidden_states" in load_fields:
                                hidden_states = outputs.hidden_states[-1]
                                hidden_states = avg_over_last_hidden_states(ranges_over_batch, hidden_states).permute((1,0,2))
                                last_hidden_states.append(hidden_states.numpy())
                                del hidden_states
                            tokens, graph_embeddings = self.get_tokens_as_df(batch["input_ids"], ranges_over_batch)
                            if "graph_embeddings" in load_fields:
                                all_graph_embeddings.append(graph_embeddings.numpy())
                            del graph_embeddings
                            tokens["labels"] = batch["labels"].tolist()
                            tokens["split"] = splits_
                            all_tokens.append(tokens)



                        # Concatenate all hidden states across batches
                        all_tokens = pd.concat(all_tokens).reset_index()
                        all_tokens.to_csv(PROMPT_SUB_TOKENS_PATH.format(self.kge_dimension, split, epoch), index = False)
                        del all_tokens             
                        if "attentions" in load_fields:
                            all_attentions = np.concatenate(all_attentions)
                            np.save(ATTENTIONS_PROMPT_PATH.format(self.kge_dimension, split, epoch), all_attentions)
                            del all_attentions
                        if "hidden_states" in load_fields:
                            last_hidden_states = np.concatenate(last_hidden_states)
                            np.save(LAST_HIDDEN_STATES_PROMPT_PATH.format(self.kge_dimension, split, epoch), last_hidden_states)
                            del last_hidden_states
                        if "graph_embeddings" in load_fields:
                            all_graph_embeddings = np.concatenate(all_graph_embeddings)
                            np.save(GRAPH_EMBEDDINGS_PROMPT_PATH.format(self.kge_dimension, split, epoch), all_graph_embeddings)
                            del all_graph_embeddings

                # all_tokens
                all_tokens = []
                for split in splits:
                    for epoch in range(epochs):
                        all_tokens.append(pd.read_csv(PROMPT_SUB_TOKENS_PATH.format(self.kge_dimension, split, epoch)))
                all_tokens = pd.concat(all_tokens).reset_index()
                all_tokens.to_csv(self.tokens_path, index = False)

                # hidden states:
                if "hidden_states" in load_fields:
                    last_hidden_states = []
                    for split in splits:
                        for epoch in range(epochs):
                            last_hidden_states.append(torch.from_numpy(np.load(LAST_HIDDEN_STATES_PROMPT_PATH.format(self.kge_dimension, split, epoch))))
                    last_hidden_states = torch.cat(last_hidden_states)
                    torch.save(last_hidden_states, self.hidden_states_path)
                    all_tokens["hidden_states"] = last_hidden_states.unbind()
                    del last_hidden_states

                # attentions:
                if "attentions" in load_fields:
                    attentions = []
                    for split in splits:
                        for epoch in range(epochs):
                            attentions.append(torch.from_numpy(np.load(ATTENTIONS_PROMPT_PATH.format(self.kge_dimension, split, epoch))))
                    attentions = torch.cat(attentions)
                    torch.save(attentions, self.attentions_path)
                    all_tokens["attentions"] = attentions.unbind()
                    del attentions

                # graph embeddings:
                if "graph_embeddings" in load_fields:
                    graph_embeddings = []
                    for split in splits:
                        for epoch in range(epochs):
                            graph_embeddings.append(torch.from_numpy(np.load(GRAPH_EMBEDDINGS_PROMPT_PATH.format(self.kge_dimension, split, epoch))))
                    graph_embeddings = torch.cat(graph_embeddings)
                    torch.save(graph_embeddings, self.graph_embeddings_path)
                    all_tokens["graph_embeddings"] = graph_embeddings.unbind()
                    del graph_embeddings
        else:
            all_tokens = pd.read_csv(self.tokens_path)
            if "hidden_states" in load_fields:
                averaged_hidden_states = torch.load(self.hidden_states_path)
                all_tokens["hidden_states"] = torch.unbind(averaged_hidden_states)
                del averaged_hidden_states
            if "attentions" in load_fields:
                averaged_attentions = torch.load(self.attentions_path)
                all_tokens["attentions"] = torch.unbind(averaged_attentions)
                del averaged_attentions
            if "graph_embeddings" in load_fields:
                graph_embeddings = torch.load(self.graph_embeddings_path)
                all_tokens["graph_embeddings"] = torch.unbind(graph_embeddings)
                del graph_embeddings
        return all_tokens

    def get_tokens_as_df(self, input_ids, all_ranges_over_batch) -> pd.DataFrame:
        user_ids = []
        titles = []
        genres = []
        user_embeddings = []
        movie_embeddings = []
        all_semantic_tokens = [user_ids, titles, genres, user_embeddings, movie_embeddings]
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
            decoded = self.tokenizer.batch_decode(gather, skip_special_tokens = True)
            if pos == 0:
                semantic_tokens.extend([decode[len("user : "):] for decode in decoded])
            if pos == 1:
                semantic_tokens.extend([decode[len("title : "):] for decode in decoded])
            if pos == 2:
                semantic_tokens.extend([decode[len("genres : "):] for decode in decoded])
            if pos == 3:
                semantic_tokens.extend([decode[len("user_embeddings :"):] for decode in decoded])
            if pos == 4:
                semantic_tokens.extend([decode[len("movie_embeddings :"):] for decode in decoded])
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
    def __init__(self, df, model_name = "google/bert_uncased_L-2_H-128_A-2", batch_size = 64,model_max_length = 256, force_recompute = False) -> None:
        self.best_model_path = LLM_VANILLA_BEST_MODEL_PATH
        self.attentions_path = VANILLA_ATTENTIONS_PATH
        self.hidden_states_path = VANILLA_HIDDEN_STATES_PATH
        self.tokens_path = VANILLA_TOKENS_PATH
        super().__init__(df = df, semantic_datapoints=ALL_SEMANTIC_TOKENS, model_name=model_name, batch_size = batch_size, model_max_length = model_max_length, force_recompute=force_recompute)
        self.data_collator = VanillaEmbeddingDataCollator(self.tokenizer, df)
    
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

        # Initialize the Trainer
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
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
        cm = confusion_matrix(labels, preds)

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues)
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
            self.model.eval()
            Path(VANILLA_ATTENTIONS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(VANILLA_HIDDEN_STATES_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(VANILLA_INPUT_IDS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(VANILLA_RANGES_DIR_PATH).mkdir(parents=True, exist_ok=True)
            Path(VANILLA_SUB_TOKENS_DIR_PATH).mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                for split in splits:
                    data_collator = self._get_data_collator(split)
                    for epoch in range(epochs):
                        last_hidden_states = []
                        all_attentions = []
                        all_tokens = []
                        print(f"Vanilla {split} Forward Epoch {epoch + 1} from {epochs}")
                        data_loader = DataLoader(dataset=dataset[split], batch_size= batch_size, collate_fn = data_collator)
                        for idx, batch in enumerate(data_loader):
                        #if True:
                        #    batch = next(iter(data_loader))
                            splits_ = [split] *  len(batch["input_ids"])
                            outputs = self.model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], output_hidden_states=True, output_attentions = True)
                            ranges_over_batch = self._get_ranges_over_batch(batch["input_ids"])
                            if "attentions" in load_fields:
                                attentions = outputs.attentions
                                attentions = [torch.sum(layer, dim=1) for layer in attentions]
                                attentions = torch.stack(attentions).permute(1,2,3,0)
                                attentions = means_over_ranges_cross(ranges_over_batch, attentions)
                                all_attentions.append(attentions.numpy())
                                del attentions
                            if "hidden_states" in load_fields:
                                hidden_states = outputs.hidden_states[-1]
                                hidden_states = avg_over_last_hidden_states(ranges_over_batch, hidden_states).permute((1,0,2))
                                last_hidden_states.append(hidden_states.numpy())
                                del hidden_states
                            tokens = self.get_tokens_as_df(batch["input_ids"], ranges_over_batch)
                            tokens["labels"] = batch["labels"].tolist()
                            tokens["split"] = splits_
                            all_tokens.append(tokens)



                        # Concatenate all hidden states across batches
                        all_tokens = pd.concat(all_tokens).reset_index()
                        all_tokens.to_csv(VANILLA_SUB_TOKENS_PATH.format(split, epoch), index = False)
                        del all_tokens
                        if "attentions" in load_fields:
                            all_attentions = np.concatenate(all_attentions)
                            np.save(ATTENTIONS_VANILLA_PATH.format(split, epoch), all_attentions)
                            del all_attentions
                        if "hidden_states" in load_fields:
                            last_hidden_states = np.concatenate(last_hidden_states)
                            np.save(LAST_HIDDEN_STATES_VANILLA_PATH.format(split, epoch), last_hidden_states)
                            del last_hidden_states

                # all_tokens
                all_tokens = []
                for split in splits:
                    for epoch in range(epochs):
                        all_tokens.append(pd.read_csv(VANILLA_SUB_TOKENS_PATH.format(split, epoch)))
                all_tokens = pd.concat(all_tokens).reset_index()
                all_tokens.to_csv(self.tokens_path, index = False)

                # hidden states:
                if "hidden_states" in load_fields:
                    last_hidden_states = []
                    for split in splits:
                        for epoch in range(epochs):
                            last_hidden_states.append(torch.from_numpy(np.load(LAST_HIDDEN_STATES_VANILLA_PATH.format(split, epoch))))
                    last_hidden_states = torch.cat(last_hidden_states)
                    torch.save(last_hidden_states, self.hidden_states_path)
                    all_tokens["hidden_states"] = last_hidden_states.unbind()
                    del last_hidden_states

                # attentions:
                if "attentions" in load_fields:
                    attentions = []
                    for split in splits:
                        for epoch in range(epochs):
                            attentions.append(torch.from_numpy(np.load(ATTENTIONS_VANILLA_PATH.format(split, epoch))))
                    attentions = torch.cat(attentions)
                    torch.save(attentions, self.attentions_path)
                    all_tokens["attentions"] = attentions.unbind()
                    del attentions

        else:
            all_tokens = pd.read_csv(self.tokens_path)
            if "hidden_states" in load_fields:
                averaged_hidden_states = torch.load(self.hidden_states_path)
                all_tokens["hidden_states"] = torch.unbind(averaged_hidden_states)
                del averaged_hidden_states
            if "attentions" in load_fields:
                averaged_attentions = torch.load(self.attentions_path)
                all_tokens["attentions"] = torch.unbind(averaged_attentions)
                del averaged_attentions
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
            decoded = self.tokenizer.batch_decode(gather, skip_special_tokens = True)
            if pos == 0:
                semantic_tokens.extend([decode[len("user : "):] for decode in decoded])
            if pos == 1:
                semantic_tokens.extend([decode[len("title : "):] for decode in decoded])
            if pos == 2:
                semantic_tokens.extend([decode[len("genres : "):] for decode in decoded])
        all_semantic_tokens[0] = [int(id) for id in all_semantic_tokens[0]]
        all_semantic_tokens[2] = [ast.literal_eval(string_list) for string_list in all_semantic_tokens[2]]
        data = {"user_id": all_semantic_tokens[0], "title": all_semantic_tokens[1], "genres": all_semantic_tokens[2]}
        df = pd.DataFrame(data)
        return df

    def _get_data_collator(self, split = None):
        return self.data_collator
        