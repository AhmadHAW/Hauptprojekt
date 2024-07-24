import random as rd
from typing import Optional, Union, Dict, Tuple, List
from abc import ABC
import os

from movie_lens_loader import row_to_prompt_datapoint, row_to_adding_embedding_datapoint, row_to_vanilla_datapoint, LLM_PROMPT_TRAINING_PATH, LLM_ADDING_TRAINING_PATH, LLM_VANILLA_TRAINING_PATH, LLM_PROMPT_BEST_MODEL_PATH, LLM_ADDING_BEST_MODEL_PATH, LLM_VANILLA_BEST_MODEL_PATH

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import datasets
import numpy as np
import evaluate
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertModel, BertTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import is_datasets_available
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ID2LABEL = {0: "FALSE", 1: "TRUE"}
LABEL2ID = {"FALSE": 0, "TRUE": 1}
PROMPT_LOG_PATH = f"{LLM_PROMPT_TRAINING_PATH}/logs"
ADDING_LOG_PATH = f"{LLM_ADDING_TRAINING_PATH}/logs"
VANILLA_LOG_PATH = f"{LLM_VANILLA_TRAINING_PATH}/logs"

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
    def __init__(self, tokenizer, df, data, get_embedding_cb, kge_dimension = 128, false_ratio = 2.0):
        super().__init__(tokenizer=tokenizer,false_ratio = false_ratio, df = df)
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
                f["graph_embeddings"] = f["graph_embeddings"].to(torch.device('cpu'))
        graph_embeddings = torch.stack([f["graph_embeddings"] for f in features])
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
        random_row["user_embedding"] = user_embedding
        random_row["movie_embedding"] = movie_embedding
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
                inputs_embeds = inputs_embeds.scatter(1, mask, graph_embeddings)
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
    def __init__(self, df, batch_size = 64, force_recompute = False) -> None:
        self.predictions = None
        self.df = df
        self.batch_size = batch_size
        self.force_recompute = force_recompute
    
    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return METRIC.compute(predictions=predictions, references=labels)
    
    def train_model_on_data(self, dataset, epochs = 3):
        trainer = self._set_up_trainer(dataset, epochs = epochs)

        # Train the model
        trainer.train()

        trainer.model.to(device = "cpu").save_pretrained(self.best_model_path)

    
    def evaluate_model_on_data(self, dataset, split):
        if split == "test":
            trainer = self._set_up_trainer(dataset["test"])
            test_results = trainer.evaluate(eval_dataset = dataset["test"])
        else:
            trainer = self._set_up_trainer(dataset["val"], self.eval_data_collator)
            test_results = trainer.evaluate(eval_dataset = dataset["val"])

        print(test_results)
        

class ClassifierOriginalArchitectureBase(ClassifierBase):
    def __init__(self, df, model_name = "google/bert_uncased_L-2_H-128_A-2", batch_size = 64, model_max_length = 256, force_recompute = False) -> None:
        super().__init__( df = df, batch_size = batch_size, force_recompute = force_recompute)
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
        
class AddingEmbeddingsBertClassifierBase(ClassifierBase):
    def __init__(self, movie_lens_loader, get_embedding_cb, model_name = "google/bert_uncased_L-2_H-128_A-2", kge_dimension = 128, batch_size = 64,model_max_length = 256, force_recompute = False) -> None:
        super().__init__(df = movie_lens_loader.llm_df, batch_size = batch_size, force_recompute = force_recompute)
        self.kge_dimension = kge_dimension
        self.best_model_path = LLM_ADDING_BEST_MODEL_PATH.format(self.kge_dimension)
        self.model_name = model_name
        
        # Initialize the model and tokenizer
        if os.path.exists(self.best_model_path) and not self.force_recompute:
            self.model = InsertEmbeddingBertForSequenceClassification.from_pretrained(self.best_model_path, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
        else:
            self.model = InsertEmbeddingBertForSequenceClassification.from_pretrained(self.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, model_max_length=model_max_length)
        self.train_data_collator = EmbeddingBasedDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_train_data, get_embedding_cb, kge_dimension=self.kge_dimension)
        self.test_data_collator = EmbeddingBasedDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_test_data, get_embedding_cb, kge_dimension=self.kge_dimension)
        self.eval_data_collator = EmbeddingBasedDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_val_data, get_embedding_cb, kge_dimension=self.kge_dimension)

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

class PromptBertClassifier(ClassifierOriginalArchitectureBase):
    def __init__(self, movie_lens_loader, get_embedding_cb, model_name = "google/bert_uncased_L-2_H-128_A-2", kge_dimension = 4, batch_size = 64,model_max_length = 256, force_recompute = False) -> None:
        assert kge_dimension <= 16
        self.kge_dimension = kge_dimension
        self.best_model_path = LLM_PROMPT_BEST_MODEL_PATH.format(self.kge_dimension)
        super().__init__(df = movie_lens_loader.llm_df, model_name=model_name, force_recompute=force_recompute, batch_size = batch_size,model_max_length = model_max_length)
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


class VanillaBertClassifier(ClassifierOriginalArchitectureBase):
    def __init__(self, df, model_name = "google/bert_uncased_L-2_H-128_A-2", batch_size = 64,model_max_length = 256, force_recompute = False) -> None:
        self.best_model_path = LLM_VANILLA_BEST_MODEL_PATH
        super().__init__(df = df, model_name=model_name, batch_size = batch_size, model_max_length = model_max_length, force_recompute=force_recompute)
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
        