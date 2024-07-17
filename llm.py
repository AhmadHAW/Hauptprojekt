import random as rd
from typing import Optional, Union
from abc import ABC, abstractmethod
import os

from movie_lens_loader import row_to_prompt_datapoint, row_to_vanilla_datapoint, LLM_PROMPT_TRAINING_PATH, LLM_VANILLA_TRAINING_PATH, LLM_PROMPT_BEST_MODEL_PATH, LLM_VANILLA_BEST_MODEL_PATH

import torch
import datasets
import numpy as np
import evaluate
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.utils import is_datasets_available
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


ID2LABEL = {0: "FALSE", 1: "TRUE"}
LABEL2ID = {"FALSE": 0, "TRUE": 1}
PROMPT_LOG_PATH = f"{LLM_PROMPT_TRAINING_PATH}/logs"
VANILLA_LOG_PATH = f"{LLM_VANILLA_TRAINING_PATH}/logs"

class DataCollatorBase(DataCollatorForLanguageModeling, ABC):
    '''
    The Data Collators are used to generate non-existing edges on the fly. The false ratio allows to decide the ratio,
    in existing edges are replaced with non-existing edges.
    '''
    def __init__(self, tokenizer, false_ratio = 2.0):
        super().__init__(tokenizer=tokenizer, mlm=False)
        assert false_ratio > 0
        self.false_ratio = false_ratio
        self.tokenizer = tokenizer

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
        input_ids = torch.tensor([f["input_ids"] for f in new_features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in new_features], dtype=torch.long)
        labels = torch.tensor([f["labels"] for f in new_features], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _find_non_existing_user_movie(self):
        while True:
            user_id = np.random.choice(self.df["mappedUserId"].unique())
            movie_id = np.random.choice(self.df["mappedMovieId"].unique())
            if not ((self.df["mappedUserId"] == user_id) & (self.df["mappedMovieId"] == movie_id)).any():
                return user_id, movie_id

class PromptEmbeddingDataCollator(DataCollatorBase):
    '''
    The Prompt Data Collator also adds embeddings to the prompt on the fly.
    '''
    def __init__(self, tokenizer, df, data, get_embedding_cb, kge_dimension = 4, false_ratio = 2.0):
        super().__init__(tokenizer=tokenizer,false_ratio = false_ratio)
        self.df = df
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
        random_row["prompt"] = row_to_prompt_datapoint(random_row, self.kge_dimension)
        tokenized = self.tokenizer(random_row["prompt"], padding="max_length", truncation=True)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label
            }
    
class VanillaEmbeddingDataCollator(DataCollatorBase):
    '''
    The vanilla data collator does only generate false edges with the prompt, title, user_id and genres.
    '''
    def __init__(self, tokenizer, df, false_ratio = 2.0):
        super().__init__(tokenizer=tokenizer,false_ratio = false_ratio)
        self.df = df


    def _transform_to_false_example(self):
        label = 0
        user_id, movie_id = self._find_non_existing_user_movie()
        random_row = self.df[self.df["mappedMovieId"] == movie_id].iloc[0]
        random_row["mappedUserId"] = user_id
        random_row["prompt"] = row_to_vanilla_datapoint(random_row)
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
    
class ClassifierBase(ABC):
    def __init__(self, df, model_name = "google/bert_uncased_L-2_H-128_A-2", batch_size = 64, kge_dimension = 4, force_recompute = False) -> None:
        assert kge_dimension <= 16
        self.model_name = model_name
        self.predictions = None
        self.df = df
        self.trainer = None
        self.batch_size = batch_size
        self.kge_dimension = kge_dimension
        # Initialize the model and tokenizer
        if os.path.exists(self.best_model_path) and not force_recompute:
            self.model = BertForSequenceClassification.from_pretrained(self.best_model_path, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

        model_max_length = 256 if kge_dimension <= 8 else 512
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
    
    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return METRIC.compute(predictions=predictions, references=labels)
    
    def train_model_on_data(self, dataset, epochs = 3):
        self._set_up_trainer(dataset, epochs = epochs)

        # Train the model
        self.trainer.train()

        self.trainer.model.save_pretrained(self.best_model_path)

    
    def evaluate_model_on_data(self, dataset, split):
        if split == "test":
            self._set_up_trainer(dataset["test"])
            test_results = self.trainer.evaluate(eval_dataset = dataset["test"])
        else:
            self._set_up_trainer(dataset["val"], self.eval_data_collator)
            test_results = self.trainer.evaluate(eval_dataset = dataset["val"])

        print(test_results)

    def plot_confusion_matrix(self, split, dataset, tokenize = False, force_recompute = False):
        if split == "test":
            self._set_up_trainer(dataset, tokenize = tokenize)
            dataset = dataset["test"]
        else:
            self._set_up_trainer(dataset, tokenize = tokenize, eval_data_collator = self.eval_data_collator)
            dataset = dataset["val"]
        if not self.predictions or force_recompute: 
        # Generate predictions
            predictions = self.trainer.predict(dataset)
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


        
        

class PromptEncoderOnlyClassifier(ClassifierBase):
    def __init__(self, movie_lens_loader, get_embedding_cb, model_name = "google/bert_uncased_L-2_H-128_A-2", kge_dimension = 4, force_recompute = False) -> None:
        self.best_model_path = LLM_PROMPT_BEST_MODEL_PATH.format(kge_dimension)
        super().__init__(df = movie_lens_loader.llm_df, model_name=model_name, force_recompute=force_recompute, kge_dimension = kge_dimension)
        self.train_data_collator = PromptEmbeddingDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_train_data, get_embedding_cb, kge_dimension = kge_dimension)
        self.test_data_collator = PromptEmbeddingDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_test_data, get_embedding_cb, kge_dimension = kge_dimension)
        self.eval_data_collator = PromptEmbeddingDataCollator(self.tokenizer, movie_lens_loader.llm_df, movie_lens_loader.gnn_val_data, get_embedding_cb, kge_dimension = kge_dimension)

    
    
    def _set_up_trainer(self, dataset, tokenize = False, eval_data_collator = None, epochs = 3):
        if not self.trainer:
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
            self.trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                data_collator=self.train_data_collator,
                eval_data_collator=eval_data_collator,
                compute_metrics=self._compute_metrics,
            )


class VanillaEncoderOnlyClassifier(ClassifierBase):
    def __init__(self, df, model_name = "google/bert_uncased_L-2_H-128_A-2", kge_dimension = 4, force_recompute = False) -> None:
        self.best_model_path = LLM_VANILLA_BEST_MODEL_PATH.format(kge_dimension)
        super().__init__(df = df, model_name=model_name, kge_dimension = kge_dimension, force_recompute=force_recompute)
        self.data_collator = VanillaEmbeddingDataCollator(self.tokenizer, df)
    
    def _set_up_trainer(self, dataset, tokenize = False, epochs = 3):
        if tokenize:
            tokenized_dataset = dataset.map(self.tokenize_function, batched = True)
        else:
            tokenized_dataset = dataset
        training_args = TrainingArguments(
            output_dir=LLM_VANILLA_TRAINING_PATH.format(self.kge_dimension),
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=VANILLA_LOG_PATH.format(self.kge_dimension),
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True
        )

        # Initialize the Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
        )
        