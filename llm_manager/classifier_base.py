from typing import Optional, Tuple, List, Callable
from abc import ABC, abstractmethod
import os
import json
from pathlib import Path

import torch
from torch.nn import Parameter
import datasets
from datasets import DatasetDict
import numpy as np
import pandas as pd
import evaluate
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import (
    mean_over_attention_ranges,
    mean_over_hidden_states,
    replace_ranges,
    get_combinations,
)

from llm_manager.trainer import CustomTrainer
from llm_manager.vanilla.config import (
    VANILLA_TOKEN_TYPE_VALUES,
    VANILLA_TOKEN_TYPE_VALUES_REVERSE,
)
from llm_manager.graph_prompter_hf.config import (
    GRAPH_PROMPTER_TOKEN_TYPE_VALUES,
    GRAPH_PROMPTER_TOKEN_TYPE_VALUES_REVERSE,
)

METRIC = evaluate.load("accuracy")

SPLIT_EPOCH_ENDING = "/split_{}_pos_{}_com_{}.npy"
TOKENS_ENDING = "/tokens.csv"


class ClassifierBase(ABC):
    def __init__(
        self,
        df,
        model,
        tokenizer,
        train_data_collator,
        test_data_collator,
        val_data_collator,  # TODO: make these optional and then use the train data collator if not passed
        root_path,
        device,
        gnn_parameters: Optional[List[Parameter]] = None,
        force_recompute=False,
    ) -> None:
        self.predictions = None  # TODO: remove
        self.df = df  # TODO: remove
        self.model = model
        self.tokenizer = tokenizer  # TODO: remove
        self.train_data_collator = train_data_collator
        self.test_data_collator = test_data_collator
        self.val_data_collator = val_data_collator
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
        self.force_recompute = force_recompute  # TODO: remove
        self.device = device
        self.gnn_parameters = gnn_parameters

    def _get_data_collator(
        self, split
    ) -> DataCollatorForLanguageModeling:  # TODO: remove instead use dict
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
        tokenize=False,  # TODO: remove because depricated
        eval_data_collator=None,  # TODO: remove because depricated
        epochs=3,
        batch_size: int = 64,
    ):
        def _compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return METRIC.compute(predictions=predictions, references=labels)

        use_cpu = False if torch.cuda.is_available() else True

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
            use_cpu=use_cpu,
        )
        if not eval_data_collator:
            eval_data_collator = self.test_data_collator

        # Initialize the Trainer
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=self.train_data_collator,
            eval_data_collator=eval_data_collator,
            compute_metrics=_compute_metrics,
            gnn_parameters=self.gnn_parameters,
        )
        return trainer

    @abstractmethod
    def tokenize_function(self, example, return_pt=False):
        pass

    def train_model_on_data(
        self,
        dataset,
        epochs=3,
        batch_size: int = 64,
    ):
        trainer = self._get_trainer(dataset, epochs=epochs, batch_size=batch_size)

        # Train the model
        trainer.train()

        trainer.model.to(device="cpu").save_pretrained(self.best_model_path)  # type: ignore
        trainer.model.to(device=self.device)  # type: ignore

    @staticmethod  # TODO: move to ExplainabilityModule
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

    def _means_over_ranges_cross(  # TODO: remove
        self,
        all_token_type_ranges: torch.Tensor,
        attentions: torch.Tensor,
    ) -> torch.Tensor:
        attentions = mean_over_attention_ranges(
            attentions,
            all_token_type_ranges[:, :, 0],
            all_token_type_ranges[:, :, 1],
        )
        return attentions

    def forward_dataset_and_save_outputs(
        self,
        dataset: datasets.Dataset | datasets.DatasetDict,
        splits: List[str] = ["train", "test", "val"],
        batch_size: int = 64,
        save_step_size: int = 1,
        load_fields: List[str] = ["attentions", "hidden_states", "logits"],
        force_recompute: bool = False,
        combination_boundaries: Optional[Tuple[int, int]] = None,
    ) -> None:
        add_hidden_states = "hidden_states" in load_fields

        add_attentions = "attentions" in load_fields
        add_logits = "logits" in load_fields
        hidden_states_exist = True
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
            if len(set(dataset[splits[0]][0]["token_type_ids"])) == len(
                set(VANILLA_TOKEN_TYPE_VALUES)
            ):
                # we are a vanilla model
                token_type_values = list(set(VANILLA_TOKEN_TYPE_VALUES))
                token_type_values_reverse = VANILLA_TOKEN_TYPE_VALUES_REVERSE
            else:
                # we are a graph prompter model
                token_type_values = list(set(GRAPH_PROMPTER_TOKEN_TYPE_VALUES))
                token_type_values_reverse = GRAPH_PROMPTER_TOKEN_TYPE_VALUES_REVERSE
            token_type_combinations = get_combinations(token_type_values)
            if combination_boundaries:
                assert (
                    combination_boundaries[0] < len(token_type_combinations)
                ), f"Expected boundaries to be smaller then the amount of combinations, but got {combination_boundaries[0]} at position 0 for {len(token_type_combinations)}"
                assert (
                    combination_boundaries[1] < len(token_type_combinations)
                ), f"Expected boundaries to be smaller then the amount of combinations, but got {combination_boundaries[1]} at position 1 for {len(token_type_combinations)}"
                assert (
                    combination_boundaries[0] < combination_boundaries[1]
                ), f"Expected boundaries at position 0 to be smaller then boundary at position 1, but got {combination_boundaries}"
                token_type_combinations = token_type_combinations[
                    combination_boundaries[0] : combination_boundaries[1]
                ]
            with torch.no_grad():
                for split in splits:
                    print("split", split)
                    data_collator = self._get_data_collator(split)
                    data_loader = DataLoader(
                        dataset=dataset[split],  # type: ignore
                        batch_size=batch_size,
                        collate_fn=data_collator,
                    )
                    for combination in token_type_combinations:
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
                        print(f"combination: {combination}")
                        total_batches = len(data_loader)
                        for idx, batch in enumerate(data_loader):
                            # idx = 0
                            # if True:
                            # batch = next(iter(data_loader))
                            splits_ = [split] * len(batch["input_ids"])
                            batch["labels"] = batch["labels"].to(self.device)
                            batch["token_type_ids"] = batch["token_type_ids"].to(
                                self.device
                            )
                            token_type_mask = torch.zeros_like(
                                batch["token_type_ids"],
                                dtype=torch.int,
                                device=self.device,
                            )
                            for key in combination:
                                for pos in token_type_values_reverse[key]:
                                    token_type_mask += (
                                        batch["token_type_ids"] == pos
                                    ).int()
                            original_attention_mask = batch["attention_mask"].to(
                                self.device
                            )
                            batch["attention_mask"] = replace_ranges(
                                original_attention_mask,
                                token_type_mask,
                                value=0,
                            )
                            batch["input_ids"] = batch["input_ids"].to(self.device)
                            outputs = self.model(
                                **batch,
                                output_hidden_states=add_hidden_states,
                                output_attentions=add_attentions,
                            )
                            if add_attentions:
                                attentions = outputs.attentions
                                attentions = [
                                    torch.sum(layer, dim=1) for layer in attentions
                                ]
                                attentions = torch.stack(attentions).permute(1, 2, 3, 0)
                                attentions = mean_over_attention_ranges(
                                    attentions,
                                    batch["token_type_ids"],
                                    attention_mask=original_attention_mask,
                                )
                                attentions_collected.append(
                                    attentions.to("cpu").numpy()
                                )
                                if (idx + 1) % save_step_size == 0 or (
                                    idx + 1
                                ) == total_batches:
                                    np.save(
                                        self.sub_attentions_path.format(
                                            split,
                                            (idx + 1) / save_step_size,
                                            combination_string,
                                        ),
                                        np.concatenate(attentions_collected),
                                    )
                                    attentions_collected = []
                            if add_logits:
                                logits = outputs.logits
                                logits_of_combination.append(logits.to("cpu").numpy())
                                del logits
                            if add_hidden_states:
                                hidden_states = torch.stack(
                                    outputs.hidden_states, dim=1
                                )
                                hidden_states = mean_over_hidden_states(
                                    hidden_states,
                                    batch["token_type_ids"],
                                    original_attention_mask,
                                )
                                hidden_states_collected.append(
                                    hidden_states.to("cpu").numpy()
                                )
                                if (idx + 1) % save_step_size == 0 or (
                                    idx + 1
                                ) == total_batches:
                                    np.save(
                                        self.sub_hidden_states_path.format(
                                            split,
                                            (idx + 1) / save_step_size,
                                            combination_string,
                                        ),
                                        np.concatenate(hidden_states_collected),
                                    )
                                    hidden_states_collected = []
                        if add_logits:
                            logits_of_combination = np.concatenate(
                                logits_of_combination
                            )
                            np.save(
                                self.sub_logits_path.format(split, combination_string),
                                logits_of_combination,
                            )

    @staticmethod
    def read_forward_dataset(
        root: str, splits: List[str] = ["train", "test", "val"]
    ):  # TODO: move to EvaluationModule
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
