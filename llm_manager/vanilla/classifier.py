from typing import Optional, Union, Tuple
from pathlib import Path
import os

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import pandas as pd
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BertConfig,
)

from llm_manager.classifier_base import ClassifierBase
from utils import get_token_type_ranges, replace_ranges, sort_ranges
from llm_manager.vanilla.config import VANILLA_TOKEN_TYPE_VALUES, VANILLA_TOKEN_TYPES
from llm_manager.vanilla.data_collator import VanillaEmbeddingDataCollator
from llm_manager.modeling_outputs import SequenceClassifierOutputOverRanges


class VanillaBertForSequenceClassification(BertForSequenceClassification):
    """
    The adjustments to this class were done by adding the TTRs.
    """

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
        token_type_ranges: Optional[torch.Tensor] = None,
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
            token_type_ranges=token_type_ranges,
        )


class VanillaClassifier(ClassifierBase):
    def __init__(
        self,
        df: pd.DataFrame,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        root_path: str | Path,
        model_name: str,
        model_max_length: int = 256,
        false_ratio: float = 0.5,
        force_recompute: bool = False,
    ) -> None:
        tokenizer = BertTokenizer.from_pretrained(
            model_name, model_max_length=model_max_length
        )
        train_data_collator = VanillaEmbeddingDataCollator(
            tokenizer, df, source_df, target_df, false_ratio=false_ratio
        )
        val_data_collator = VanillaEmbeddingDataCollator(
            tokenizer, df, source_df, target_df, false_ratio=-1.0
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device", self.device)
        best_model_path = f"{root_path}/training/best"
        config = BertConfig.from_pretrained(model_name)
        config.type_vocab_size = VANILLA_TOKEN_TYPES
        config.num_labels = 2
        config.id2label = {0: "FALSE", 1: "TRUE"}
        config.label2id = {"FALSE": 0, "TRUE": 1}
        # Initialize the model and tokenizer
        if os.path.exists(best_model_path) and not force_recompute:
            model = VanillaBertForSequenceClassification.from_pretrained(
                best_model_path, config=config, ignore_mismatched_sizes=True
            )
        else:
            model = VanillaBertForSequenceClassification.from_pretrained(
                model_name, config=config, ignore_mismatched_sizes=True
            )
        assert isinstance(model, BertForSequenceClassification)
        super().__init__(
            df=df,
            model=model,
            tokenizer=tokenizer,
            train_data_collator=train_data_collator,
            test_data_collator=val_data_collator,
            val_data_collator=val_data_collator,
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
        token_type_ranges = get_token_type_ranges(
            tokenized["input_ids"], self.tokenizer.sep_token_id
        )
        token_type_ranges = sort_ranges(token_type_ranges)
        token_type_ids = torch.zeros_like(tokenized["attention_mask"])
        for token_type, range_position in zip(
            VANILLA_TOKEN_TYPE_VALUES, range(token_type_ranges.shape[1])
        ):
            token_type_ids = replace_ranges(
                token_type_ids, token_type_ranges[:, range_position], value=token_type
            )
        if return_pt:
            result = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": example["labels"],
                "token_type_ranges": token_type_ranges,
                "token_type_ids": token_type_ids,
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
                "token_type_ranges": token_type_ranges.detach().to("cpu").tolist(),
                "token_type_ids": token_type_ids.detach().to("cpu").tolist(),
                "source_id": example["source_id"],
                "target_id": example["target_id"],
            }
        return result

    def plot_training_loss_and_accuracy(self):
        model_type = "Vanilla"
        self._plot_training_loss_and_accuracy(model_type)
