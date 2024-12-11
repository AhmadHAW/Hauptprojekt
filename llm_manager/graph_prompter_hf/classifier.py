from typing import Optional, Union, Tuple, List, Callable
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, Parameter
from torch_geometric.data import HeteroData
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BertConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
)
from utils import (
    replace_ranges,
    get_token_type_ranges,
    sort_ranges,
    token_ranges_to_mask,
)

from llm_manager.classifier_base import ClassifierBase
from llm_manager.graph_prompter_hf.data_collator import GraphPrompterHFDataCollator
from llm_manager.graph_prompter_hf.config import (
    GRAPH_PROMPTER_TOKEN_TYPES,
    GRAPH_PROMPTER_TOKEN_TYPE_VALUES,
)
from dataset_manager.kg_manager import ROOT, KGManger


class GraphPrompterHFBertEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    """
    Addition by this works context author: We add the TTRs and KGEs, so that we can replace
    the padding placeholders with the KGEs, before adding the positional or type encodings.
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        source_kges: Optional[torch.FloatTensor] = None,
        target_kges: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        mask_source_kge: bool = False,
        mask_target_kge: bool = False,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            assert (
                inputs_embeds is not None
            ), (
                "Either input ids or input embeds need to be set"
            )  # added by the author for better typing
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
                new_token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )
                assert isinstance(
                    new_token_type_ids, torch.LongTensor
                )  # added by author for better typing
                token_type_ids = new_token_type_ids
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            if (
                token_type_ids is not None
                and source_kges is not None
                and target_kges is not None
                and inputs_embeds is not None
                and not (mask_source_kge and mask_target_kge)
            ):
                if not mask_source_kge:
                    source_mask = (
                        token_type_ids == GRAPH_PROMPTER_TOKEN_TYPE_VALUES[-4]
                    ).int()
                    source_mask = source_mask.unsqueeze(-1).repeat(
                        1, 1, inputs_embeds.shape[2]
                    )
                else:
                    source_mask = torch.zeros_like(inputs_embeds)
                if not mask_target_kge:
                    target_mask = (
                        token_type_ids == GRAPH_PROMPTER_TOKEN_TYPE_VALUES[-2]
                    ).int()
                    target_mask = target_mask.unsqueeze(-1).repeat(
                        1, 1, inputs_embeds.shape[2]
                    )
                else:
                    target_mask = torch.zeros_like(inputs_embeds)
                new_inputs_embeds = (
                    inputs_embeds * (1 - source_mask)
                    + source_kges.unsqueeze(1).repeat(1, inputs_embeds.shape[1], 1)
                    * source_mask
                ) * (1 - target_mask) + target_kges.unsqueeze(1).repeat(
                    1, inputs_embeds.shape[1], 1
                ) * target_mask
                inputs_embeds = new_inputs_embeds

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GraphPrompterHFBertModel(BertModel):
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
    Addition by this works context author: We add graph embeddings, so that we can replace
    the padding placeholders with the KGEs, before adding the positional and type encodings.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = GraphPrompterHFBertEmbeddings(
            config
        )  # changed by this works author.
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()


class GraphPrompterHFBertForSequenceClassification(BertForSequenceClassification):
    """The bert base model has been adjusted so it also takes TTR and KGEs."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = GraphPrompterHFBertModel(config)  # changed by author
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    # we change the forward methods expected parameters and add kges, ttrs, source and target id.
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
        source_kges: Optional[torch.Tensor] = None,
        target_kges: Optional[torch.Tensor] = None,
        source_id: Optional[List[int]] = None,
        target_id: Optional[List[int]] = None,
        split: Optional[str] = None,
        mask_source_kge: bool = False,
        mask_target_kge: bool = False,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if not split:
            split = "train"
        if inputs_embeds is None:
            if source_kges is None or target_kges is None:
                print("should not be here")
                if source_id is not None and target_id is not None:
                    if self.detach_kges:
                        with torch.no_grad():
                            print("no grad")
                            source_kges, target_kges = self.get_embeddings_cb(
                                self.data[split], source_id, target_id
                            )
                            assert source_kges is not None
                            assert target_kges is not None
                            source_kges = source_kges.detach()
                            target_kges = target_kges.detach()
                        source_kges = source_kges.requires_grad_(True)
                        target_kges = target_kges.requires_grad_(True)
                else:
                    if not (mask_source_kge and mask_target_kge):
                        source_kges, target_kges = self.get_embeddings_cb(
                            self.data[split], source_id, target_id
                        )
                        if mask_source_kge:
                            source_kges = source_kges.detach()
                        if mask_target_kge:
                            target_kges = target_kges.detach()
                    else:
                        source_kges = torch.zeros(
                            (input_ids.shape[0], self.config.hidden_size),
                            device=self.device,
                        )
                        target_kges = torch.zeros(
                            (input_ids.shape[0], self.config.hidden_size),
                            device=self.device,
                        )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            labels = labels.to(self.device)
            inputs_embeds = self.bert.embeddings(
                input_ids,
                source_kges=source_kges,
                target_kges=target_kges,
                token_type_ids=token_type_ids,
                mask_source_kge=mask_source_kge,
                mask_target_kge=mask_target_kge,
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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GraphPrompterHF(ClassifierBase):
    def __init__(
        self,
        kge_manager: KGManger,
        get_embeddings_cb: Callable,
        model_name: str,
        root_path: Optional[str | Path] = f"{ROOT}/llm/graph_prompter_hf",
        model_max_length: int = 256,
        vanilla_model_path: Optional[str | Path] = None,
        gnn_parameters: Optional[List[Parameter]] = None,
        false_ratio: float = 0.5,
        force_recompute: bool = False,
    ) -> None:
        training_path = f"{root_path}/training"
        model_path = f"{training_path}/best"

        config = BertConfig.from_pretrained(model_name)
        config.type_vocab_size = GRAPH_PROMPTER_TOKEN_TYPES

        config.num_labels = 2
        config.id2label = {0: "FALSE", 1: "TRUE"}
        config.label2id = {"FALSE": 0, "TRUE": 1}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device", self.device)
        # Initialize the model and tokenizer
        if (
            vanilla_model_path
            and os.path.exists(vanilla_model_path)
            and not force_recompute
        ):
            model = GraphPrompterHFBertForSequenceClassification.from_pretrained(
                vanilla_model_path,
                config=config,
                ignore_mismatched_sizes=True,
            )
        elif os.path.exists(model_path) and not force_recompute:
            model = GraphPrompterHFBertForSequenceClassification.from_pretrained(
                model_path, config=config, ignore_mismatched_sizes=True
            )
        else:
            model = GraphPrompterHFBertForSequenceClassification.from_pretrained(
                model_name, config=config, ignore_mismatched_sizes=True
            )

        assert isinstance(model, GraphPrompterHFBertForSequenceClassification)
        assert isinstance(model, nn.Module)
        model.to(self.device)
        model.get_embeddings_cb = get_embeddings_cb
        model.detach_kges = gnn_parameters is None
        model.data = {
            "train": kge_manager.gnn_train_data,
            "test": kge_manager.gnn_test_data,
            "val": kge_manager.gnn_val_data,
        }

        tokenizer = BertTokenizer.from_pretrained(
            model_name, model_max_length=model_max_length
        )
        train_data_collator = GraphPrompterHFDataCollator(
            tokenizer,
            self.device,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            split="train",
            false_ratio=false_ratio,
        )
        test_data_collator = GraphPrompterHFDataCollator(
            tokenizer,
            self.device,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            split="test",
            false_ratio=-1.0,
        )
        val_data_collator = GraphPrompterHFDataCollator(
            tokenizer,
            self.device,
            kge_manager.llm_df,
            kge_manager.source_df,
            kge_manager.target_df,
            split="val",
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
            gnn_parameters=gnn_parameters,
            root_path=root_path,
            device=self.device,
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
            GRAPH_PROMPTER_TOKEN_TYPE_VALUES, range(token_type_ranges.shape[1])
        ):
            token_type_mask = token_ranges_to_mask(
                token_type_ids.shape[1], token_type_ranges[:, range_position]
            )
            token_type_ids = replace_ranges(
                token_type_ids, token_type_mask, value=token_type
            )

        if return_pt:
            result = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": example["labels"],
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
                "token_type_ids": token_type_ids.detach().to("cpu").tolist(),
                "source_id": example["source_id"],
                "target_id": example["target_id"],
            }
        return result

    def plot_training_loss_and_accuracy(self):
        model_type = "Graph Prompter HF"
        self._plot_training_loss_and_accuracy(model_type)
