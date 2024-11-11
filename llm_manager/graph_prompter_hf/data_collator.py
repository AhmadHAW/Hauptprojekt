from typing import Dict, List

import torch
import pandas as pd

from utils import (
    get_token_type_ranges,
    replace_ranges,
    find_non_existing_source_targets,
    sort_ranges,
)
from llm_manager.graph_prompter_hf.utils import row_to_graph_prompter_hf_datapoint

from llm_manager.data_collator_base import DataCollatorBase
from llm_manager.graph_prompter_hf.config import GRAPH_PROMPTER_TOKEN_TYPE_VALUES


class GraphPrompterHFDataCollator(DataCollatorBase):
    def __init__(
        self,
        tokenizer,
        device,
        df,
        source_df,
        target_df,
        data,
        get_embeddings_cb,
        detach_kges,
        false_ratio=0.5,
    ):
        super().__init__(
            tokenizer, df, source_df, target_df, false_ratio=false_ratio, device=device
        )
        self.data = data
        self.get_embeddings_cb = get_embeddings_cb
        self.detach_kges = detach_kges

    def _convert_features_into_batches(self, features: List[Dict]) -> Dict:
        input_ids = []
        attention_mask = []
        labels = []
        token_type_ranges = []
        token_type_ids = []
        source_ids = []
        target_ids = []
        source_kges = []
        target_kges = []
        for f in features:
            input_ids.append(f["input_ids"])
            attention_mask.append(f["attention_mask"])
            labels.append(f["labels"])
            token_type_ranges.append(f["token_type_ranges"])
            token_type_ids.append(f["token_type_ids"])
            source_ids.append(f["source_id"])
            target_ids.append(f["target_id"])
            if "source_kges" in f:
                source_kges.append(f["source_kges"])
            if "target_kges" in f:
                target_kges.append(f["target_kges"])
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        token_type_ranges = torch.tensor(token_type_ranges, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        if len(source_kges) == 0:
            source_kges, target_kges = self.get_embeddings_cb(
                self.data, source_ids, target_ids
            )
            if self.detach_kges:
                source_kges = source_kges.detach()
                target_kges = target_kges.detach()
        else:
            source_kges = torch.stack(source_kges)
            target_kges = torch.stack(target_kges)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source_kges": source_kges,
            "target_kges": target_kges,
            "token_type_ranges": token_type_ranges,
            "token_type_ids": token_type_ids,
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
            lambda row: row_to_graph_prompter_hf_datapoint(
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
        token_type_ranges = get_token_type_ranges(
            input_ids,
            sep_token_id,
        )
        token_type_ranges = sort_ranges(token_type_ranges)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        assert isinstance(input_ids, List)
        assert isinstance(attention_mask, List)
        token_type_ids = torch.zeros(len(attention_mask), len(attention_mask[0]))
        for token_type, range_position in zip(
            GRAPH_PROMPTER_TOKEN_TYPE_VALUES, range(token_type_ranges.shape[1])
        ):
            token_type_ids = replace_ranges(
                token_type_ids, token_type_ranges[:, range_position], value=token_type
            )
        result_dict = [
            {
                "input_ids": input_ids_,
                "attention_mask": attention_mask_,
                "labels": 0,
                "token_type_ranges": token_type_ranges_.to("cpu").detach().tolist(),
                "token_type_ids": token_type_ids_.to("cpu").detach().tolist(),
                "source_id": node_pair[0],
                "target_id": node_pair[1],
            }
            for token_type_ranges_, token_type_ids_, input_ids_, attention_mask_, node_pair in zip(
                token_type_ranges,
                token_type_ids,
                input_ids,
                attention_mask,
                node_pairs,
            )
        ]
        return result_dict
