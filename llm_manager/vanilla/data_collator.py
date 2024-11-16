from typing import Dict, List
from abc import ABC

import torch
import pandas as pd

from utils import (
    get_token_type_ranges,
    replace_ranges,
    find_non_existing_source_targets,
    sort_ranges,
)
from llm_manager.vanilla.utils import row_to_vanilla_datapoint
from llm_manager.data_collator_base import DataCollatorBase
from llm_manager.vanilla.config import VANILLA_TOKEN_TYPE_VALUES


class VanillaDataCollator(DataCollatorBase):
    """
    The vanilla data collator in addition to the original DataCollator, this collator generates false examples
    depending on the false ratio set during initialization.
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
        token_type_ranges = get_token_type_ranges(
            input_ids,
            sep_token_id,
        )
        token_type_ranges = sort_ranges(token_type_ranges)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        assert isinstance(attention_mask, List)
        assert isinstance(input_ids, List)
        token_type_ids = torch.zeros(len(attention_mask), len(attention_mask[0]))
        for token_type, range_position in zip(
            VANILLA_TOKEN_TYPE_VALUES, range(token_type_ranges.shape[1])
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
            }
            for token_type_ranges_, token_type_ids_, input_ids_, attention_mask_ in zip(
                token_type_ranges, token_type_ids, input_ids, attention_mask
            )
        ]
        return result_dict

    def _convert_features_into_batches(self, features: List[Dict]) -> Dict:
        input_ids = []
        attention_mask = []
        labels = []
        token_type_ranges = []
        token_type_ids = []

        for f in features:
            input_ids.append(f["input_ids"])
            attention_mask.append(f["attention_mask"])
            labels.append(f["labels"])
            token_type_ranges.append(f["token_type_ranges"])
            token_type_ids.append(f["token_type_ids"])
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        token_type_ranges = torch.tensor(token_type_ranges, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "token_type_ranges": token_type_ranges,
            "token_type_ids": token_type_ids,
        }
