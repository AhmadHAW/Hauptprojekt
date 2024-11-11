import random as rd
from typing import Dict, Tuple, List, Set
from abc import ABC, abstractmethod

import torch
import pandas as pd
from transformers import DataCollatorForLanguageModeling


class DataCollatorBase(DataCollatorForLanguageModeling, ABC):
    """
    The Data Collators are used to generate non-existing edges on the fly. The false ratio allows to decide the ratio,
    in existing edges are replaced with non-existing edges.
    """

    def __init__(
        self, tokenizer, df, source_df, target_df, device="cpu", false_ratio=0.5
    ):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.false_ratio = false_ratio
        self.tokenizer = tokenizer
        self.source_ids, self.target_ids, self.edges = self.__df_to_nodes(df)
        self.source_df = source_df
        self.target_df = target_df
        self.device = device

    def __df_to_nodes(
        self, df: pd.DataFrame
    ) -> Tuple[List[int], List[int], Set[Tuple[int, int]]]:
        source_ids: List[int] = list(df["source_id"].unique())
        target_ids: List[int] = list(df["target_id"].unique())
        edges: Set[Tuple[int, int]] = set(
            df[["source_id", "target_id"]].itertuples(index=False, name=None)
        )
        return source_ids, target_ids, edges

    def __call__(self, features):
        if self.false_ratio != -1:
            total_features = len(features)
            new_feature_amount = int(self.false_ratio * total_features)
            with torch.no_grad():
                new_features = self._generate_false_examples(k=new_feature_amount)
            chosen_indices = rd.choices(range(total_features), k=new_feature_amount)
            for chosen_index, feature in zip(chosen_indices, new_features):
                features[chosen_index] = feature
        # Convert features into batches
        return self._convert_features_into_batches(features)

    @abstractmethod
    def _generate_false_examples(self, k: int) -> List[Dict]:
        pass

    @abstractmethod
    def _convert_features_into_batches(self, features: List[Dict]) -> Dict:
        pass
