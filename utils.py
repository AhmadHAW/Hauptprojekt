from typing import Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch

PROMPT_COLUMNS = ["source_id", "title", "genres"]


def transform_embeddings_to_string(embeddings: torch.Tensor, float_numbers: int = 16):
    return str([format(embedding, f".{float_numbers}f") for embedding in embeddings])


def _find_non_existing_source_target(
    df: pd.DataFrame, already_added_pairs: Optional[Set[Tuple[int, int]]] = None
) -> Tuple[int, int]:
    while True:
        source_id = np.random.choice(df["source_id"].unique())
        target_id = np.random.choice(df["target_id"].unique())
        if not ((df["source_id"] == source_id) & (df["target_id"] == target_id)).any():
            if (
                already_added_pairs is None
                or (source_id, target_id) not in already_added_pairs
            ):
                if already_added_pairs is not None:
                    already_added_pairs.add((source_id, target_id))
                return source_id, target_id


def row_to_attention_datapoint(
    row: pd.Series,
    sep_token: str = "[SEP]",
    pad_token: str = "[PAD]",
) -> str:
    prompt = row_to_vanilla_datapoint(row, sep_token)
    prompt = f"{prompt}s{sep_token}{pad_token}{sep_token}{pad_token}"
    return prompt


def row_to_prompt_datapoint(row: pd.Series, sep_token: str = "[SEP]") -> str:
    prompt = row_to_vanilla_datapoint(row, sep_token)
    prompt_source_embedding = row["prompt_source_embedding"]
    prompt_target_embedding = row["prompt_target_embedding"]
    prompt = f"{prompt}{sep_token}{transform_embeddings_to_string(prompt_source_embedding)}{sep_token}{transform_embeddings_to_string(prompt_target_embedding)}"
    return prompt


def row_to_vanilla_datapoint(row: pd.Series, sep_token: str) -> str:
    prompt = ""
    for prompt_column in PROMPT_COLUMNS:
        prompt += f"{row[prompt_column]}{sep_token}"
    return prompt[: -len(sep_token)]
