from typing import Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch

PROMPT_COLUMNS = ["source_id", "title", "genres"]


def find_non_existing_source_target(
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


def mean_over_ranges(
    tens: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    """
    This operation allows to produce a mean over ranges of different sizes in torch tensor manner only.
    We use padding positions to calculate an average and then remove the padded positions afterwards.
    This code was based on ChatGPT suggestions and works on the assumption:
    Let S be the sum of the padded list of numbers.
    Let n be the number of elements in the padded list.
    Let μ be the average of the padded list of numbers.
    Let r be the difference between the actual range lengths and the max range length
    Let x be the number that is at the padding position.
    μ' = ((μ * n)-(r * x)) / (n - r)
    """
    # input: # ends: torch.tensor([2, 5, 6]) starts: tensor([0, 2, 4])
    # Compute the maximum length of the ranges
    max_length = (ends - starts).max().item()
    range_diffs = (max_length - (ends - starts)).unsqueeze(
        1
    )  # the amount of times, the range had to be padded
    # Create a range tensor from 0 to max_length-1
    range_tensor = torch.arange(max_length).unsqueeze(0)

    # Compute the ranges using broadcasting and masking
    ranges = starts.unsqueeze(1) + range_tensor
    mask = ranges < ends.unsqueeze(1)

    # Apply the mask
    result = (
        ranges * mask
    )  # result: tensor([[0, 1, 0], [2, 3, 4], [4, 5, 0]]) here padding index is 0
    #                        -                     -    positions were padded
    result = result.unsqueeze(dim=2).repeat(1, 1, tens.shape[2])
    gather = tens.gather(dim=1, index=result)
    means = torch.mean(
        gather, dim=1
    )  # The mean was computed with the padding positions. We will remove the values from the mean now,
    values_to_remove = range_diffs * tens[:, 0]  # the summed value at padded position
    actual_means = (means * max_length - values_to_remove) / (
        max_length - range_diffs
    )  # the actual mean without padding positions
    return actual_means


def row_to_input_embeds_replace_datapoint(
    row: pd.Series,
    sep_token: str = "[SEP]",
    pad_token: str = "[PAD]",
) -> str:
    prompt = row_to_vanilla_datapoint(row, sep_token)
    prompt = f"{prompt}{sep_token}{pad_token}{sep_token}{pad_token}"
    return prompt


def row_to_prompt_datapoint(row: pd.Series, sep_token: str = "[SEP]") -> str:
    prompt = row_to_vanilla_datapoint(row, sep_token)
    prompt_source_embedding = row["prompt_source_embedding"]
    prompt_target_embedding = row["prompt_target_embedding"]
    prompt = f"{prompt}{sep_token}{transform_embeddings_to_string(prompt_source_embedding)}{sep_token}{transform_embeddings_to_string(prompt_target_embedding)}"
    return prompt


def row_to_vanilla_datapoint(row: pd.Series, sep_token: str = "[SEP]") -> str:
    prompt = ""
    prompt_columns = list(
        filter(
            lambda column: column.startswith("prompt_feature_")
            or column in ["source_id", "target_id"],
            row.keys(),
        )
    )
    for prompt_column in prompt_columns:
        prompt += f"{row[prompt_column]}{sep_token}"
    return prompt[: -len(sep_token)]


def transform_embeddings_to_string(embeddings: torch.Tensor, float_numbers: int = 16):
    return str([format(embedding, f".{float_numbers}f") for embedding in embeddings])


def replace_ranges(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    semantic_positional_encodings: torch.Tensor,
    replace_value: int,
) -> torch.Tensor:
    assert attention_mask.count_nonzero(dim=1).max() < attention_mask.shape[1]
    for pos in range(semantic_positional_encodings.shape[1]):
        semantic_positional_encoding = semantic_positional_encodings[:, pos]
        zero_index = attention_mask.sum(dim=1).unsqueeze(1)
        starts = semantic_positional_encoding[:, 0]
        ends = semantic_positional_encoding[:, 1]
        max_length = (ends - starts).max().item()
        range_tensor = torch.arange(max_length).unsqueeze(0)
        ranges = starts.unsqueeze(1) + range_tensor
        mask = ranges >= ends.unsqueeze(1)
        diffs = zero_index.repeat((1, ranges.shape[1])) - ranges
        diffs_masked = diffs * mask
        ranges_masked = ranges + diffs_masked
        input_ids = input_ids.detach().clone()
        replace_tensor = torch.tensor(replace_value).repeat(input_ids.shape)
        input_ids = input_ids.scatter(1, ranges_masked, replace_tensor)
    return input_ids
