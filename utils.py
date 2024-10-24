from typing import Tuple, List, Set

import numpy as np
import pandas as pd
import torch
import random as rd
import itertools


def find_non_existing_source_targets(
    edges: Set[Tuple[int, int]],
    source_ids: List[int],
    target_ids: List[int],
    k: int = 1,
) -> torch.Tensor:
    random_sources = rd.choices(source_ids, k=k)
    random_targets = rd.choices(target_ids, k=k)
    random_edges = [
        (random_source, random_target)
        for random_source, random_target in zip(random_sources, random_targets)
    ]
    duplicates = set()
    for idx, random_edge in enumerate(random_edges):
        if random_edge in edges:
            duplicates.add(idx)
    while len(duplicates) > 0:
        random_sources = rd.choices(source_ids, k=len(duplicates))
        random_targets = rd.choices(target_ids, k=len(duplicates))
        new_random_edges = [
            (random_source, random_target)
            for random_source, random_target in zip(random_sources, random_targets)
        ]
        new_duplicates = set()
        for idx, random_edge in zip(duplicates, new_random_edges):
            if random_edge in edges:
                new_duplicates.add(idx)
            else:
                random_edges[idx] = random_edge
        duplicates = new_duplicates
    return torch.tensor(random_edges)


def mean_over_attention_ranges(
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
    tens = tens.to(torch.float64)
    batch_size = tens.shape[0]
    positional_encodings = starts.shape[1]
    sequence_length = tens.shape[1]
    layers = tens.shape[-1]
    # input: # ends: torch.tensor([2, 5, 6]) starts: tensor([0, 2, 4])
    # Compute the maximum length of the ranges
    max_length = (ends - starts).max().item()
    # Create a range tensor from 0 to max_length-1
    range_tensor = torch.arange(max_length).repeat(
        (batch_size, positional_encodings, 1)
    )
    # Compute the ranges using broadcasting and masking
    ranges = starts.unsqueeze(-1).repeat(1, 1, int(max_length)) + range_tensor
    mask = ranges < ends.unsqueeze(-1).repeat(1, 1, int(max_length))

    # Apply the mask
    masked_ranges = (
        ranges * mask
    )  # result: tensor([[0, 1, 0], [2, 3, 4], [4, 5, 0]]) here padding index is 0
    #                        -                     -    positions were padded
    tens_unsqueezed = tens.unsqueeze(dim=2).repeat(1, 1, positional_encodings, 1, 1)
    result = (
        masked_ranges.unsqueeze(dim=1)
        .unsqueeze(-1)
        .repeat(1, sequence_length, 1, 1, layers)
    )
    gather = tens_unsqueezed.gather(dim=3, index=result)
    means = torch.mean(
        gather, dim=3
    )  # The mean was computed with the padding positions. We will remove the values from the mean now,

    range_diffs = (
        (max_length - (ends - starts))
        .unsqueeze(1)
        .unsqueeze(-1)
        .repeat(1, sequence_length, 1, layers)
    )  # the amount of times, the range had to be padded
    values_to_remove = (
        range_diffs * tens_unsqueezed[:, :, :, 0]
    )  # the summed value at padded position
    means = (means * max_length - values_to_remove) / (
        max_length - range_diffs
    )  # the actual mean without padding positions
    result_2 = (
        (ranges * mask)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(1, 1, 1, positional_encodings, layers)
    )
    means = means.unsqueeze(1).repeat(1, positional_encodings, 1, 1, 1)
    means_s2 = means.gather(dim=2, index=result_2).mean(dim=2)
    range_diffs = (
        (max_length - (ends - starts))
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(1, 1, positional_encodings, layers)
    )
    values_to_remove = range_diffs * means[:, :, 0]
    means_s2 = (means_s2 * max_length - values_to_remove) / (max_length - range_diffs)
    means_s2 = means_s2.to(torch.float32)
    return means_s2


def mean_over_attention_ranges_python_slow(
    tens: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """This python method is the corresponding test method to the mean_over_attention_ranges method, written in simple python,
    so the steps are simple to understand. This implementation is inefficient."""
    tens = tens.to(torch.float64)
    batch_positions = []
    for batch in range(starts.shape[0]):
        positions = []
        for start, end in zip(starts[batch], ends[batch]):
            assert (
                start < end
            ), f"Expected the start position to be smaller then end position, but got start: {start}, end: {end} instead."
            positions.append(tens[batch, :, start:end].mean(dim=1))
        positions = torch.stack(positions, dim=1)
        batch_positions.append(positions)
    tens = torch.stack(batch_positions, dim=0)

    batch_positions = []
    for batch in range(starts.shape[0]):
        positions = []
        for start, end in zip(starts[batch], ends[batch]):
            positions.append(tens[batch, start:end].mean(dim=0))
        positions = torch.stack(positions, dim=0)
        batch_positions.append(positions)
    tens = torch.stack(batch_positions, dim=0)
    tens = tens.to(torch.float32)
    return tens


def mean_over_hidden_states(
    tens: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    tens = tens.to(torch.float64)
    batch_size = tens.shape[0]
    positional_encodings = starts.shape[1]
    layers = tens.shape[1]
    hidden_state_size = tens.shape[-1]
    # input: # ends: torch.tensor([2, 5, 6]) starts: tensor([0, 2, 4])
    # Compute the maximum length of the ranges
    max_length = (ends - starts).max().item()
    # Create a range tensor from 0 to max_length-1
    range_tensor = torch.arange(max_length).repeat(
        (batch_size, positional_encodings, 1)
    )
    # Compute the ranges using broadcasting and masking
    ranges = starts.unsqueeze(-1).repeat(1, 1, int(max_length)) + range_tensor
    mask = ranges < ends.unsqueeze(-1).repeat(1, 1, int(max_length))

    # Apply the mask
    masked_ranges = (
        ranges * mask
    )  # result: tensor([[0, 1, 0], [2, 3, 4], [4, 5, 0]]) here padding index is 0
    #                        -                     -    positions were padded
    tens_unsqueezed = tens.unsqueeze(dim=2).repeat(1, 1, positional_encodings, 1, 1)
    result = (
        masked_ranges.unsqueeze(dim=1)
        .unsqueeze(-1)
        .repeat(1, layers, 1, 1, hidden_state_size)
    )
    gather = tens_unsqueezed.gather(dim=3, index=result)
    means = torch.mean(
        gather, dim=3
    )  # The mean was computed with the padding positions. We will remove the values from the mean now,
    range_diffs = (
        (max_length - (ends - starts))
        .unsqueeze(1)
        .unsqueeze(-1)
        .repeat(1, layers, 1, hidden_state_size)
    )  # the amount of times, the range had to be padded
    values_to_remove = (
        range_diffs * tens_unsqueezed[:, :, :, 0]
    )  # the summed value at padded position
    means = (means * max_length - values_to_remove) / (
        max_length - range_diffs
    )  # the actual mean without padding positions
    means = means.to(torch.float32)
    return means


def mean_over_hidden_states_python_slow(
    tens: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """This python method is the corresponding test method to the mean_over_attention_ranges method, written in simple python,
    so the steps are simple to understand. This implementation is inefficient."""
    tens = tens.to(torch.float64)
    batch_positions = []
    for batch in range(starts.shape[0]):
        layer_positions = []
        for layer in range(tens.shape[1]):
            positions = []
            for start, end in zip(starts[batch], ends[batch]):
                positions.append(tens[batch, layer, start:end].mean(dim=0))
            positions = torch.stack(positions)
            layer_positions.append(positions)
        layer_positions = torch.stack(layer_positions, dim=1)
        batch_positions.append(layer_positions)
    tens = torch.stack(batch_positions, dim=0).permute(0, 2, 1, 3)
    tens = tens.to(torch.float32)
    return tens


def replace_ranges(
    tensor: torch.Tensor, token_type_ranges: torch.Tensor, value: int = 0
) -> torch.Tensor:
    assert (tensor[:, -1] == 0).all()
    zero_index = torch.tensor(tensor.shape[1] - 1)
    starts = token_type_ranges[:, 0]
    ends = token_type_ranges[:, 1]
    max_length = (ends - starts).max().item()
    range_tensor = torch.arange(max_length).unsqueeze(0)
    ranges = starts.unsqueeze(1) + range_tensor
    mask = ranges >= ends.unsqueeze(1)
    diffs = zero_index.repeat((1, ranges.shape[1])) - ranges
    diffs_masked = diffs * mask
    ranges_masked = ranges + diffs_masked
    tensor = tensor.detach().clone()
    replace_tensor = torch.tensor(value, dtype=tensor.dtype).repeat(tensor.shape)
    tensor = tensor.scatter(1, ranges_masked, replace_tensor)
    tensor[:, -1] = 0
    return tensor


def replace_ranges_slow(
    tensor: torch.Tensor, token_type_ranges: torch.Tensor, value: int = 0
) -> torch.Tensor:
    tensor = tensor.detach().clone()
    for batch in range(len(tensor)):
        tensor[batch, token_type_ranges[batch, 0] : token_type_ranges[batch, 1]] = value
    return tensor


# Get all permutations (order does not matter)
def get_combinations(elements: List[str]) -> List[Set[str]]:
    result = []
    for r in range(1, len(elements) + 1):  # r is the length of permutations
        result.extend(itertools.combinations(elements, r))
    result_sets = []
    for permutation in result:
        result_sets.append(frozenset(permutation))
    result_sets.append(frozenset())  # the permutations of no keys.
    return result_sets


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def row_to_graph_prompter_hf_datapoint(
    row: pd.Series,
    sep_token: str = "[SEP]",
    pad_token: str = "[PAD]",
) -> str:
    prompt = row_to_vanilla_datapoint(row, sep_token)
    prompt = f"{prompt}{sep_token}{pad_token}{sep_token}{pad_token}"
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
