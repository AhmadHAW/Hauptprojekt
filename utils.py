from typing import Tuple, List, Set, FrozenSet

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
    tens: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    This operation allows to produce a mean over grouped token type ids
    """
    tens_means = []
    for token_type in torch.unique(token_type_ids):
        token_type_mask = (token_type_ids == token_type).int() * attention_mask
        token_type_mask = (
            token_type_mask.unsqueeze(1)
            .unsqueeze(-1)
            .repeat(1, tens.shape[1], 1, tens.shape[-1])
        )
        sum_ = torch.sum(token_type_mask, dim=2)
        result = torch.sum(tens * token_type_mask, dim=2) / sum_
        tens_means.append(result)
    tens = torch.stack(tens_means, dim=2)

    tens_means = []
    for token_type in torch.unique(token_type_ids):
        token_type_mask = (token_type_ids == token_type).int() * attention_mask
        token_type_mask = (
            token_type_mask.unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, 1, tens.shape[2], tens.shape[3])
        )
        sum_ = torch.sum(token_type_mask, dim=1)
        result = torch.sum(tens * token_type_mask, dim=1) / sum_
        tens_means.append(result)
    tens = torch.stack(tens_means, dim=1)
    return tens


def test_mean_over_attention_ranges():
    # Case 1: Simple example
    tens = torch.tensor(
        [
            [
                [[1, 2], [3, 4], [5, 6]],
                [[7, 8], [9, 10], [11, 12]],
                [[13, 14], [15, 16], [17, 18]],
            ]
        ],
        dtype=torch.float32,
    )
    token_type_ids = torch.tensor([[0, 0, 1]])
    # expected = torch.tensor(
    #     [[[[2.0, 3.0], [5.0, 6.0]],
    #       [[8.0, 9.0], [11.0, 12.0]],
    #       [[14.0, 15.0], [17.0, 18.0]]]], dtype=torch.float32
    # )
    expected = torch.tensor(
        [[[[[5.0, 6.0], [8.0, 9.0]], [[14.0, 15.0], [17.0, 18.0]]]]],
        dtype=torch.float32,
    )
    result = mean_over_attention_ranges(tens, token_type_ids)
    print(result.shape)
    assert torch.allclose(
        result, expected
    ), f"Test failed! Got {result}, expected {expected}"

    # Add other cases...

    print("All tests passed!")


def mean_over_hidden_states(
    tens: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    This operation allows to produce a mean over grouped token type ids
    """
    tens_means = []
    for token_type in torch.unique(token_type_ids):
        token_type_mask = (token_type_ids == token_type).int() * attention_mask
        token_type_mask = (
            token_type_mask.unsqueeze(1)
            .unsqueeze(-1)
            .repeat(1, tens.shape[1], 1, tens.shape[-1])
        )
        sum_ = torch.sum(token_type_mask, dim=2)
        result = torch.sum(tens * token_type_mask, dim=2) / sum_
        tens_means.append(result)
    tens = torch.stack(tens_means, dim=1)
    return tens


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


def token_ranges_to_mask(
    max_size: int, token_type_ranges: torch.Tensor
) -> torch.Tensor:
    ranges = torch.arange(max_size).unsqueeze(0).repeat(len(token_type_ranges), 1)
    lower_boundaries = token_type_ranges[:, 0].unsqueeze(-1).repeat(1, max_size)
    upper_boundaries = token_type_ranges[:, 1].unsqueeze(-1).repeat(1, max_size)
    return ((ranges >= lower_boundaries) & (ranges < upper_boundaries)).int()


def replace_ranges(
    tensor: torch.Tensor, token_type_mask: torch.Tensor, value: int = 0
) -> torch.Tensor:
    value_tensor = torch.tensor([[value]], device=tensor.device, dtype=torch.long)
    value_tensor = value_tensor.repeat(tensor.shape)
    result = tensor * (1 - token_type_mask) + value_tensor * token_type_mask
    return result


def replace_ranges_slow(
    tensor: torch.Tensor, token_type_ranges: torch.Tensor, value: int = 0
) -> torch.Tensor:
    tensor = tensor.detach().clone()
    for batch in range(len(tensor)):
        tensor[batch, token_type_ranges[batch, 0] : token_type_ranges[batch, 1]] = value
    return tensor


def get_token_type_ranges(input_ids: torch.Tensor, sep_token_id: int) -> torch.Tensor:
    mask = input_ids == sep_token_id
    positions = mask.nonzero(as_tuple=True)
    cols = positions[1]

    # Step 3: Determine the number of True values per row
    num_trues_per_row = mask.sum(dim=1)
    max_trues_per_row = num_trues_per_row.max().item()
    # Step 4: Create an empty tensor to hold the result
    token_type_ranges = -torch.ones(
        (mask.size(0), max_trues_per_row),  # type: ignore
        dtype=torch.long,
    )

    # Step 5: Use scatter to place column indices in the token_type_ranges tensor
    # Create an index tensor that assigns each column index to the correct position in token_type_ranges tensor
    row_indices = torch.arange(mask.size(0)).repeat_interleave(num_trues_per_row)
    column_indices = torch.cat([torch.arange(n) for n in num_trues_per_row])  # type: ignore

    token_type_ranges[row_indices, column_indices] = cols
    token_type_ranges = torch.stack(
        [
            token_type_ranges[:, :-1] + 1,
            token_type_ranges[:, 1:],
        ],
        dim=2,
    )
    # Create a tensor of zeros to represent the starting points
    second_points = torch.ones(
        token_type_ranges.size(0),
        1,
        2,
        dtype=token_type_ranges.dtype,
    )
    # Set the second column to be the first element of the first range
    second_points[:, 0, 1] = token_type_ranges[:, 0, 0] - 1
    # Concatenate the start_points tensor with the original token_type_ranges tensor
    token_type_ranges = torch.cat((second_points, token_type_ranges), dim=1)
    return token_type_ranges


def sort_ranges(token_type_ranges: torch.Tensor):
    # Extract the second element (end of the current ranges excluded the starting cps token)
    end_elements = token_type_ranges[:, :, 1]
    # Create the new ranges by adding 1 to the end elements
    new_ranges = torch.stack([end_elements, end_elements + 1], dim=-1)
    # add the cls positions to it
    cls_positions = torch.tensor([0, 1])
    cls_positions = cls_positions.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 2)
    cls_positions = cls_positions.expand(
        new_ranges.size(0), 1, -1
    )  # Shape (batch_size, 1, 2)
    new_ranges = torch.cat((new_ranges, cls_positions), dim=1)
    # Concatenate the original ranges with the new ranges
    token_type_ranges = torch.cat((token_type_ranges, new_ranges), dim=1)
    # Step 1: Extract the last value of dimension 2
    last_values = token_type_ranges[:, :, -1]  # Shape (batch_size, num_elements)

    # Step 2: Sort the indices based on these last values
    # 'values' gives the sorted values (optional), 'indices' gives the indices to sort along dim 1
    _, indices = torch.sort(last_values, dim=1, descending=False)

    # Step 3: Apply the sorting indices to the original tensor
    token_type_ranges = torch.gather(
        token_type_ranges,
        1,
        indices.unsqueeze(-1).expand(-1, -1, token_type_ranges.size(2)),
    )
    return token_type_ranges


# Get all permutations (order does not matter)
def get_combinations(elements: List[int]) -> List[FrozenSet[int]]:
    result = []
    for r in range(1, len(elements) + 1):  # r is the length of permutations
        result.extend(itertools.combinations(elements, r))
    empty_combination: FrozenSet[int] = frozenset()
    result_sets = [empty_combination]  # the permutations of no keys.
    for permutation in result:
        result_sets.append(frozenset(permutation))
    return result_sets


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def transform_embeddings_to_string(embeddings: torch.Tensor, float_numbers: int = 16):
    return str([format(embedding, f".{float_numbers}f") for embedding in embeddings])


def find_all_index(value, elements: List) -> List[int]:
    return [i for i, x in enumerate(elements) if x == value]
