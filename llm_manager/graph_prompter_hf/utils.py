import pandas as pd

from llm_manager.vanilla.utils import row_to_vanilla_datapoint


def row_to_graph_prompter_hf_datapoint(
    row: pd.Series,
    sep_token: str = "[SEP]",
    pad_token: str = "[PAD]",
) -> str:
    prompt = row_to_vanilla_datapoint(row, sep_token)
    prompt = f"{prompt}{sep_token}{pad_token}{sep_token}{pad_token}"
    return prompt
