import pandas as pd


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
