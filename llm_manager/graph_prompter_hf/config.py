from llm_manager.vanilla.config import VANILLA_TOKEN_TYPE_VALUES
from utils import find_all_index

GRAPH_PROMPTER_TOKEN_TYPE_VALUES = VANILLA_TOKEN_TYPE_VALUES + [3, 0, 4, 0]
GRAPH_PROMPTER_TOKEN_TYPES = len(set(GRAPH_PROMPTER_TOKEN_TYPE_VALUES))
GRAPH_PROMPTER_TOKEN_TYPE_VALUES_REVERSE = {
    graph_prompter_token_type: find_all_index(
        graph_prompter_token_type, GRAPH_PROMPTER_TOKEN_TYPE_VALUES
    )
    for graph_prompter_token_type in set(GRAPH_PROMPTER_TOKEN_TYPE_VALUES)
}