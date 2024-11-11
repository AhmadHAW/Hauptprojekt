from utils import find_all_index

VANILLA_TOKEN_TYPE_VALUES = [0, 1, 0, 2, 0, 2, 0, 2, 0]
VANILLA_TOKEN_TYPES = len(set(VANILLA_TOKEN_TYPE_VALUES))
VANILLA_TOKEN_TYPE_VALUES_REVERSE = {
    vanilla_token_type: find_all_index(vanilla_token_type, VANILLA_TOKEN_TYPE_VALUES)
    for vanilla_token_type in set(VANILLA_TOKEN_TYPE_VALUES)
}
