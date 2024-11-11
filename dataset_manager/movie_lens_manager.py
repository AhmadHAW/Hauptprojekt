from typing import List, Tuple, Optional
import ast
import re

import torch
from torch_geometric.data import download_url, extract_zip
import pandas as pd
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from dataset_manager.kg_manager import KGManger, ROOT


class MovieLensManager(KGManger):
    """
    The MovieLensManager manages the original graph data set and the pre-processed data sets for GNNs and LLMs.
    """

    def __init__(
        self,
        source_df: Optional[pd.DataFrame] = None,
        target_df: Optional[pd.DataFrame] = None,
        edge_df: Optional[pd.DataFrame] = None,
        force_recompute: bool = False,
    ) -> None:
        """
        The constructor allows general settings, like forcing to reload even tho there are datasets present.
        The preprocessing of the dataset can be read in detail in the original gnn link prediction tutorial of
        torch geometrics (https://colab.research.google.com/drive/1xpzn1Nvai1ygd_P5Yambc_oe4VBPK_ZT?usp=sharing#scrollTo=vit8xKCiXAue)

        Parameters
        __________
        force_recompute:            bool
                                    Whether to force reloading and recomputing datasets and values.
                                    Default False -> Loads and computes only if missing.


        """
        if not self._data_present() or force_recompute:
            if source_df:
                assert target_df
                assert edge_df
            else:
                movies_df, movies_llm_df, ratings_df = self.__download_dataset()
                source_df, target_df, edge_df = self.__map_to_unique(
                    movies_df, movies_llm_df, ratings_df
                )
            super().__init__(source_df, target_df, edge_df, force_recompute)
        else:
            self._load_datasets_from_disk()

    def __download_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Downloads the https://files.grouplens.org/datasets/movielens/ml-32m.zip dataset and
        extracts the content into ROOT//ml-32m/"""
        movies_path = f"{ROOT}/ml-32m/movies.csv"
        ratings_path = f"{ROOT}/ml-32m/ratings.csv"
        url = "https://files.grouplens.org/datasets/ml-32m.zip"
        extract_zip(download_url(url, ROOT), ROOT)
        movies_df = pd.read_csv(movies_path, index_col="movieId")
        movies_llm_df = pd.read_csv(movies_path, index_col="movieId")
        movies_llm_df["genres"] = movies_llm_df["genres"].apply(
            lambda genres: list(genres.split("|"))
        )
        ratings_df = pd.read_csv(ratings_path)
        return movies_df, movies_llm_df, ratings_df

    def __map_to_unique(
        self,
        movies_df: pd.DataFrame,
        movies_llm_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Maps the IDs from the data set to a compact ID series.
        Then merge the data points so that <userID, movieID, title, genres] quadruples are created.
        """
        # Create a mapping from unique user indices to range [0, num_user_nodes):
        unique_user_id = ratings_df["userId"].unique()
        unique_user_id = pd.DataFrame(
            data={
                "userId": unique_user_id,
                "mappedUserId": pd.RangeIndex(len(unique_user_id)),
            }
        )
        source_df = pd.DataFrame({"id": pd.RangeIndex(len(unique_user_id))})
        # Create a mapping from unique movie indices to range [0, num_movie_nodes):
        unique_movie_id = pd.DataFrame(
            data={
                "movieId": movies_df.index,
                "mappedMovieId": pd.RangeIndex(len(movies_df)),
            }
        )
        target_df = pd.DataFrame(
            data={
                "id": pd.RangeIndex(len(movies_llm_df)),
                "title": movies_llm_df["title"].tolist(),
                "genres": movies_llm_df["genres"].tolist(),
            }
        )
        genre_dummies = (
            pd.get_dummies(target_df["genres"].explode(), prefix="gnn_feature")
            .groupby(level=0)
            .sum()
        )
        target_df = pd.concat([target_df, genre_dummies], axis=1)

        # Perform merge to obtain the edges from users and movies:
        ratings_user_id = pd.merge(
            ratings_df["userId"],
            unique_user_id,
            left_on="userId",
            right_on="userId",
            how="left",
        )
        ratings_user_id = torch.from_numpy(ratings_user_id["mappedUserId"].values)
        ratings_movie_id = pd.merge(
            ratings_df["movieId"],
            unique_movie_id,
            left_on="movieId",
            right_on="movieId",
            how="left",
        )
        ratings_movie_id = torch.from_numpy(ratings_movie_id["mappedMovieId"].values)
        edge_df = pd.DataFrame(
            {"source_id": ratings_user_id, "target_id": ratings_movie_id}
        )
        target_df = target_df.rename(
            columns={"title": "prompt_feature_title", "genres": "prompt_feature_genres"}
        )
        return source_df, target_df, edge_df

    def get_vanilla_tokens_as_df(
        self,
        input_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        all_token_type_ranges: torch.Tensor,
    ) -> pd.DataFrame:
        all_token_type_ranges = all_token_type_ranges[:, [1, 3, 5, 7]]
        user_ids = []
        titles = []
        genres = []
        movie_ids = []
        all_semantic_tokens = [user_ids, movie_ids, titles, genres]
        self.fill_all_semantic_tokens(
            all_semantic_tokens, input_ids, tokenizer, all_token_type_ranges
        )
        all_semantic_tokens[0] = [int(id) for id in all_semantic_tokens[0]]
        all_semantic_tokens[1] = [int(id) for id in all_semantic_tokens[1]]
        all_semantic_tokens[3] = [
            ast.literal_eval(string_list) for string_list in all_semantic_tokens[3]
        ]
        data = {
            "source_id": all_semantic_tokens[0],
            "target_id": all_semantic_tokens[1],
            "title": all_semantic_tokens[2],
            "genres": all_semantic_tokens[3],
        }
        df = pd.DataFrame(data)
        return df

    def get_prompt_tokens_as_df(
        self,
        input_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        all_token_type_ranges: torch.Tensor,
    ) -> pd.DataFrame:
        all_token_type_ranges = all_token_type_ranges[:, [1, 3, 5, 7, 9, 11]]
        all_semantic_tokens = [[], [], [], [], [], []]
        self.fill_all_semantic_tokens(
            all_semantic_tokens, input_ids, tokenizer, all_token_type_ranges
        )
        all_semantic_tokens[0] = [int(id) for id in all_semantic_tokens[0]]
        all_semantic_tokens[1] = [int(id) for id in all_semantic_tokens[1]]
        all_semantic_tokens[3] = [
            ast.literal_eval(string_list) for string_list in all_semantic_tokens[3]
        ]
        data = {
            "source_id": all_semantic_tokens[0],
            "target_id": all_semantic_tokens[1],
            "title": all_semantic_tokens[2],
            "genres": all_semantic_tokens[3],
        }
        df = pd.DataFrame(data)
        return df

    @staticmethod
    def load_dataset_from_disk(path_to_hf_dataset: str) -> pd.DataFrame:
        dataset = load_from_disk(path_to_hf_dataset)
        dfs = []
        for split in ["train", "test", "val"]:
            dataset_split = dataset[split]
            assert isinstance(dataset_split, Dataset | DatasetDict)
            df = dataset_split.to_pandas()
            dfs.append(df)
        df = pd.concat(dfs)
        regex = r"(vanilla|graph_prompter_hf"
        model_types: List = []
        for column in df.columns:
            match = re.search(regex, column)
            if match:
                model_types.append(match.group(1))
        for attention_column in model_types:
            df[f"{attention_column}_attentions"] = df.apply(
                lambda row: row[f"{attention_column}_attentions"].reshape(
                    row[f"{attention_column}_attentions_original_shape"]
                ),
                axis=1,
            )

        for hidden_state_column in model_types:
            df[f"{hidden_state_column}_hidden_states"] = df.apply(
                lambda row: row[f"{hidden_state_column}_hidden_states"].reshape(
                    row[f"{hidden_state_column}_hidden_states_original_shape"]
                ),
                axis=1,
            )
        return df

    @staticmethod
    def load_dataset_from_hub(path_to_dataset_hub: str) -> pd.DataFrame:
        dataset = load_dataset(path_to_dataset_hub)
        assert isinstance(dataset, Dataset | DatasetDict)
        dfs = []
        for split in ["train", "test", "val"]:
            dataset_split = dataset[split]
            assert isinstance(dataset_split, Dataset | DatasetDict)
            df = dataset_split.to_pandas()
            dfs.append(df)
        df = pd.concat(dfs)
        regex = r"(vanilla|graph_prompter_hf"
        model_types: List = []
        for column in df.columns:
            match = re.search(regex, column)
            if match:
                model_types.append(match.group(1))
        for attention_column in model_types:
            df[f"{attention_column}_attentions"] = df.apply(
                lambda row: row[f"{attention_column}_attentions"].reshape(
                    row[f"{attention_column}_attentions_original_shape"]
                ),
                axis=1,
            )

        for hidden_state_column in model_types:
            df[f"{hidden_state_column}_hidden_states"] = df.apply(
                lambda row: row[f"{hidden_state_column}_hidden_states"].reshape(
                    row[f"{hidden_state_column}_hidden_states_original_shape"]
                ),
                axis=1,
            )
        return df
