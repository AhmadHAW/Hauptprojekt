import os
from typing import List, Tuple, Callable, Optional, Dict, Union, Set, Union
from pathlib import Path
from abc import ABC, abstractmethod

import torch
from torch_geometric.data import download_url, extract_zip
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
import numpy as np


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


ROOT = "./data"  # The root path where models and datasets are saved at.

GNN_PATH = f"{ROOT}/gnn"  # The path, where gnn datasets and models are saved at.
GNN_TRAIN_DATASET_PATH = (
    f"{GNN_PATH}/train"  # The path where the gnn training dataset is saved at.
)
GNN_TEST_DATASET_PATH = (
    f"{GNN_PATH}/test"  # The path where the gnn test dataset is saved at.
)
GNN_VAL_DATASET_PATH = (
    f"{GNN_PATH}/val"  # The path where the gnn validation dataset is saved at.
)
GNN_COMPLETE_DATASET_PATH = (
    f"{GNN_PATH}/complete"  # The path where the complete gnn dataset is saved at.
)

LLM_PATH = f"{ROOT}/llm"  # The path, where LLM datasets and models are saved at.
LLM_DATASET_PATH = (
    f"{LLM_PATH}/dataset.csv"  # The path where the LLM dataset is saved at.
)
LLM_SOURCE_DF_PATH = (
    f"{LLM_PATH}/source.csv"  # The path where the LLM sources dataset is saved at.
)
LLM_TARGET_DF_PATH = (
    f"{LLM_PATH}/target.csv"  # The path where the LLM targets dataset is saved at.
)
LLM_MODEL_DIMENSION_PATH = f"{LLM_PATH}/{{}}"
PROMPT_KGE_DIMENSION = 4
EMBEDDING_KGE_DIMENSION = 128
LLM_PROMPT_PATH = f"{LLM_PATH}/prompt"
LLM_EMBEDDING_PATH = f"{LLM_PATH}/embedding"
LLM_VANILLA_PATH = f"{LLM_PATH}/vanilla"
LLM_PROMPT_TRAINING_PATH = f"{LLM_PROMPT_PATH}/training"  # The path where the LLM training outputs are saved at.
LLM_EMBEDDING_TRAINING_PATH = f"{LLM_EMBEDDING_PATH}/training"  # The path where the LLM training outputs are saved at.
LLM_VANILLA_TRAINING_PATH = f"{LLM_VANILLA_PATH}/training"  # The path where the LLM training outputs are saved at.
LLM_PROMPT_BEST_MODEL_PATH = f"{LLM_PROMPT_TRAINING_PATH}/best"  # The path where the best trained LLM model is saved at.
LLM_EMBEDDING_BEST_MODEL_PATH = f"{LLM_EMBEDDING_TRAINING_PATH}/best"  # The path where the best trained LLM model is saved at.
LLM_VANILLA_BEST_MODEL_PATH = f"{LLM_VANILLA_TRAINING_PATH}/best"  # The path where the best trained LLM model is saved at.
LLM_PROMPT_DATASET_PATH = f"{LLM_MODEL_DIMENSION_PATH}/prompt_dataset"  # The path where the huggingface prompt dataset (tokenized) is saved at.
LLM_EMBEDDING_DATASET_PATH = f"{LLM_MODEL_DIMENSION_PATH}/adding_dataset"  # The path where the huggingface prompt dataset (tokenized) is saved at.
LLM_VANILLA_DATASET_PATH = f"{LLM_PATH}/vanilla_dataset"  # The path where the huggingface vanilla dataset (tokenized) is saved at.
LLM_PROMPT_FIXED_DATASET_PATH = f"{LLM_MODEL_DIMENSION_PATH}/prompt_fixed_dataset"  # The path where the huggingface prompt dataset (tokenized) is saved at.
LLM_EMBEDDING_FIXED_DATASET_PATH = f"{LLM_MODEL_DIMENSION_PATH}/adding_fixed_dataset"  # The path where the huggingface prompt dataset (tokenized) is saved at.
LLM_VANILLA_FIXED_DATASET_PATH = f"{LLM_PATH}/vanilla_fixed_dataset"  # The path where the huggingface vanilla dataset (tokenized) is saved at.

PCA_PATH = f"{ROOT}/pca"

DIRS_TO_INIT = [
    LLM_PROMPT_PATH,
    LLM_EMBEDDING_PATH,
    GNN_PATH,
    LLM_VANILLA_PATH,
    PCA_PATH,
]

PROMPT_COLUMNS = ["source_id", "title", "genres"]


def row_to_adding_embedding_datapoint(
    row: pd.Series,
    sep_token: str = "[SEP]",
    pad_token: str = "[PAD]",
) -> str:
    prompt = row_to_vanilla_datapoint(row, sep_token)
    prompt = f"{prompt}{pad_token}{sep_token}{pad_token}"
    return prompt


def row_to_prompt_datapoint(row: pd.Series, sep_token: str = "[SEP]") -> str:
    prompt = row_to_vanilla_datapoint(row, sep_token)
    prompt_source_embedding = row["prompt_source_embedding"]
    prompt_target_embedding = row["prompt_target_embedding"]
    prompt = f"{prompt}{transform_embeddings_to_string(prompt_source_embedding)}{sep_token}{transform_embeddings_to_string(prompt_target_embedding)}"
    return prompt


def row_to_vanilla_datapoint(row: pd.Series, sep_token: str) -> str:
    prompt = ""
    for prompt_column in PROMPT_COLUMNS:
        prompt += f"{row[prompt_column]}{sep_token}"
    return prompt


class KGLoader(ABC):
    """
    The KGLoader manages and pre-processes the datasets for GNNs and LLMs.
    """

    def __init__(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        force_recompute: bool = False,
    ) -> None:
        """
        We expect the source df and target df to have at least the column "id" with consecutive sequence of numbers starting at 0
        In addition if source df and target df column begin with "feature_...", they are considered a one hot encoded feature of whatever it is.
        All other columns are ignored in source_df and target_df.

        We expect edge_df to have at least the columns "source_id" and "target_id" which each have to map to either the source_df ids and target_df ids.
        All dataframes need to have at least one entry.
        This returns a HeteroData object, which is then used to train the GNN
        """
        super().__init__()
        if (
            not self.data_present() or force_recompute
        ):  # If the datas have already been preprocessed and saved, we can pull them up from the disk.
            assert len(edge_df[~(edge_df["source_id"].isin(source_df["id"]))]) == 0
            assert len(edge_df[~(edge_df["target_id"].isin(target_df["id"]))]) == 0
            assert len(edge_df) > 0
            self.__create_dirs()
            self.source_df = source_df
            self.target_df = target_df
            self.edge_df = edge_df
            self.llm_df = edge_df.merge(
                source_df, left_on="source_id", right_on="id", how="left"
            ).merge(target_df, left_on="target_id", right_on="id", how="left")
            feature_columns = list(
                filter(
                    lambda column: column.startswith("feature_"), self.llm_df.columns
                )
            )
            self.llm_df = self.llm_df.drop(columns=feature_columns)
            self.llm_df["labels"] = 1
            self.data = self.generate_hetero_dataset()
            # split HeteroDataset dataset into train, dev and test
            self.split_data()
            # generate pandas dataframe with prompts and labels for LLM
            self.generate_llm_dataset()
            # split llm dataset according to the HeteroDataset split into train, dev, test and rest
            self.__split_llm_dataset()

            self.source_df.to_csv(LLM_SOURCE_DF_PATH, index=False)
            self.target_df.to_csv(LLM_TARGET_DF_PATH, index=False)
        else:
            self.load_datasets_from_disk()

    def data_present(self) -> bool:
        """Returns True, if GNN_TRAIN_DATASET_PATH, GNN_TEST_DATASET_PATH, GNN_VAL_DATASET_PATH, LLM_DATASET_PATH are files."""
        return (
            os.path.isfile(GNN_TRAIN_DATASET_PATH)
            and os.path.isfile(GNN_TEST_DATASET_PATH)
            and os.path.isfile(GNN_VAL_DATASET_PATH)
            and os.path.isfile(LLM_DATASET_PATH)
            and os.path.isfile(LLM_SOURCE_DF_PATH)
            and os.path.isfile(LLM_TARGET_DF_PATH)
        )

    def load_datasets_from_disk(self):
        self.gnn_train_data = torch.load(GNN_TRAIN_DATASET_PATH)
        self.gnn_val_data = torch.load(GNN_VAL_DATASET_PATH)
        self.gnn_test_data = torch.load(GNN_TEST_DATASET_PATH)
        self.data = torch.load(GNN_COMPLETE_DATASET_PATH)
        self.llm_df = pd.read_csv(LLM_DATASET_PATH)
        self.source_df = pd.read_csv(LLM_SOURCE_DF_PATH)
        self.target_df = pd.read_csv(LLM_TARGET_DF_PATH)

    def __create_dirs(self) -> None:
        """
        Create dir system if not exist
        """
        for directory in DIRS_TO_INIT:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def generate_hetero_dataset(self) -> HeteroData:
        """(from torch geometrics Tutorial but adjusted for the general use case)
        Original Quote:
            With this, we are ready to initialize our `HeteroData` object and pass the necessary information to it.
            Note that we also pass in a `node_id` vector to each node type in order to reconstruct the original node indices from sampled subgraphs.
            We also take care of adding reverse edges to the `HeteroData` object.
            This allows our GNN model to use both directions of the edge for message passing:
        """
        data = HeteroData()

        # Save node indices:
        data["source"].node_id = torch.tensor(self.source_df["id"].tolist())
        data["target"].node_id = torch.tensor(self.target_df["id"].tolist())
        source_one_hot_feature_cols = [
            col
            for col in self.source_df
            if col.startswith("feature_")  # type: ignore
        ]
        # Add the node features and edge indices:
        if len(source_one_hot_feature_cols) > 0:
            data["source"].x = torch.from_numpy(
                self.source_df[source_one_hot_feature_cols].values
            ).to(
                torch.float
            )  # expects a float tensor one hot encoding of shape[target nodes, feature] either 0.0 or 1.0
        target_one_hot_feature_cols = [
            col
            for col in self.target_df
            if col.startswith("feature_")  # type: ignore
        ]
        # Add the node features and edge indices:
        if len(target_one_hot_feature_cols) > 0:
            data["target"].x = torch.from_numpy(
                self.target_df[target_one_hot_feature_cols].values
            ).to(
                torch.float
            )  # expects a float tensor one hot encoding of shape[target nodes, feature] either 0.0 or 1.0
        data["source", "edge", "target"].edge_index = torch.stack(
            [
                torch.tensor(self.edge_df["source_id"].tolist()),
                torch.tensor(self.edge_df["target_id"].tolist()),
            ]
        )

        # We also need to make sure to add the reverse edges from movies to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        transform = T.ToUndirected()
        data = transform(data)

        assert data.node_types == ["source", "target"]
        assert data.edge_types == [
            ("source", "edge", "target"),
            ("target", "rev_edge", "source"),
        ]
        assert data["source"].num_nodes == len(self.source_df)
        assert data["target"].num_nodes == len(self.target_df)
        assert data["source"].num_features == len(source_one_hot_feature_cols)
        assert data["target"].num_features == len(target_one_hot_feature_cols)
        assert data["source", "edge", "target"].num_edges == len(self.edge_df)
        assert data["target", "rev_edge", "source"].num_edges == len(self.edge_df)
        return data

    def split_data(self) -> None:
        """(From Torch Geometrics Tutorial)
        Original Quote:
            Defining Edge-level Training Splits
            Since our data is now ready-to-be-used, we can split the ratings of users into training, validation, and test splits.
            This is needed in order to ensure that we leak no information about edges used during evaluation into the training phase.
            For this, we make use of the [`transforms.RandomLinkSplit`]
            (https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.RandomLinkSplit) transformation from PyG.
            This transforms randomly divides the edges in the `("user", "rates", "movie")` into training, validation and test edges.
            The `disjoint_train_ratio` parameter further separates edges in the training split into edges used for message passing (`edge_index`)
            and edges used for supervision (`edge_label_index`).
            Note that we also need to specify the reverse edge type `("movie", "rev_rates", "user")`.
            This allows the `RandomLinkSplit` transform to drop reverse edges accordingly to not leak any information into the training phase.
        Additional:
            We are expecting for this methiod the constants GNN_TRAIN_DATASET_PATH, GNN_VAL_DATASET_PATH, GNN_TEST_DATASET_PATH and GNN_COMPLETE_DATASET_PATH
            to be defined as valid paths for large torch tensors.
        """
        # For this, we first split the set of edges into
        # training (80%), validation (10%), and testing edges (10%).
        # Across the training edges, we use 70% of edges for message passing,
        # and 30% of edges for supervision.
        # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
        # Negative edges during training will be generated on-the-fly, so we don't want to
        # add them to the graph right away.
        # Overall, we can leverage the `RandomLinkSplit()` transform for this from PyG:
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=("source", "edge", "target"),
            rev_edge_types=("target", "rev_edge", "source"),
        )
        self.gnn_train_data, self.gnn_val_data, self.gnn_test_data = transform(
            self.data
        )
        self.gnn_train_data: HeteroData = self.gnn_train_data
        self.gnn_val_data: HeteroData = self.gnn_val_data
        self.gnn_test_data: HeteroData = self.gnn_test_data
        torch.save(self.gnn_train_data, GNN_TRAIN_DATASET_PATH)
        torch.save(self.gnn_val_data, GNN_VAL_DATASET_PATH)
        torch.save(self.gnn_test_data, GNN_TEST_DATASET_PATH)
        torch.save(self.data, GNN_COMPLETE_DATASET_PATH)

    def generate_llm_dataset(self, sep_token="[SEP]") -> None:
        """
        This method produces prompts given the how row_to_vanilla_datapoint is defined in the inheriting class
        """
        print("generate llm dataset...")
        self.llm_df["prompt"] = self.llm_df.apply(
            lambda row: row_to_vanilla_datapoint(row, sep_token=sep_token),
            axis=1,
        )

    def append_prompt_graph_embeddings(
        self, graph_embeddings: pd.DataFrame, save: bool = True
    ):
        assert len(self.llm_df) == len(graph_embeddings)
        self.llm_df["prompt_source_embedding"] = graph_embeddings["source_embedding"]
        self.llm_df["prompt_target_embedding"] = graph_embeddings["target_embedding"]
        if save:
            self.save_llm_df()

    def append_embedding_graph_embeddings(
        self, graph_embeddings: pd.DataFrame, save: bool = True
    ):
        assert len(self.llm_df) == len(graph_embeddings)
        self.llm_df["embedding_source_embedding"] = graph_embeddings["source_embedding"]
        self.llm_df["embedding_target_embedding"] = graph_embeddings["target_embedding"]
        if save:
            self.save_llm_df()

    def __is_in_split(self, edge_index, source_id, target_id) -> bool:
        """
        This methods returns the True if the current gnn dataset split contains the edge between given user and movie.
        """
        test_tensor = torch.Tensor([source_id, target_id])
        return len(torch.nonzero(torch.all(edge_index == test_tensor, dim=1))) > 0

    def __find_split(
        self, row, train_edge_index, val_edge_index, test_edge_index
    ) -> str:
        """
        Returns the split train, test, val or rest if the given edge is found in either gnn dataset split.
        the datapoint is assigned to train if present or if overlapping between val and test in a 50/50 proportion.
        """
        source_id = row["source_id"]
        target_id = row["target_id"]
        if self.__is_in_split(train_edge_index, source_id, target_id):
            split = "train"
        elif self.__is_in_split(val_edge_index, source_id, target_id):
            split = "val" if self.__last == "test" else "test"
            self.__last = split
        elif self.__is_in_split(test_edge_index, source_id, target_id):
            split = "test"
        else:
            split = "rest"
        return split

    def __split_llm_dataset(self) -> None:
        """
        This method assigns all datapoints in the LLM dataset to the same split as they are found in the gnn dataset split.
        """
        train_edge_index = self.gnn_train_data["source", "edge", "target"][
            "edge_index"
        ].T
        test_edge_index = self.gnn_val_data["source", "edge", "target"]["edge_index"].T
        val_edge_index = self.gnn_test_data["source", "edge", "target"]["edge_index"].T
        self.__last = "test"
        print("splitting LLM dataset")
        self.llm_df["split"] = self.llm_df.apply(
            lambda row: self.__find_split(
                row, train_edge_index, val_edge_index, test_edge_index
            ),
            axis=1,
        )
        self.save_llm_df()

    def replace_llm_df(self, df: pd.DataFrame):
        """
        overrides the current llm dataset with the given dataset."""
        self.llm_df = df
        self.save_llm_df()

    def save_llm_df(self):
        """
        We expect the the constant LLM_DATASET_PATH to be a valid path for a csv file and GNN_PATH a valid dir to where large torch tensors can be saved.
        """
        columns_without_embeddings = list(
            filter(lambda column: "embedding" not in column, self.llm_df.columns)
        )
        columns_with_embeddings = list(
            filter(lambda column: "embedding" in column, self.llm_df.columns)
        )
        self.llm_df[columns_without_embeddings].to_csv(LLM_DATASET_PATH, index=False)
        for column in columns_with_embeddings:
            torch.save(
                torch.tensor(self.llm_df[column].tolist()), f"{GNN_PATH}/{column}.pt"
            )


def create_dirs(dirs_to_init: List[str]) -> None:
    """
    Create dir structure if not exist
    """
    for directory in dirs_to_init:
        if not os.path.exists(directory):
            os.makedirs(directory)


class MovieLensLoader(KGLoader):
    """
    The MovieLensLoader manages the original graph data set and the pre-processed data sets for GNNs and LLMs.
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
        if not self.data_present() or force_recompute:
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
            self.load_datasets_from_disk()

    def __download_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Downloads the https://files.grouplens.org/datasets/movielens/ml-latest-small.zip dataset and
        extracts the content into ROOT//ml-latest-small/"""
        movies_path = f"{ROOT}/ml-latest-small/movies.csv"
        ratings_path = f"{ROOT}/ml-latest-small/ratings.csv"
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
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
            pd.get_dummies(target_df["genres"].explode(), prefix="feature")
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
        return source_df, target_df, edge_df

    def __dataset_from_df(self, df: pd.DataFrame) -> DatasetDict:
        """
        Generates the LLM datasets from the pandas dataframe.
        """
        dataset_train = Dataset.from_pandas(df[df["split"] == "train"])
        dataset_val = Dataset.from_pandas(df[df["split"] == "val"])
        dataset_test = Dataset.from_pandas(df[df["split"] == "test"])
        return DatasetDict(
            {
                "train": dataset_train,
                "val": dataset_val,
                "test": dataset_test,
            }
        )

    def generate_prompt_embedding_dataset(
        self,
        tokenize_function: Optional[Callable] = None,
        sep_token: str = "[SEP]",
        suffix="",
        force_recompute: bool = False,
    ) -> Union[DatasetDict, Dataset]:
        """
        Generates the dataset for training the prompt model,
        by passing the tokenizer.tokenize function and
        the embedding dimension of the target prompt model.
        """
        llm_prompt_dataset_path = LLM_PROMPT_DATASET_PATH + suffix
        if os.path.exists(llm_prompt_dataset_path) and not force_recompute:
            dataset = datasets.load_from_disk(llm_prompt_dataset_path)
        else:
            assert "prompt_source_embedding" in self.llm_df
            assert "prompt_target_embedding" in self.llm_df
            assert self.llm_df["prompt_source_embedding"].notna().all()
            assert self.llm_df["prompt_target_embedding"].notna().all()
            llm_df = self.llm_df.copy(deep=True)
            llm_df["prompt"] = self.llm_df.apply(
                lambda row: row_to_prompt_datapoint(row, sep_token=sep_token),
                axis=1,
            )
            dataset = self.__dataset_from_df(llm_df)
            if tokenize_function:
                dataset = dataset.map(tokenize_function, batched=True)
            dataset.save_to_disk(llm_prompt_dataset_path)

        return dataset

    def __generate_embeddings(self, row) -> torch.Tensor:
        source_embeddings = row["embedding_source_embedding"]
        target_embeddings = row["embedding_target_embedding"]
        embeddings = torch.tensor([source_embeddings, target_embeddings])
        return embeddings

    def generate_embedding_dataset(
        self,
        sep_token,
        pad_token,
        tokenize_function: Optional[Callable] = None,
        suffix="",
        force_recompute: bool = False,
    ) -> Union[DatasetDict, Dataset]:
        """
        Generates the dataset for training the adding model,
        by passing the tokenizer.tokenize function and
        the embedding dimension of the target adding model.
        """
        llm_adding_dataset_path = LLM_EMBEDDING_DATASET_PATH + suffix
        if os.path.exists(llm_adding_dataset_path) and not force_recompute:
            dataset = datasets.load_from_disk(llm_adding_dataset_path)
        else:
            assert "embedding_source_embedding" in self.llm_df
            assert "embedding_target_embedding" in self.llm_df
            assert self.llm_df["embedding_source_embedding"].notna().all()
            assert self.llm_df["embedding_target_embedding"].notna().all()
            llm_df = self.llm_df.copy(deep=True)
            llm_df["prompt"] = llm_df.apply(
                lambda row: row_to_adding_embedding_datapoint(
                    row, sep_token, pad_token
                ),
                axis=1,
            )
            llm_df["graph_embeddings"] = llm_df.apply(
                lambda row: self.__generate_embeddings(row),  # type: ignore
                axis=1,
            )  # type: ignore
            llm_df["graph_embeddings"] = llm_df["graph_embeddings"].apply(
                lambda embeddings: embeddings.detach().to("cpu").tolist()
            )
            dataset = self.__dataset_from_df(llm_df)
            if tokenize_function:
                dataset = dataset.map(tokenize_function, batched=True)
            dataset.save_to_disk(llm_adding_dataset_path)
        return dataset

    def generate_vanilla_dataset(
        self,
        tokenize_function: Optional[Callable] = None,
        sep_token="[SEP]",
        suffix="",
        force_recompute: bool = False,
    ) -> Union[DatasetDict, Dataset]:
        """
        Generates the dataset for training the vanilla model,
        by passing the tokenizer.tokenize function.
        """
        filepath = LLM_VANILLA_DATASET_PATH + suffix
        if os.path.exists(filepath) and not force_recompute:
            dataset = datasets.load_from_disk(filepath)
        else:
            llm_df = self.llm_df.copy(deep=True)
            llm_df["prompt"] = self.llm_df.apply(
                lambda row: row_to_vanilla_datapoint(row, sep_token=sep_token),
                axis=1,
            )
            llm_df["prompt"].iloc[0]
            dataset = self.__dataset_from_df(llm_df)
            if tokenize_function:
                dataset = dataset.map(tokenize_function, batched=True)
            dataset.save_to_disk(filepath)
        return dataset

    def add_graph_embeddings(
        self,
        row: pd.Series,
        get_prompt_embedding_cb: Callable,
        get_embedding_embedding_cb: Callable,
    ) -> pd.Series:
        split = row["split"]
        source_id = row["source_id"]
        target_id = row["target_id"]
        data = (
            self.gnn_train_data
            if split == "train"
            else self.gnn_val_data
            if split == "val"
            else self.gnn_test_data
            if split == "test"
            else self.data
        )
        prompt_source_embedding, prompt_target_embedding = get_prompt_embedding_cb(
            data, source_id, target_id
        )
        row["prompt_source_embedding"] = prompt_source_embedding.detach().tolist()
        row["prompt_target_embedding"] = prompt_target_embedding.detach().tolist()
        embedding_source_embedding, embedding_target_embedding = (
            get_embedding_embedding_cb(data, source_id, target_id)
        )
        row["embedding_source_embedding"] = embedding_source_embedding.detach().tolist()
        row["embedding_target_embedding"] = embedding_target_embedding.detach().tolist()
        return row

    def add_false_edges(
        self,
        false_ratio: float = 2.0,
        prompt_get_embedding_cb: Optional[Callable] = None,
        embedding_get_embedding_cb: Optional[Callable] = None,
        sep_token="[SEP]",
    ):
        if 0 not in self.llm_df["labels"].unique():
            df = self.llm_df.copy()
            df["labels"] = 1
            already_added_pairs = set()
            new_rows = []
            split_proportions = [
                int(len(df[df["split"] == "train"]) * false_ratio),
                int(len(df[df["split"] == "test"]) * false_ratio),
                int(len(df[df["split"] == "val"]) * false_ratio),
            ]
            for split_proportion, split in zip(
                split_proportions, ["train", "test", "val"]
            ):
                print(f"Adding {split_proportion} false edges for {split}.")
                for idx in range(split_proportion):
                    source_id, target_id = _find_non_existing_source_target(
                        self.llm_df, already_added_pairs
                    )
                    random_source: pd.DataFrame = (
                        self.source_df[self.source_df["id"] == source_id]
                        .sample(1)
                        .rename(columns={"id": "source_id"})
                        .reset_index(drop=True)
                    )
                    random_target: pd.DataFrame = (
                        self.target_df[self.target_df["id"] == target_id]
                        .sample(1)
                        .rename(columns={"id": "target_id"})
                        .reset_index(drop=True)
                    )
                    random_row = pd.concat([random_source, random_target], axis=1).iloc[
                        0
                    ]
                    random_row["target_id"] = target_id
                    random_row["labels"] = 0
                    random_row["split"] = split
                    if prompt_get_embedding_cb and embedding_get_embedding_cb:
                        random_row = self.add_graph_embeddings(
                            random_row,
                            prompt_get_embedding_cb,
                            embedding_get_embedding_cb,
                        )
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df["prompt"] = df.apply(
                lambda row: row_to_vanilla_datapoint(row, sep_token=sep_token), axis=1
            )
            self.replace_llm_df(df)
        return self.llm_df

    def sample_datapoints(
        self,
        existing,
        vanilla_classifier,
        prompt_classifier,
        embedding_classifier,
        prompt_getting_embedding_cb,
        embedding_getting_embedding_cb,
        singular_title=False,
        singular_user=False,
        split: str = "val",
        sep_token="[SEP]",
        pad_token="[PAD]",
    ):
        df = self.llm_df[self.llm_df["split"] == split]
        if singular_user:
            df = df[df["mappedUserId"] == df.sample(1).iloc[0]["mappedUserId"]]
        elif singular_title:
            df = df[df["mappedMovieId"] == df.sample(1).iloc[0]["mappedMovieId"]]
        if existing:
            random_row = df.sample(1).iloc[0]
            random_row["graph_embeddings"] = self.__generate_embeddings(random_row)
            print(random_row[f"user_embedding_{prompt_classifier.kge_dimension}"])
        else:
            dataset = (
                self.gnn_train_data
                if split == "train"
                else self.gnn_val_data
                if split == "val"
                else self.gnn_test_data
                if split == "test"
                else self.data
            )
            existing = True
            while existing:
                user_id = df["mappedUserId"].sample(1).iloc[0]
                random_row = self.llm_df.sample(1).iloc[0]
                movie_id = random_row["mappedMovieId"]
                existing = (
                    (self.llm_df["mappedMovieId"] == movie_id)
                    & (self.llm_df["mappedUserId"] == user_id)
                ).any()
                if not existing:
                    random_row = random_row.copy(deep=True)
                    random_row["mappedUserId"] = user_id
                    user_embedding_prompt, movie_embedding_prompt = (
                        prompt_getting_embedding_cb(dataset, user_id, movie_id)
                    )
                    random_row[f"user_embedding_{prompt_classifier.kge_dimension}"] = (
                        user_embedding_prompt
                    )
                    random_row[f"movie_embedding_{prompt_classifier.kge_dimension}"] = (
                        movie_embedding_prompt
                    )
                    user_embedding_embedding, movie_embedding_embedding = (
                        embedding_getting_embedding_cb(dataset, user_id, movie_id)
                    )
                    random_row[
                        f"user_embedding_{embedding_classifier.kge_dimension}"
                    ] = user_embedding_embedding
                    random_row[
                        f"movie_embedding_{embedding_classifier.kge_dimension}"
                    ] = movie_embedding_embedding
        prompt_vanilla = row_to_vanilla_datapoint(random_row, sep_token=sep_token)
        prompt_prompt = row_to_prompt_datapoint(
            random_row,
            sep_token=sep_token,
        )
        prompt_embedding = row_to_adding_embedding_datapoint(
            random_row, sep_token=sep_token, pad_token=pad_token
        )
        labels = 1 if existing else 0
        result_vanilla = {"prompt": prompt_vanilla, "labels": labels}
        result_prompt = {"prompt": prompt_prompt, "labels": labels}
        graph_embeddings = torch.stack(
            (
                random_row[f"user_embedding_{embedding_classifier.kge_dimension}"],
                random_row[f"movie_embedding_{embedding_classifier.kge_dimension}"],
            )
        )
        result_embedding = {
            "prompt": prompt_embedding,
            "labels": labels,
            "graph_embeddings": graph_embeddings.unsqueeze(dim=0),
        }
        return (
            vanilla_classifier.tokenize_function(result_vanilla, return_pt=True),
            prompt_classifier.tokenize_function(result_prompt, return_pt=True),
            embedding_classifier.tokenize_function(result_embedding, return_pt=True),
            random_row,
        )

    def replace_llm_df(self, df: pd.DataFrame):
        """
        overrides the current llm dataset with the given dataset."""
        self.llm_df = df
        self.save_llm_df()

    def save_llm_df(self):
        columns_without_embeddings = list(
            filter(lambda column: "embedding" not in column, self.llm_df.columns)
        )
        columns_with_embeddings = list(
            filter(lambda column: "embedding" in column, self.llm_df.columns)
        )
        self.llm_df[columns_without_embeddings].to_csv(LLM_DATASET_PATH, index=False)
        for column in columns_with_embeddings:
            torch.save(
                torch.tensor(self.llm_df[column].tolist()), f"{GNN_PATH}/{column}.pt"
            )
