import os
from typing import List, Tuple, Callable, Optional, Dict, Union, Set, Union
from abc import ABC
import ast

import torch
from torch_geometric.data import download_url, extract_zip
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import pandas as pd
import datasets
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from transformers import PreTrainedTokenizer

from utils import (
    find_non_existing_source_target,
    row_to_vanilla_datapoint,
    row_to_prompt_datapoint,
    row_to_input_embeds_replace_datapoint,
)


ROOT = "./data"  # The root path where models and datasets are saved at.

PROMPT_KGE_DIMENSION = 4
INPUT_EMBEDS_REPLACE_KGE_DIMENSION = 128


class KGManger(ABC):
    """
    The KGManger manages and pre-processes the datasets for GNNs and LLMs.
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
                    lambda column: column.startswith("gnn_feature_"),
                    self.llm_df.columns,
                )
            )
            self.llm_df = self.llm_df.drop(columns=feature_columns)
            self.llm_df["labels"] = 1
            self.data = self.generate_hetero_dataset()
            # split HeteroDataset dataset into train, dev and test
            self.split_data()
            # split llm dataset according to the HeteroDataset split into train, dev, test and rest
            self.__split_llm_dataset()
            # generate pandas dataframe with prompts and labels for LLM
            self.generate_llm_dataset()

            self.source_df.to_csv(f"{ROOT}/llm/source.csv", index=False)
            self.target_df.to_csv(f"{ROOT}/llm/target.csv", index=False)
        else:
            self.load_datasets_from_disk()

    def data_present(self) -> bool:
        """Returns True, if GNN_TRAIN_DATASET_PATH, f"{ROOT}/gnn/test", f"{ROOT}/gnn/val", f"{ROOT}/llm/dataset.csv" are files."""
        return (
            os.path.isfile(f"{ROOT}/gnn/train")
            and os.path.isfile(f"{ROOT}/gnn/test")
            and os.path.isfile(f"{ROOT}/gnn/val")
            and os.path.isfile(f"{ROOT}/llm/dataset.csv")
            and os.path.isfile(f"{ROOT}/llm/source.csv")
            and os.path.isfile(f"{ROOT}/llm/target.csv")
        )

    def load_datasets_from_disk(self):
        self.gnn_train_data = torch.load(f"{ROOT}/gnn/train")
        self.gnn_val_data = torch.load(f"{ROOT}/gnn/val")
        self.gnn_test_data = torch.load(f"{ROOT}/gnn/test")
        self.data = torch.load(f"{ROOT}/gnn/complete")
        self.llm_df = pd.read_csv(f"{ROOT}/llm/dataset.csv")
        self.source_df = pd.read_csv(f"{ROOT}/llm/source.csv")
        self.target_df = pd.read_csv(f"{ROOT}/llm/target.csv")

    def __create_dirs(self) -> None:
        """
        Create dir system if not exist
        """
        for directory_suffix in [
            "gnn",
            "llm/vanilla",
            "llm/prompt",
            "llm/input_embeds_replace",
        ]:
            directory = f"{ROOT}/{directory_suffix}"
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
            if col.startswith("gnn_feature_")  # type: ignore
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
            if col.startswith("gnn_feature_")  # type: ignore
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
            We are expecting for this method the constants the plits of the dataset to be present and
            to be defined as valid paths for large torch tensors.
        """
        # For this, we first split the set of edges into
        # training (80%), validation (10%), and testing edges (10%).
        # Across the training edges, we use 70% of edges for message passing,
        # and 30% of edges for supervision.
        # We further want to generate fixed negative edges for evaluation with a ratio of 1:1.
        # Negative edges during training will be generated on-the-fly, so we don't want to
        # add them to the graph right away.
        # Overall, we can leverage the `RandomLinkSplit()` transform for this from PyG:
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=1.0,
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
        torch.save(self.gnn_train_data, f"{ROOT}/gnn/train")
        torch.save(self.gnn_val_data, f"{ROOT}/gnn/val")
        torch.save(self.gnn_test_data, f"{ROOT}/gnn/test")
        torch.save(self.data, f"{ROOT}/gnn/complete")

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

    def append_input_embeds_replace_graph_embeddings(
        self, graph_embeddings: pd.DataFrame, save: bool = True
    ):
        assert len(self.llm_df) == len(graph_embeddings)
        self.llm_df["input_embeds_replace_source_embedding"] = graph_embeddings[
            "source_embedding"
        ]
        self.llm_df["input_embeds_replace_target_embedding"] = graph_embeddings[
            "target_embedding"
        ]
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
        We expect the the constant f"{ROOT}/llm/dataset.csv" to be a valid path for a csv file and GNN_PATH a valid dir to where large torch tensors can be saved.
        """
        columns_without_embeddings = list(
            filter(lambda column: "embedding" not in column, self.llm_df.columns)
        )
        columns_with_embeddings = list(
            filter(lambda column: "embedding" in column, self.llm_df.columns)
        )
        self.llm_df[columns_without_embeddings].to_csv(
            f"{ROOT}/llm/dataset.csv", index=False
        )
        for column in columns_with_embeddings:
            torch.save(
                torch.tensor(self.llm_df[column].tolist()), f"{ROOT}/gnn/{column}.pt"
            )

    def generate_prompt_embedding_dataset(
        self,
        tokenize_function: Optional[Callable] = None,
        sep_token: str = "[SEP]",
        suffix="",
        df: Optional[pd.DataFrame] = None,
        force_recompute: bool = False,
    ) -> Union[DatasetDict, Dataset]:
        """
        Generates the dataset for training the prompt model,
        by passing the tokenizer.tokenize function and
        the embedding dimension of the target prompt model.
        """
        llm_prompt_dataset_path = f"{ROOT}/llm/prompt{suffix}/dataset"
        if os.path.exists(llm_prompt_dataset_path) and not force_recompute:
            dataset = datasets.load_from_disk(llm_prompt_dataset_path)
        else:
            assert "prompt_source_embedding" in self.llm_df
            assert "prompt_target_embedding" in self.llm_df
            assert self.llm_df["prompt_source_embedding"].notna().all()
            assert self.llm_df["prompt_target_embedding"].notna().all()
            if isinstance(df, pd.DataFrame):
                llm_df = df.copy(deep=True)
            else:
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
        source_embeddings = row["input_embeds_replace_source_embedding"]
        target_embeddings = row["input_embeds_replace_target_embedding"]
        embeddings = torch.tensor([source_embeddings, target_embeddings])
        return embeddings

    def generate_input_embeds_replace_embedding_dataset(
        self,
        sep_token,
        pad_token,
        tokenize_function: Optional[Callable] = None,
        suffix="",
        df: Optional[pd.DataFrame] = None,
        force_recompute: bool = False,
    ) -> Union[DatasetDict, Dataset]:
        """
        Generates the dataset for training the adding model,
        by passing the tokenizer.tokenize function and
        the embedding dimension of the target adding model.
        """
        llm_adding_dataset_path = f"{ROOT}/llm/input_embeds_replace{suffix}/dataset"
        if os.path.exists(llm_adding_dataset_path) and not force_recompute:
            dataset = datasets.load_from_disk(llm_adding_dataset_path)
        else:
            assert "input_embeds_replace_source_embedding" in self.llm_df
            assert "input_embeds_replace_target_embedding" in self.llm_df
            assert self.llm_df["input_embeds_replace_source_embedding"].notna().all()
            assert self.llm_df["input_embeds_replace_target_embedding"].notna().all()
            if isinstance(df, pd.DataFrame):
                llm_df = df.copy(deep=True)
            else:
                llm_df = self.llm_df.copy(deep=True)
            llm_df["prompt"] = llm_df.apply(
                lambda row: row_to_input_embeds_replace_datapoint(
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
        df: Optional[pd.DataFrame] = None,
        force_recompute: bool = False,
    ) -> Union[DatasetDict, Dataset]:
        """
        Generates the dataset for training the vanilla model,
        by passing the tokenizer.tokenize function.
        """
        filepath = f"{ROOT}/llm/vanilla{suffix}/dataset"
        if os.path.exists(filepath) and not force_recompute:
            dataset = datasets.load_from_disk(filepath)
        else:
            if isinstance(df, pd.DataFrame):
                llm_df = df.copy(deep=True)
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
        get_attention_embedding_cb: Callable,
    ) -> pd.Series:
        split = row["split"]
        source_id = row["source_id"]
        target_id = row["target_id"]
        data = None
        if split == "train":
            data = self.gnn_train_data
        elif split == "val":
            data = self.gnn_val_data
        elif split == "test":
            data = self.gnn_test_data
        else:
            raise Exception("no split found where it should have")
        prompt_source_embedding, prompt_target_embedding = get_prompt_embedding_cb(
            data, source_id, target_id
        )
        row["prompt_source_embedding"] = prompt_source_embedding.detach().tolist()
        row["prompt_target_embedding"] = prompt_target_embedding.detach().tolist()
        input_embeds_replace_source_embedding, input_embeds_replace_target_embedding = (
            get_attention_embedding_cb(data, source_id, target_id)
        )
        row["input_embeds_replace_source_embedding"] = (
            input_embeds_replace_source_embedding.detach().tolist()
        )
        row["input_embeds_replace_target_embedding"] = (
            input_embeds_replace_target_embedding.detach().tolist()
        )
        return row

    def add_false_edges(
        self,
        false_ratio: float = 1.0,
        prompt_get_embedding_cb: Optional[Callable] = None,
        attention_get_embedding_cb: Optional[Callable] = None,
        sep_token="[SEP]",
        splits=["train", "test", "val"],
    ):
        df = self.llm_df.copy()
        already_added_pairs = set()
        new_rows = []
        for split in splits:
            split_proportion = int((df["split"] == split).sum() * false_ratio)
            print(f"Adding {split_proportion} false edges for {split}.")
            for idx in range(split_proportion):
                source_id, target_id = find_non_existing_source_target(
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
                random_row = pd.concat([random_source, random_target], axis=1).iloc[0]
                random_row["labels"] = 0
                random_row["split"] = split
                if prompt_get_embedding_cb and attention_get_embedding_cb:
                    random_row = self.add_graph_embeddings(
                        random_row,
                        prompt_get_embedding_cb,
                        attention_get_embedding_cb,
                    )
                new_rows.append(random_row)
        new_rows_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_rows_df], ignore_index=True)
        df["prompt"] = df.apply(
            lambda row: row_to_vanilla_datapoint(row, sep_token=sep_token),
            axis=1,
        )
        self.replace_llm_df(df)
        return df

    def generate_huggingface_dataset(
        self,
        dfs: List[pd.DataFrame],
        df_prefix_list: List[str],
    ) -> DatasetDict:
        assert len(dfs) == len(df_prefix_list)
        len_df = len(dfs[0])
        for df in dfs:
            assert len(df) == len_df
            df["attentions_original_shape"] = df["attentions"].apply(
                lambda attention: attention.shape
            )
            df["attentions"] = df["attentions"].apply(
                lambda attention: attention.flatten()
            )
            df["hidden_states_original_shape"] = df["hidden_states"].apply(
                lambda hidden_states: hidden_states.shape
            )
            df["hidden_states"] = df["hidden_states"].apply(
                lambda attention: attention.flatten()
            )
        export_llm = self.llm_df
        for df, df_prefix in zip(dfs, df_prefix_list):
            export_llm = export_llm.merge(
                df[
                    [
                        "source_id",
                        "target_id",
                        "attentions",
                        "hidden_states",
                        "attentions_original_shape",
                        "hidden_states_original_shape",
                    ]
                ].rename(
                    columns={
                        "attentions": f"{df_prefix}_attentions",
                        "hidden_states": f"{df_prefix}_hidden_states",
                        "attentions_original_shape": f"{df_prefix}_attentions_original_shape",
                        "hidden_states_original_shape": f"{df_prefix}_hidden_states_original_shape",
                    }
                ),
                on=["source_id", "target_id"],
            )
        dataset = self.__dataset_from_df(export_llm)
        return dataset

    def __dataset_from_df(self, df: pd.DataFrame) -> DatasetDict:
        """
        Generates the LLM datasets from the pandas dataframe.
        """
        dataset_train = Dataset.from_pandas(
            df[df["split"] == "train"], preserve_index=False
        )
        dataset_val = Dataset.from_pandas(
            df[df["split"] == "val"], preserve_index=False
        )
        dataset_test = Dataset.from_pandas(
            df[df["split"] == "test"], preserve_index=False
        )
        return DatasetDict(
            {
                "train": dataset_train,
                "val": dataset_val,
                "test": dataset_test,
            }
        )

    def fill_all_semantic_tokens(
        self,
        all_semantic_tokens: List[List],
        input_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        all_semantic_positional_encoding: torch.Tensor,
    ) -> None:
        ends = all_semantic_positional_encoding[:, :, 1]
        starts = all_semantic_positional_encoding[:, :, 0]
        # input: # ends: torch.tensor([2, 5, 6]) starts: tensor([0, 2, 4])
        # Compute the maximum length of the ranges
        max_length = (ends - starts).max()
        # Create a range tensor from 0 to max_length-1
        range_tensor = torch.arange(max_length).unsqueeze(0)  # type: ignore
        for pos, semantic_tokens in enumerate(all_semantic_tokens):
            # Compute the ranges using broadcasting and masking
            ranges = starts[:, pos].unsqueeze(1) + range_tensor
            mask = ranges < ends[:, pos].unsqueeze(1)

            # Apply the mask
            result = (
                ranges * mask
            )  # result: tensor([[0, 1, 0], [2, 3, 4], [4, 5, 0]]) here padding index is 0
            #                        -                     -    positions were padded
            # result = result.unsqueeze(dim = 2).repeat(1,1, input_ids.shape[2])
            gather = input_ids.gather(dim=1, index=result)
            decoded = tokenizer.batch_decode(gather, skip_special_tokens=False)
            semantic_tokens.extend([decode.replace(" [CLS]", "") for decode in decoded])


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
        all_semantic_positional_encoding: torch.Tensor,
    ) -> pd.DataFrame:
        all_semantic_positional_encoding = all_semantic_positional_encoding[
            :, [1, 3, 5, 7]
        ]
        user_ids = []
        titles = []
        genres = []
        movie_ids = []
        all_semantic_tokens = [user_ids, movie_ids, titles, genres]
        self.fill_all_semantic_tokens(
            all_semantic_tokens, input_ids, tokenizer, all_semantic_positional_encoding
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
        all_semantic_positional_encoding: torch.Tensor,
    ) -> Tuple[pd.DataFrame, torch.Tensor]:
        all_semantic_positional_encoding = all_semantic_positional_encoding[
            :, [1, 3, 5, 7, 9, 11]
        ]
        all_semantic_tokens = [[], [], [], [], [], []]
        self.fill_all_semantic_tokens(
            all_semantic_tokens, input_ids, tokenizer, all_semantic_positional_encoding
        )
        all_semantic_tokens[0] = [int(id) for id in all_semantic_tokens[0]]
        all_semantic_tokens[1] = [int(id) for id in all_semantic_tokens[1]]
        all_semantic_tokens[3] = [
            ast.literal_eval(string_list) for string_list in all_semantic_tokens[3]
        ]
        all_semantic_tokens[4] = [
            [
                float(str_float)
                for str_float in ast.literal_eval(string_list.replace(" ", ""))
            ]
            for string_list in all_semantic_tokens[4]
        ]
        all_semantic_tokens[5] = [
            [
                float(str_float)
                for str_float in ast.literal_eval(string_list.replace(" ", ""))
            ]
            for string_list in all_semantic_tokens[5]
        ]
        user_embeddings = torch.tensor(all_semantic_tokens[4])
        movie_embeddings = torch.tensor(all_semantic_tokens[5])
        graph_embeddings = torch.stack([user_embeddings, movie_embeddings]).permute(
            (1, 0, 2)
        )
        data = {
            "source_id": all_semantic_tokens[0],
            "target_id": all_semantic_tokens[1],
            "title": all_semantic_tokens[2],
            "genres": all_semantic_tokens[3],
        }
        df = pd.DataFrame(data)
        return df, graph_embeddings

    @staticmethod
    def load_dataset_from_disk(path_to_hf_dataset: str) -> pd.DataFrame:
        dataset = load_from_disk(path_to_hf_dataset)
        dfs = []
        for split in ["train", "test", "val"]:
            df = dataset[split].to_pandas()
            dfs.append(df)
        df = pd.concat(dfs)
        for attention_column in ["vanilla", "prompt", "input_embeds_replace"]:
            df[f"{attention_column}_attentions"] = df.apply(
                lambda row: row[f"{attention_column}_attentions"].reshape(
                    row[f"{attention_column}_attentions_original_shape"]
                ),
                axis=1,
            )

        for hidden_state_column in ["vanilla", "prompt", "input_embeds_replace"]:
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
        dfs = []
        for split in ["train", "test", "val"]:
            df = dataset[split].to_pandas()
            dfs.append(df)
        df = pd.concat(dfs)
        for attention_column in ["vanilla", "prompt", "input_embeds_replace"]:
            df[f"{attention_column}_attentions"] = df.apply(
                lambda row: row[f"{attention_column}_attentions"].reshape(
                    row[f"{attention_column}_attentions_original_shape"]
                ),
                axis=1,
            )

        for hidden_state_column in ["vanilla", "prompt", "input_embeds_replace"]:
            df[f"{hidden_state_column}_hidden_states"] = df.apply(
                lambda row: row[f"{hidden_state_column}_hidden_states"].reshape(
                    row[f"{hidden_state_column}_hidden_states_original_shape"]
                ),
                axis=1,
            )
        return df
