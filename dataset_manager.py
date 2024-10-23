import os
from typing import List, Tuple, Callable, Optional, Dict, Union, Set, Union
from abc import ABC
import ast
import re

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
    row_to_vanilla_datapoint,
    row_to_prompt_datapoint,
    row_to_input_embeds_replace_datapoint,
    find_non_existing_source_targets,
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
            not self._data_present() or force_recompute
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
            self.data = self.__generate_hetero_dataset()
            # split HeteroDataset dataset into train, dev and test
            self.__split_data()
            # split llm dataset according to the HeteroDataset split into train, dev, test and rest
            self.__split_llm_dataset()
            # generate pandas dataframe with prompts and labels for LLM
            # self.generate_llm_dataset()

            self.source_df.to_csv(f"{ROOT}/llm/source.csv", index=False)
            self.target_df.to_csv(f"{ROOT}/llm/target.csv", index=False)
        else:
            self._load_datasets_from_disk()

    def _data_present(self) -> bool:
        """Returns True, if GNN_TRAIN_DATASET_PATH, f"{ROOT}/gnn/test", f"{ROOT}/gnn/val", f"{ROOT}/llm/dataset.csv" are files."""
        return (
            os.path.isfile(f"{ROOT}/gnn/train")
            and os.path.isfile(f"{ROOT}/gnn/test")
            and os.path.isfile(f"{ROOT}/gnn/val")
            and os.path.isfile(f"{ROOT}/llm/dataset.csv")
            and os.path.isfile(f"{ROOT}/llm/source.csv")
            and os.path.isfile(f"{ROOT}/llm/target.csv")
        )

    def _load_datasets_from_disk(self):
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

    def __generate_hetero_dataset(self) -> HeteroData:
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

    def __split_data(self) -> None:
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
        print("appending input embeds for replace model")
        assert len(self.llm_df) == len(graph_embeddings)
        self.llm_df["input_embeds_replace_source_embedding"] = graph_embeddings[
            "source_embedding"
        ]
        self.llm_df["input_embeds_replace_target_embedding"] = graph_embeddings[
            "target_embedding"
        ]
        if save:
            self.save_llm_df()

    def __split_llm_dataset(self) -> None:
        """
        This method assigns all datapoints in the LLM dataset to the same split as they are found in the gnn dataset split.
        """
        train_edge_index: torch.Tensor = self.gnn_train_data[
            "source", "edge", "target"
        ].edge_label_index
        test_edge_index: torch.Tensor = self.gnn_test_data[
            "source", "edge", "target"
        ].edge_label_index
        val_edge_index: torch.Tensor = self.gnn_val_data[
            "source", "edge", "target"
        ].edge_label_index
        print("splitting LLM dataset")
        df_train = pd.DataFrame(
            {
                "source_id": train_edge_index[0].tolist(),
                "target_id": train_edge_index[1].tolist(),
                "split": "train",
                "labels": 1,
            }
        )
        df_test = pd.DataFrame(
            {
                "source_id": test_edge_index[0].tolist(),
                "target_id": test_edge_index[1].tolist(),
                "split": "test",
                "labels": self.gnn_test_data["source", "edge", "target"]
                .edge_label.int()
                .tolist(),
            }
        )
        df_val = pd.DataFrame(
            {
                "source_id": val_edge_index[0].tolist(),
                "target_id": val_edge_index[1].tolist(),
                "split": "val",
                "labels": self.gnn_val_data["source", "edge", "target"]
                .edge_label.int()
                .tolist(),
            }
        )
        combined_df = pd.concat([df_train, df_test, df_val])
        combined_df = combined_df.drop_duplicates(
            subset=["source_id", "target_id"], keep="first"
        )
        duplicated_df = (
            combined_df[combined_df.duplicated(subset=["source_id", "target_id"])]
            .sample(frac=1)
            .drop_duplicates(subset=["source_id", "target_id"], keep="first")
        )
        split_df = pd.concat([combined_df, duplicated_df])
        split_df = self.llm_df.merge(
            split_df[["source_id", "target_id", "split", "labels"]],
            on=["source_id", "target_id"],
        )
        self.llm_df = split_df
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
        splits: List[str] = ["train", "test", "val"],
        add_embeddings: bool = True,
        force_recompute: bool = False,
    ) -> Union[DatasetDict, Dataset]:
        """
        Generates the dataset for training the adding model,
        by passing the tokenizer.tokenize function and
        the embedding dimension of the target adding model.
        """
        llm_adding_dataset_path = f"{ROOT}/llm/input_embeds_replace/dataset{suffix}"
        if os.path.exists(llm_adding_dataset_path) and not force_recompute:
            dataset = datasets.load_from_disk(llm_adding_dataset_path)
        else:
            if isinstance(df, pd.DataFrame):
                llm_df = df.copy(deep=True)
            else:
                llm_df = self.llm_df.copy(deep=True)
            llm_df = llm_df[llm_df["split"] in splits]
            llm_df["prompt"] = llm_df.apply(
                lambda row: row_to_input_embeds_replace_datapoint(
                    row, sep_token, pad_token
                ),
                axis=1,
            )
            if add_embeddings:
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
        splits: List[str] = ["train", "test", "val"],
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
            llm_df = llm_df[llm_df["split"] in splits]
            llm_df["prompt"] = llm_df.apply(
                lambda row: row_to_vanilla_datapoint(row, sep_token=sep_token),
                axis=1,
            )
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

    def __flatten_and_rename_if_present(
        self, df: pd.DataFrame, prefix: str, add_tokens: bool = False
    ):
        df = df.copy(deep=True)
        non_token_columns = {"source_id", "target_id", "split"}

        for field in ["attentions", "hidden_states"]:
            if field in df.columns:
                df[f"{field}_original_shape"] = df[field].apply(
                    lambda field: field.shape
                )
                df[field] = df[field].apply(lambda field: field.flatten())
        if "hidden_states" in df.columns:
            df = df.rename(
                columns={
                    "logits": f"{prefix}_logits",
                    "predictions": f"{prefix}_predictions",
                }
            )
            non_token_columns = non_token_columns.union(
                {f"{prefix}_logits", f"{prefix}_predictions"}
            )
        for field in ["attentions", "hidden_states"]:
            if field in df.columns:
                new_column_name = f"{prefix}_{field}"
                new_column_shape_name = f"{prefix}_{field}_original_shape"
                df = df.rename(
                    columns={
                        field: new_column_name,
                        f"{field}_original_shape": new_column_shape_name,
                    }
                )
                non_token_columns.add(new_column_name)
                non_token_columns.add(new_column_shape_name)
        if add_tokens:
            columns_to_return = list(df.columns)
        else:
            columns_to_return = list(non_token_columns)
        return df[columns_to_return]

    def generate_huggingface_dataset(
        self, dfs: List[pd.DataFrame], df_prefix_list: List[str], add_tokens=False
    ) -> DatasetDict:
        assert len(dfs) > 1
        assert len(dfs) == len(df_prefix_list)
        df_0 = dfs[0]
        dfs = dfs[1:]
        df_prefix_0 = df_prefix_list[0]
        df_prefix_list = df_prefix_list[1:]
        df_columns = set(df_0.columns)
        len_df = len(df_0)
        for df in dfs:
            assert len(df) == len_df
            for field in ["attentions", "hidden_states", "logits"]:
                if field in df_columns:
                    assert field in df.columns

        df_0 = self.__flatten_and_rename_if_present(
            df_0.copy(deep=True), df_prefix_0, add_tokens=add_tokens
        )
        for df, df_prefix in zip(dfs, df_prefix_list):
            df_0 = df_0.merge(
                self.__flatten_and_rename_if_present(df, df_prefix, add_tokens=False),
                on=["source_id", "target_id", "split"],
            )
        if not add_tokens:
            df_0 = df_0.drop(columns=["source_id", "target_id"])
        dataset = self.__dataset_from_df(df_0)
        return dataset

    def __dataset_from_df(self, df: pd.DataFrame) -> DatasetDict:
        """
        Generates the LLM datasets from the pandas dataframe.
        """
        dataset_dict = {}
        if "train" in df["split"].unique():
            dataset_dict["train"] = Dataset.from_pandas(
                df[df["split"] == "train"], preserve_index=False
            )
        if "val" in df["split"].unique():
            dataset_dict["val"] = Dataset.from_pandas(
                df[df["split"] == "val"], preserve_index=False
            )
        if "test" in df["split"].unique():
            dataset_dict["test"] = Dataset.from_pandas(
                df[df["split"] == "test"], preserve_index=False
            )
        if len(dataset_dict) == 0:
            raise Exception("not enough splits in datasets found.")
        return DatasetDict(dataset_dict)

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

    def add_negative_edges_to_train_data(
        self, false_ratio: float = 1.0, force_recompute: bool = False
    ) -> None:
        train_split = self.llm_df[self.llm_df["split"] == "train"]
        amount_false_edges = int(len(train_split) * false_ratio)
        if force_recompute or not (train_split["labels"] == 0).any():
            positive_source_ids = list(train_split["source_id"].unique())
            positive_target_ids = list(train_split["target_id"].unique())
            positive_edges = set(zip(self.llm_df.source_id, self.llm_df.target_id))
            node_pairs = find_non_existing_source_targets(
                positive_edges,
                positive_source_ids,
                positive_target_ids,
                k=amount_false_edges,
            )
            df = pd.DataFrame(
                {
                    "source_id": node_pairs[:, 0],
                    "target_id": node_pairs[:, 1],
                    "labels": 0,
                    "split": "train",
                }
            )
            df = (
                df.merge(self.source_df, left_on="source_id", right_on="id")
                .reset_index(drop=True)
                .merge(self.target_df, left_on="target_id", right_on="id")
                .reset_index(drop=True)
            )
            self.llm_df = pd.concat([self.llm_df, df])
            self.save_llm_df()


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
        url = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
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
    ) -> pd.DataFrame:
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
            df = dataset[split].to_pandas()
            dfs.append(df)
        df = pd.concat(dfs)
        regex = r"(vanilla|prompt|input_embeds_replace|input_embeds_replace_frozen)_attentions_original_shape"
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
        dfs = []
        for split in ["train", "test", "val"]:
            df = dataset[split].to_pandas()
            dfs.append(df)
        df = pd.concat(dfs)
        regex = r"(vanilla|prompt|input_embeds_replace|input_embeds_replace_frozen)_attentions_original_shape"
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
