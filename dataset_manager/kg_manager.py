import os
from typing import List, Callable, Optional, Union, Tuple
from abc import ABC
import random
from os import listdir
from os.path import isfile, join
from pathlib import Path
import ast


import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import pandas as pd
import datasets
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import PreTrainedTokenizer

from utils import find_non_existing_source_targets
from llm_manager.vanilla.utils import row_to_vanilla_datapoint
from llm_manager.graph_prompter_hf.utils import row_to_graph_prompter_hf_datapoint


ROOT = "./data"  # The root path where models and datasets are saved at.


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
            "llm/graph_prompter_hf",
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
            disjoint_train_ratio=0.7,
            neg_sampling_ratio=1,
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

    def append_graph_prompter_hf_graph_embeddings(
        self, graph_embeddings: pd.DataFrame, save: bool = True
    ):
        print("appending input embeds for replace model")
        assert len(self.llm_df) == len(graph_embeddings)
        self.llm_df["graph_prompter_hf_source_embedding"] = graph_embeddings[
            "source_embedding"
        ]
        self.llm_df["graph_prompter_hf_target_embedding"] = graph_embeddings[
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
        self.llm_df = combined_df.merge(
            self.source_df, left_on="source_id", right_on="id"
        ).merge(self.target_df, left_on="target_id", right_on="id")
        feature_columns = list(
            filter(
                lambda column: column.startswith("gnn_feature_"),
                self.llm_df.columns,
            )
        )
        self.llm_df = self.llm_df.drop(columns=feature_columns + ["id_x", "id_y"])
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

    def generate_graph_prompter_hf_embedding_dataset(
        self,
        sep_token,
        pad_token,
        tokenize_function: Optional[Callable] = None,
        suffix="",
        df: Optional[pd.DataFrame] = None,
        splits: List[str] = ["train", "test", "val"],
        get_embeddings_cb: Optional[Callable] = None,
        force_recompute: bool = False,
    ) -> Union[DatasetDict, Dataset]:
        """
        Generates the dataset for training the adding model,
        by passing the tokenizer.tokenize function and
        the embedding dimension of the target adding model.
        """
        llm_adding_dataset_path = f"{ROOT}/llm/graph_prompter_hf/dataset{suffix}"
        llm_adding_dataset_temp_path = (
            f"{ROOT}/llm/graph_prompter_hf/dataset{suffix}_temp"
        )
        if os.path.exists(llm_adding_dataset_path) and not force_recompute:
            dataset = datasets.load_from_disk(llm_adding_dataset_path)
        else:
            if not os.path.exists(llm_adding_dataset_temp_path) or force_recompute:
                if isinstance(df, pd.DataFrame):
                    llm_df = df.copy(deep=True)
                else:
                    llm_df = self.llm_df.copy(deep=True)
                llm_df = llm_df[llm_df["split"].isin(splits)]
                llm_df["prompt"] = llm_df.apply(
                    lambda row: row_to_graph_prompter_hf_datapoint(
                        row, sep_token, pad_token
                    ),
                    axis=1,
                )
                if get_embeddings_cb:
                    chunk_size = 20000
                    print("chunking for KGEs")
                    chunks = []
                    for i in range(0, len(llm_df), chunk_size):
                        chunk = llm_df.iloc[i : i + chunk_size].copy()  # Create chunk
                        source_kges, target_kges = get_embeddings_cb(
                            self.data,
                            chunk["source_id"].to_list(),
                            chunk["target_id"].to_list(),
                        )
                        chunk["source_kges"] = source_kges.to("cpu").detach().tolist()
                        chunk["target_kges"] = target_kges.to("cpu").detach().tolist()
                        chunks.append(chunk)
                    llm_df = pd.concat(chunks)
                    print(llm_df)
                dataset = self.__dataset_from_df(llm_df)
                dataset.save_to_disk(llm_adding_dataset_temp_path)
            dataset = load_from_disk(llm_adding_dataset_temp_path)
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
        filepath = f"{ROOT}/llm/vanilla/dataset{suffix}"
        temp_filepath = f"{ROOT}/llm/vanilla/dataset{suffix}_temp"
        if os.path.exists(filepath) and not force_recompute:
            dataset = datasets.load_from_disk(filepath)
        else:
            if not os.path.exists(temp_filepath) or force_recompute:
                print("dataset does not exist.")
                if isinstance(df, pd.DataFrame):
                    llm_df = df.copy(deep=True)
                else:
                    llm_df = self.llm_df.copy(deep=True)
                llm_df = llm_df[llm_df["split"].isin(splits)]
                llm_df["prompt"] = llm_df.apply(
                    lambda row: row_to_vanilla_datapoint(row, sep_token=sep_token),
                    axis=1,
                )
                dataset = self.__dataset_from_df(llm_df)
                dataset.save_to_disk(temp_filepath)
            dataset = load_from_disk(temp_filepath)

            if tokenize_function:
                dataset = dataset.map(tokenize_function, batched=True)
            dataset.save_to_disk(filepath)
        return dataset

    def shard_dataset_randomly(
        self,
        vanilla_root: str,
        graph_prompter_hf_root: str,
        graph_prompter_hf_frozen_root: str,
        vanilla_dataset: Optional[DatasetDict] = None,
        graph_prompter_hf_dataset: Optional[DatasetDict] = None,
        shard_size: int = 100000,
        splits: List[str] = ["test", "val"],
        force_recompute: bool = False,
    ) -> Tuple[DatasetDict, DatasetDict]:
        vanilla_path = vanilla_root + "/dataset_shard_{}".format(shard_size)
        graph_prompter_hf_path = graph_prompter_hf_root + "/dataset_shard_{}".format(
            shard_size
        )
        graph_prompter_hf_frozen_path = (
            graph_prompter_hf_frozen_root + "/dataset_shard_{}".format(shard_size)
        )

        if force_recompute or not (
            os.path.exists(vanilla_path) and os.path.exists(graph_prompter_hf_path)
        ):
            seed = random.randint(0, 255)
            s_vanilla = vanilla_dataset.shuffle(seed=seed)
            vanilla_dict = {}
            s_graph_prompter_hf = graph_prompter_hf_dataset.shuffle(seed=seed)
            graph_prompter_hf_dict = {}

            for split in splits:
                vanilla_dict[split] = s_vanilla[split].shard(
                    int(len(s_vanilla[split]) / shard_size - 1), 0
                )
                graph_prompter_hf_dict[split] = s_graph_prompter_hf[split].shard(
                    int(len(s_vanilla[split]) / shard_size - 1), 0
                )

            vanilla_dataset = DatasetDict(vanilla_dict)
            graph_prompter_hf_dataset = DatasetDict(graph_prompter_hf_dict)
            vanilla_dataset.save_to_disk(vanilla_path)
            graph_prompter_hf_dataset.save_to_disk(graph_prompter_hf_frozen_path)
            graph_prompter_hf_dataset.save_to_disk(graph_prompter_hf_path)
        else:
            vanilla_dataset = load_from_disk(vanilla_path)
            graph_prompter_hf_dataset = load_from_disk(graph_prompter_hf_path)

        return (
            vanilla_dataset,
            graph_prompter_hf_dataset,
        )

    def fuse_xai_shards(self, root: str, dataset: DatasetDict) -> None:
        def split_path(
            filename: str, basepath: str
        ) -> Tuple[str, str, float, Tuple[int]]:
            file_name_splits = filename.split("_")
            return (
                join(basepath, filename),
                file_name_splits[1],
                float(file_name_splits[3]),
                tuple(ast.literal_eval(file_name_splits[5].split(".")[0])),
            )

        def generate_df_for_splits(
            basepath: str,
            attention_map_files: List[Tuple[str, str, float, Tuple[int]]],
            hidden_state_files: List[Tuple[str, str, float, Tuple[int]]],
            logit_files: List[Tuple[str, str, Tuple[int]]],
        ):
            xai_artifacts_dir = f"{basepath}/xai_artifacts"
            Path(xai_artifacts_dir).mkdir(parents=True, exist_ok=True)
            file_dict = {}
            for attention_map_file, hidden_state_file in zip(
                attention_map_files, hidden_state_files
            ):
                if attention_map_file[1] not in file_dict:  # split
                    file_dict[attention_map_file[1]] = {}
                if hidden_state_file[1] not in file_dict:  # split
                    file_dict[hidden_state_file[1]] = {}

                if (
                    attention_map_file[3] not in file_dict[attention_map_file[1]]
                ):  # mask
                    file_dict[attention_map_file[1]][attention_map_file[3]] = {
                        "attention_maps": [],
                        "hidden_states": [],
                    }
                if hidden_state_file[3] not in file_dict[hidden_state_file[1]]:  # mask
                    file_dict[hidden_state_file[1]][hidden_state_file[3]] = {
                        "attention_maps": [],
                        "hidden_states": [],
                    }

                file_dict[attention_map_file[1]][attention_map_file[3]][
                    "attention_maps"
                ].append((attention_map_file[0], attention_map_file[2]))

                file_dict[hidden_state_file[1]][hidden_state_file[3]][
                    "hidden_states"
                ].append((hidden_state_file[0], hidden_state_file[2]))

            for logit_file in logit_files:
                file_dict[logit_file[1]][logit_file[2]]["logits"] = logit_file[0]

            for split, split_dict in file_dict.items():
                labels = dataset[split]["labels"]
                source_ids = dataset[split]["source_id"]
                target_ids = dataset[split]["target_id"]
                for mask, mask_dict in split_dict.items():
                    mask_dict["attention_maps"] = list(
                        map(
                            lambda tup: tup[0],
                            sorted(mask_dict["attention_maps"], key=lambda tup: tup[1]),
                        )
                    )
                    mask_dict["hidden_states"] = list(
                        map(
                            lambda tup: tup[0],
                            sorted(mask_dict["hidden_states"], key=lambda tup: tup[1]),
                        )
                    )
                    attention_maps = np.concatenate(
                        [np.load(filepath) for filepath in mask_dict["attention_maps"]]
                    )
                    hidden_states = np.concatenate(
                        [np.load(filepath) for filepath in mask_dict["hidden_states"]]
                    )
                    logits = np.load(mask_dict["logits"])

                    print("save to ", f"{basepath}/xai_artifacts/{split}_{mask}.csv")
                    pd.DataFrame(
                        {
                            "attention_maps": attention_maps.tolist(),
                            "hidden_states": hidden_states.tolist(),
                            "logits": logits.tolist(),
                            "labels": labels,
                            "source_ids": source_ids,
                            "target_ids": target_ids,
                        }
                    ).to_csv(f"{xai_artifacts_dir}/{split}_{mask}.csv", index=False)

        attention_paths = f"{root}/attentions"
        hidden_state_paths = f"{root}/hidden_states"
        logits_path = f"{root}/logits"

        attention_files = [
            split_path(f, attention_paths)
            for f in listdir(attention_paths)
            if isfile(join(attention_paths, f))
        ]

        hidden_state_files = [
            split_path(f, hidden_state_paths)
            for f in listdir(hidden_state_paths)
            if isfile(join(hidden_state_paths, f))
        ]

        logit_files = [
            (
                join(logits_path, f),
                f.split("_")[1],
                tuple(ast.literal_eval(f.split("_")[3].split(".")[0])),
            )
            for f in listdir(logits_path)
            if isfile(join(logits_path, f))
        ]

        generate_df_for_splits(root, attention_files, hidden_state_files, logit_files)

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
        all_token_type_ranges: torch.Tensor,
    ) -> None:
        ends = all_token_type_ranges[:, :, 1]
        starts = all_token_type_ranges[:, :, 0]
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
