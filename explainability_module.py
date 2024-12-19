import json
from typing import List, Tuple, Optional, Callable, Dict
import random as rd
from pathlib import Path
import itertools
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from matplotlib.collections import PathCollection
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from datasets import DatasetDict
from matplotlib.collections import LineCollection

from dataset_manager.kg_manager import ROOT
from llm_manager.vanilla.config import (
    VANILLA_TOKEN_DICT,
    VANILLA_TOKEN_TYPE_VALUES,
)
from llm_manager.graph_prompter_hf.config import (
    GRAPH_PROMPTER_TOKEN_DICT,
    GRAPH_PROMPTER_TOKEN_TYPE_VALUES,
)
from utils import get_combinations


class ExplainabilityModule:
    def __init__(self, load_xai_artifacts_cb: Callable, dataset: DatasetDict) -> None:
        self.llm_df = pd.read_csv(f"{ROOT}/llm/dataset.csv")
        self.vanilla_path = f"{ROOT}/llm/vanilla"
        self.graph_prompter_hf_path = f"{ROOT}/llm/graph_prompter_hf"
        self.graph_prompter_hf_frozen_path = f"{ROOT}/llm/graph_prompter_hf_frozen"
        self.vanilla_training_path = (
            f"{self.vanilla_path}/training/checkpoint-140002/trainer_state.json"
        )
        self.graph_prompter_hf_training_path = f"{self.graph_prompter_hf_path}/training/checkpoint-140002/trainer_state.json"
        self.graph_prompter_hf_frozen_training_path = f"{self.graph_prompter_hf_frozen_path}/training/checkpoint-140002/trainer_state.json"
        self.load_xai_artifacts_cb: Callable = load_xai_artifacts_cb
        self.dataset: DatasetDict = dataset

    def load_df(
        self,
        split: str,
        model: str,
        token_type_mask: List[int],
        filter: Optional[Callable] = None,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        if model == "vanilla":
            xai_root = self.vanilla_path
            token_type_dict = VANILLA_TOKEN_DICT
        elif model == "graph_prompter_hf_frozen":
            xai_root = self.graph_prompter_hf_frozen_path
            token_type_dict = GRAPH_PROMPTER_TOKEN_DICT
        else:
            xai_root = self.graph_prompter_hf_path
            token_type_dict = GRAPH_PROMPTER_TOKEN_DICT
        token_type_dict_reverse = {v: k for k, v in token_type_dict.items()}
        df = self.load_xai_artifacts_cb(
            xai_root, split, token_type_mask, self.dataset, filter
        )
        df["model"] = model
        return df, token_type_dict, token_type_dict_reverse

    def plot_training_losses(self, save_plot: bool = True):
        with open(self.vanilla_training_path, "r") as f:
            vanilla_training_process = json.load(f)
        with open(self.graph_prompter_hf_frozen_training_path, "r") as f:
            graph_prompter_hf_frozen_training_process = json.load(f)
        with open(self.graph_prompter_hf_training_path, "r") as f:
            graph_prompter_hf_training_process = json.load(f)
        # Extract the loss values
        vanilla_losses = [
            entry["loss"]
            for entry in vanilla_training_process["log_history"]
            if "loss" in entry
        ]
        min_vanilla_losses = min(vanilla_losses)

        graph_prompter_frozen_losses_ = [
            entry["loss"]
            for entry in graph_prompter_hf_frozen_training_process["log_history"]
            if "loss" in entry
        ]
        min_graph_prompter_frozen_losses = min(graph_prompter_frozen_losses_)
        graph_prompter_frozen_losses = [vanilla_losses[-1]]
        graph_prompter_frozen_losses.extend(graph_prompter_frozen_losses_)

        graph_prompter_losses_ = [
            entry["loss"]
            for entry in graph_prompter_hf_training_process["log_history"]
            if "loss" in entry
        ]
        min_graph_prompter_losses = min(graph_prompter_frozen_losses_)
        graph_prompter_losses = [graph_prompter_frozen_losses[-1]]
        graph_prompter_losses.extend(graph_prompter_losses_)

        print(f"Min Vanilla model loss is {min_vanilla_losses}.")
        print(
            f"Min GraphPrompterHF frozen model loss is {min_graph_prompter_frozen_losses}."
        )
        print(f"Min GraphPrompterHF model loss is {min_graph_prompter_losses}.")
        # Extract steps
        vanilla_steps = [
            entry["step"]
            for entry in vanilla_training_process["log_history"]
            if "step" in entry and "loss" in entry
        ]

        graph_prompter_frozen_steps = [vanilla_steps[-1]]
        graph_prompter_frozen_steps.extend(
            [
                entry["step"]
                + vanilla_training_process[
                    "global_step"
                ]  # add last step, so we can make a continues loss curve
                for entry in graph_prompter_hf_frozen_training_process["log_history"]
                if "step" in entry and "loss" in entry
            ]
        )
        graph_prompter_steps = [graph_prompter_frozen_steps[-1]]
        graph_prompter_steps.extend(
            [
                entry["step"]
                + vanilla_training_process["global_step"]
                + graph_prompter_hf_frozen_training_process[
                    "global_step"
                ]  # add last step, so we can make a continues loss curve
                for entry in graph_prompter_hf_training_process["log_history"]
                if "step" in entry and "loss" in entry
            ]
        )
        vanilla_label = "Vanilla Model"
        graph_prompter_frozen_label = "GraphPrompterHF Frozen Model"
        graph_prompter_label = "GraphPrompterHF Model"

        def plot(losses: List[float], steps: list[float], label: str):
            # Plot the loss curve
            plt.plot(steps, losses, label=label)

        plot(vanilla_losses, vanilla_steps, vanilla_label)
        plot(
            graph_prompter_frozen_losses,
            graph_prompter_frozen_steps,
            graph_prompter_frozen_label,
        )
        plot(graph_prompter_losses, graph_prompter_steps, graph_prompter_label)
        # Add title and labels
        plt.title("Losses of Training Process")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")

        # Add legend
        plt.legend()
        if save_plot:
            plt.savefig("./images/training_losses.png")

        # Show plot
        plt.show()

    def plot_training_accuracies(self, save_plot: bool = True):
        with open(self.vanilla_training_path, "r") as f:
            vanilla_training_process = json.load(f)
        with open(self.graph_prompter_hf_frozen_training_path, "r") as f:
            graph_prompter_hf_frozen_training_process = json.load(f)
        with open(self.graph_prompter_hf_training_path, "r") as f:
            graph_prompter_hf_training_process = json.load(f)
        gnn_frozen_accuracy = np.load("./data/gnn/gnn_frozen_val.npy").item()
        gnn_accuracy = np.load("./data/gnn/gnn_val.npy").item()
        # Extract the loss values
        vanilla_eval_accuracy = [
            entry["eval_accuracy"]
            for entry in vanilla_training_process["log_history"]
            if "eval_accuracy" in entry
        ]
        max_vanilla_eval_accuracy = max(vanilla_eval_accuracy)

        graph_prompter_frozen_eval_accuracy_ = [
            entry["eval_accuracy"]
            for entry in graph_prompter_hf_frozen_training_process["log_history"]
            if "eval_accuracy" in entry
        ]
        max_graph_prompter_frozen_eval_accuracy = max(
            graph_prompter_frozen_eval_accuracy_
        )
        graph_prompter_frozen_eval_accuracy = [vanilla_eval_accuracy[-1]]
        graph_prompter_frozen_eval_accuracy.extend(graph_prompter_frozen_eval_accuracy_)

        graph_prompter_eval_accuracy_ = [
            entry["eval_accuracy"]
            for entry in graph_prompter_hf_training_process["log_history"]
            if "eval_accuracy" in entry
        ]
        max_graph_prompter_eval_accuracy = max(graph_prompter_eval_accuracy_)
        graph_prompter_eval_accuracy = [graph_prompter_frozen_eval_accuracy[-1]]
        graph_prompter_eval_accuracy.extend(graph_prompter_eval_accuracy_)
        print(f"Max Vanilla model accuracy is {max_vanilla_eval_accuracy}.")
        print(
            f"Max GraphPrompterHF frozen model accuracy is {max_graph_prompter_frozen_eval_accuracy}."
        )
        print(
            f"Max GraphPrompterHF model accuracy is {max_graph_prompter_eval_accuracy}."
        )
        print(f"GNN frozen eval accuracy is {gnn_frozen_accuracy}.")
        print(f"GNN eval accuracy is {gnn_accuracy}.")

        # Extract epochs
        vanilla_epochs = [
            entry["epoch"]
            for entry in vanilla_training_process["log_history"]
            if "epoch" in entry and "eval_accuracy" in entry
        ]
        graph_prompter_frozen_epochs = [vanilla_epochs[-1]]
        graph_prompter_frozen_epochs.extend(
            [
                entry["epoch"]
                + vanilla_training_process[
                    "epoch"
                ]  # add last epoch, so we can make a continues eval_accuracy curve
                for entry in graph_prompter_hf_frozen_training_process["log_history"]
                if "epoch" in entry and "eval_accuracy" in entry
            ]
        )
        graph_prompter_epochs = [graph_prompter_frozen_epochs[-1]]
        graph_prompter_epochs.extend(
            [
                entry["epoch"]
                + vanilla_training_process["epoch"]
                + graph_prompter_hf_frozen_training_process[
                    "epoch"
                ]  # add last epoch, so we can make a continues eval_accuracy curve
                for entry in graph_prompter_hf_training_process["log_history"]
                if "epoch" in entry and "eval_accuracy" in entry
            ]
        )
        vanilla_label = "Vanilla Model"
        graph_prompter_frozen_label = "GraphPrompterHF Frozen Model"
        graph_prompter_label = "GraphPrompterHF Model"

        def plot(
            accuracies: List[float],
            epochs: list[float],
            label: str,
        ):
            # Plot the loss curve
            plt.plot(epochs, accuracies, label=label)

        plot(vanilla_eval_accuracy, vanilla_epochs, vanilla_label)
        plot(
            graph_prompter_frozen_eval_accuracy,
            graph_prompter_frozen_epochs,
            graph_prompter_frozen_label,
        )
        plot(graph_prompter_eval_accuracy, graph_prompter_epochs, graph_prompter_label)
        len_pre_end_to_end_training = (
            len(vanilla_eval_accuracy)
            + len(graph_prompter_frozen_eval_accuracy_)
            + len(graph_prompter_eval_accuracy_)
        )
        gnn_accuracies = [gnn_frozen_accuracy] * (len_pre_end_to_end_training - 2)
        gnn_accuracies.append(gnn_accuracy)
        gnn_epochs = list(range(1, len_pre_end_to_end_training - 1))
        gnn_epochs = [float(epoch) for epoch in gnn_epochs]
        gnn_epochs.append(len_pre_end_to_end_training)
        print(gnn_epochs)
        print(gnn_accuracies)
        plot(
            gnn_accuracies,
            gnn_epochs,
            "GNN eval accuracy",
        )
        # Add title and labels
        plt.title(
            "Evaluation Accuracy Curves of Vanilla and GraphPrompterHF (Frozen) Models"
        )
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")

        # Add legend
        plt.legend(loc=4)
        if save_plot:
            plt.savefig("./images/training_accuracies.png")

        # Show plot
        plt.show()

    def shap_accuracies(self, save_plot: bool = True, force_recompute: bool = False):
        df = None
        accuracies = {}
        logits_path = "./data/llm/logits.pkl"
        accuracies_path = "./data/llm/accuracies.json"
        if (
            force_recompute
            or not os.path.exists(accuracies_path)
            or not os.path.exists(logits_path)
        ):
            print(
                "rearrange logits",
                os.path.exists(logits_path),
                os.path.exists(accuracies_path),
            )
            for model in ["vanilla", "graph_prompter_hf_frozen", "graph_prompter_hf"]:
                if model == "vanilla":
                    token_type_ids = list(set(VANILLA_TOKEN_TYPE_VALUES))
                else:
                    token_type_ids = list(set(GRAPH_PROMPTER_TOKEN_TYPE_VALUES))
                for token_type_mask in get_combinations(token_type_ids):
                    df_model, token_type_dict, token_type_dict_reverse = self.load_df(
                        "val", model, list(token_type_mask)
                    )
                    predictions = np.argmax(
                        np.stack(df_model["logits"].tolist()),
                        axis=-1,
                    )
                    accuracies[f"{model}_{list(token_type_mask)}"] = accuracy_score(
                        predictions, np.stack(df_model["labels"].tolist())
                    )
                    logits_column = f"logits_{model}_{list(token_type_mask)}"
                    df_model = df_model.rename(columns={"logits": logits_column})
                    if df is None:
                        df = df_model
                    else:
                        df = df.merge(
                            df_model[
                                [
                                    "source_id",
                                    "target_id",
                                    logits_column,
                                ]
                            ],
                            on=["source_id", "target_id"],
                            suffixes=("", f"_{model}_{list(token_type_mask)}"),
                        )

            assert isinstance(df, pd.DataFrame)
            df.to_pickle(logits_path)
            print(accuracies)
            with open(accuracies_path, "w") as f:
                json.dump(accuracies, f)
        else:
            df = pd.read_pickle(logits_path)

        print(list(filter(lambda column: "vanilla" in column, list(df.columns))))
        for model in ["vanilla", "graph_prompter_hf", "graph_prompter_hf_frozen"]:
            if model == "vanilla":
                token_type_values = list(set(VANILLA_TOKEN_TYPE_VALUES))
                token_dict = VANILLA_TOKEN_DICT
            else:
                # we are a graph prompter model
                token_type_values = list(set(GRAPH_PROMPTER_TOKEN_TYPE_VALUES))
                token_dict = GRAPH_PROMPTER_TOKEN_DICT
            token_type_combinations = get_combinations(token_type_values)[1:]
            # Initialize SHAP value arrays for both classes
            shap_values: Dict[str, List[float]] = {}  # 2 classes, num_features
            len_N = len(token_type_values)
            for token_type in token_type_values:
                shap_values[token_dict[token_type]] = [0.0, 0.0]
                N_without_i = list(
                    filter(
                        lambda combination: token_type in combination,
                        token_type_combinations,
                    )
                )
                for S in N_without_i:
                    len_S = len_N - len(S)
                    weight = (
                        np.math.factorial(len_S)
                        * np.math.factorial((len_N - 1 - len_S))
                    ) / np.math.factorial(len_N)
                    S_with_i = list(S.difference(set([token_type])))

                    S_with_i_logits = np.mean(
                        np.stack(df[f"logits_{model}_{S_with_i}"].tolist()),
                        axis=0,
                    )
                    S_logits = np.mean(
                        np.stack(df[f"logits_{model}_{list(S)}"].tolist()),
                        axis=0,
                    )
                    shap_value = (weight * (S_with_i_logits - S_logits)).tolist()
                    shap_values[token_dict[token_type]][0] += shap_value[0]
                    shap_values[token_dict[token_type]][1] += shap_value[1]
                print(token_type, shap_values[token_dict[token_type]])
            with open(f"./data/llm/{model}/shap_values.json", "w") as f:
                json.dump(shap_values, f)

    def plot_shap_values(
        self,
        bar_width=0.4,
        fig_size=(16, 8),
        fig_dpi: int = 100,
        save_plot: bool = True,
    ) -> None:
        plt.rcParams["figure.figsize"] = fig_size
        plt.figure(figsize=fig_size)
        for idx, model in enumerate(
            ["vanilla", "graph_prompter_hf_frozen", "graph_prompter_hf"]
        ):
            x_axis = []
            x_labels = []
            y_axis_1 = []
            y_axis_2 = []
            with open(f"./data/llm/{model}/shap_values.json", "r") as f:
                shap_values = json.load(f)
                for feature, shap_value in shap_values.items():
                    x_axis.append(len(x_axis) * 3)
                    x_labels.append(f"- {feature}")
                    y_axis_1.append(shap_value[0])
                    y_axis_2.append(shap_value[1])
            plt.bar(
                (np.array(x_axis) - bar_width / 2) + (idx - 1) * bar_width * 2,
                y_axis_1,
                align="center",
                width=bar_width,
                label=f"negative {model}",
            )
            plt.bar(
                (np.array(x_axis) + bar_width / 2) + (idx - 1) * bar_width * 2,
                y_axis_2,
                align="center",
                width=bar_width,
                label=f"positive {model}",
            )
            plt.xticks(x_axis, x_labels)
        plt.xlabel("Features")
        plt.ylabel("SHAP Values")
        # Add a single legend
        plt.legend(
            loc="upper left",
            ncol=2,  # Optional: Place legends in two columns
            title="Legend",
        )
        if save_plot:
            plt.savefig("./images/shap_values.png")

    def plot_confusion_map(
        self,
        split: str,
        mask: List[int],
        model: str = "vanilla",
        filter: Optional[Callable] = None,
        save_plot: bool = True,
    ):
        df, _, _ = self.load_df(split, model, mask, filter)
        logits = np.stack(df["logits"].tolist())
        # Get predicted labels and true labels
        preds = np.argmax(logits, axis=-1)
        labels = df["labels"].tolist()
        # Compute confusion matrix
        cm = confusion_matrix(labels, preds)  # type: ignore

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Negative", "Positive"]
        )
        disp.plot(cmap=plt.cm.Blues)  # type: ignore
        if save_plot:
            plt.savefig(f"./images/confusion_map_{model}_{split}_{mask}.png")
        plt.show()

    def plot_attention_maps(
        self,
        split: str,
        mask: List[int],
        model: str = "vanilla",
        filter: Optional[Callable] = None,
        weight_coef: int | float = 15,
        fig_dpi: int = 100,
        fig_size: Tuple[int, int] = (8, 8),
        save_plot: bool = True,
    ) -> None:
        assert fig_dpi > 0
        assert fig_size[0] > 0
        assert fig_size[1] > 0
        assert weight_coef > 0
        df, token_type_dict, token_type_dict_reverse = self.load_df(
            split, model, mask, filter
        )
        attention_maps = np.stack(df["attention_map"].tolist())
        plt.rcParams["figure.figsize"] = fig_size
        plt.rcParams["figure.dpi"] = fig_dpi  # 200 e.g. is really fine, but slower
        plt.figure(figsize=fig_size, dpi=fig_dpi)
        # Create an undirected graph
        G = nx.Graph()
        labels = {}
        attentions = np.mean(attention_maps, axis=0).transpose(2, 0, 1)
        for layer, attentions_ in enumerate(attentions):
            for from_, inner in enumerate(attentions_):
                from_name = f"{token_type_dict[from_]}_{layer}"
                if layer == 0:
                    labels[from_name] = token_type_dict[from_]
                G.add_node(from_name, name=from_name, layer=layer)
                for to_, weight in enumerate(inner):
                    to_name = f"{token_type_dict[to_]}_{layer+1}"
                    G.add_node(to_name, name=to_name, layer=layer + 1)
                    G.add_edge(from_name, to_name, weight=weight)

        pos = nx.multipartite_layout(G, subset_key="layer")
        for node, (x, y) in pos.items():
            y = len(token_type_dict) - token_type_dict_reverse[node.split("_")[0]]
            layer = G.nodes[node]["layer"]
            pos[node] = (layer, y)  # type: ignore # Fixing the x-coordinate to be the layer and y-coordinate can be customized

        nx.draw(G, pos=pos, with_labels=False)

        edge_weights = nx.get_edge_attributes(G, "weight")

        # Create a list of edge thicknesses based on weights
        # Normalize the weights to get thickness values
        max_weight = max(edge_weights.values())
        edge_thickness = [
            edge_weights[edge] / max_weight * weight_coef for edge in G.edges()
        ]  # Draw edges with varying thicknesses
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=edge_thickness)  # type: ignore
        shift_to_left = 0.4
        label_pos = {node: (x - shift_to_left, y) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(G, label_pos, labels)
        # Get the current axis limits
        x_min, x_max = plt.xlim()

        # Adjust x-axis limits to add more whitespace on the left
        plt.xlim(x_min - (shift_to_left + 0.1), x_max)  # Increase the left limit by 0.1
        plt.title(
            f"Attention map of model {model}, split {split} and mask(s) {mask if len(mask) > 0 else 'None'}"
        )
        if save_plot:
            plt.savefig(f"./images/attention_map_{model}_{split}_{mask}.png")
        plt.show()

    def get_verbose_df(self, n: Optional[int] = None) -> pd.DataFrame:
        def make_numpy_verbose(row: pd.Series) -> pd.Series:
            for column in row.keys():
                if isinstance(row[column], np.ndarray):
                    row[column] = f"{row[column].dtype}: {row[column].shape}"
            return row

        df = self.llm_df
        if n:
            assert n > 0
            df = self.llm_df.head(n)
        return df.apply(make_numpy_verbose, axis=1)

    def fit_pca(self, embeddings: np.ndarray) -> PCA:
        pca = PCA(n_components=2)
        pca.fit_transform(embeddings)
        return pca

    def _forward_hidden_states_to_pcas(
        self, hidden_states: np.ndarray, pcas: List[PCA]
    ) -> np.ndarray:
        assert len(hidden_states) == len(pcas)
        low_dim_reps = []
        for hidden_states_, pca in zip(hidden_states, pcas):
            low_dim_reps.append(pca.transform(hidden_states_))
        result = np.stack(low_dim_reps)
        return result

    def get_list_index_from_list(self, input_list: List, index_list: List[int]) -> List:
        result_list = []
        for idx, input_elem in enumerate(input_list):
            if idx in index_list:
                result_list.append(input_elem)
        return result_list

    def _preconditions_split_and_fig_size(
        self,
        samples: int,
        fig_size: Tuple[int, int],
        fig_dpi: int,
    ) -> None:
        assert samples >= 1
        assert fig_size[0] >= 1
        assert fig_size[1] >= 1
        assert fig_dpi >= 1
        plt.rcParams["figure.figsize"] = fig_size
        plt.rcParams["figure.dpi"] = fig_dpi  # 200 e.g. is really fine, but slower
        return None

    def _title_figure_save(
        self,
        title: str,
        fig_size: Tuple[int, int],
        fig_dpi: int,
        scatter_legends: List[PathCollection],
        token_labels: List[str],
        save_path: Optional[str | Path],
        max_columns: int = 3,
    ) -> None:
        plt.legend(
            scatter_legends,
            token_labels,
            scatterpoints=1,
            loc="upper right",
            ncol=max_columns,
            fontsize=8,
        )
        plt.title(title)
        plt.figure(figsize=fig_size, dpi=fig_dpi)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def _scatter_plot_over_low_dim_reps(
        self,
        low_dim_reps: List[np.ndarray],
        markers: List[List[str]],
        colors: List[List],
        alpha: float = 1,
    ) -> List[PathCollection]:
        """
        This method expects shapes in form of: [model, positions]
        """
        assert len(low_dim_reps) == len(markers)
        assert len(markers) == len(colors)
        assert len(low_dim_reps[0]) == len(markers[0])
        assert len(markers[0]) == len(colors[0])
        assert alpha > 0
        scatter_legends = []
        for low_dim_reps_on_model, markers_on_model, colors_on_model in zip(
            low_dim_reps, markers, colors
        ):
            for low_dim_reps_on_position, markers_on_positon, colors_on_position in zip(
                low_dim_reps_on_model, markers_on_model, colors_on_model
            ):
                scatter_legends.append(
                    plt.scatter(
                        low_dim_reps_on_position[:, 0],
                        low_dim_reps_on_position[:, 1],
                        marker=markers_on_positon,
                        color=colors_on_position,
                        alpha=alpha,
                    )
                )
        return scatter_legends

    def group_by_source_target_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        df_vanilla = df[df["model"] == "vanilla"].rename(
            columns={"hidden_states": "vanilla_hidden_states"}
        )
        df_graph_prompter_hf_frozen = df[
            df["model"] == "graph_prompter_hf_frozen"
        ].rename(columns={"hidden_states": "graph_prompter_hf_frozen_hidden_states"})
        df_graph_prompter_hf = df[df["model"] == "graph_prompter_hf"].rename(
            columns={"hidden_states": "graph_prompter_hf_hidden_states"}
        )
        if len(df_vanilla) > 0:
            if len(df_graph_prompter_hf_frozen) > 0:
                df = df_vanilla.merge(
                    df_graph_prompter_hf_frozen,
                    on=["source_id", "target_id"],
                    suffixes=("", "_y"),
                )
            else:
                df = df_vanilla
        else:
            df = df_graph_prompter_hf_frozen
        if len(df_graph_prompter_hf) > 0:
            df = df.merge(
                df_graph_prompter_hf,
                on=["source_id", "target_id"],
                suffixes=("", "_y"),
            )
        return df

    def plot_cls_embeddings(
        self,
        samples=1,
        fig_size: Tuple[int, int] = (8, 8),
        fig_dpi: int = 200,
        save_plot: bool = True,
    ):
        assert samples >= 1
        self._preconditions_split_and_fig_size(samples, fig_size, fig_dpi)
        all_dfs = []
        for split, model in itertools.product(
            ["test", "val"],
            ["vanilla", "graph_prompter_hf_frozen", "graph_prompter_hf"],
        ):
            df, _, _ = self.load_df(split, model, [])
            df = df[
                ["model", "split", "source_id", "target_id", "hidden_states", "labels"]
            ]
            if model == "vanilla":
                hidden_states = np.stack(df["hidden_states"].tolist())
                df["hidden_states"] = np.pad(
                    hidden_states, ((0, 0), (0, 2), (0, 0), (0, 0)), mode="constant"
                ).tolist()
            all_dfs.append(df)
        df = pd.concat(all_dfs)
        df["degree"] = df.groupby("target_id")["target_id"].transform("count")
        df_test = df[df["split"] == "test"]
        df_val = self.group_by_source_target_ids(df[df["split"] == "val"])
        # produce plot for model separated
        df_val_with_edges = df_val[df_val["labels"] == 1]
        lower_bound = df_val_with_edges["degree"].quantile(1 / 3)
        upper_bound = df_val_with_edges["degree"].quantile(2 / 3)
        df_val = df_val.sample(samples)
        # produce plot for model separated
        df_val_with_edges = df_val[df_val["labels"] == 1]
        df_val_without_edges = df_val[df_val["labels"] == 0]

        pca = self.fit_pca(np.stack(df_test["hidden_states"].tolist())[:, 0, 2])

        vanilla_hidden_states_with_edges = np.stack(
            df_val_with_edges["vanilla_hidden_states"].tolist()
        )[:, 0, 2]
        graph_prompter_hf_frozen_hidden_states_with_edges = np.stack(
            df_val_with_edges["graph_prompter_hf_frozen_hidden_states"]
        )[:, 0, 2]
        graph_prompter_hf_hidden_states_with_edges = np.stack(
            df_val_with_edges["graph_prompter_hf_hidden_states"]
        )[:, 0, 2]
        vanilla_hidden_states_with_edges = np.expand_dims(
            pca.transform(vanilla_hidden_states_with_edges), axis=0
        )
        graph_prompter_hf_frozen_hidden_states_with_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_frozen_hidden_states_with_edges), axis=0
        )
        graph_prompter_hf_hidden_states_with_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_hidden_states_with_edges), axis=0
        )

        vanilla_hidden_states_without_edges = np.stack(
            df_val_without_edges["vanilla_hidden_states"].tolist()
        )[:, 0, 2]
        graph_prompter_hf_frozen_hidden_states_without_edges = np.stack(
            df_val_without_edges["graph_prompter_hf_frozen_hidden_states"]
        )[:, 0, 2]
        graph_prompter_hf_hidden_states_without_edges = np.stack(
            df_val_without_edges["graph_prompter_hf_hidden_states"]
        )[:, 0, 2]
        vanilla_hidden_states_without_edges = np.expand_dims(
            pca.transform(vanilla_hidden_states_without_edges), axis=0
        )
        graph_prompter_hf_frozen_hidden_states_without_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_frozen_hidden_states_without_edges), axis=0
        )
        graph_prompter_hf_hidden_states_without_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_hidden_states_without_edges), axis=0
        )
        hidden_states = [
            vanilla_hidden_states_with_edges,
            vanilla_hidden_states_without_edges,
            graph_prompter_hf_frozen_hidden_states_with_edges,
            graph_prompter_hf_frozen_hidden_states_without_edges,
            graph_prompter_hf_hidden_states_with_edges,
            graph_prompter_hf_hidden_states_without_edges,
        ]

        colors = cm.rainbow(np.linspace(0, 1, 3))  # type: ignore
        scatter_legends = self._scatter_plot_over_low_dim_reps(
            hidden_states,
            markers=[["."], ["x"], ["."], ["x"], ["."], ["x"]],
            colors=[
                [colors[0]],
                [colors[0]],
                [colors[1]],
                [colors[1]],
                [colors[2]],
                [colors[2]],
            ],
            alpha=0.7,
        )
        save_path = "./images/cls_hidden_states.png" if save_plot else None
        self._title_figure_save(
            "CLS Hidden States",
            fig_size,
            fig_dpi,
            scatter_legends,
            [
                "Vanilla CLS with edges",
                "Vanilla CLS without edges",
                "GraphPrompterHF frozen CLS with edges",
                "GraphPrompterHF frozen CLS without edges",
                "GraphPrompterHF CLS with edges",
                "GraphPrompterHF CLS without edges",
            ],
            save_path,
            max_columns=2,
        )

        hidden_states_lower = df_val[df_val["degree"] <= lower_bound]
        hidden_states_middle = df_val[
            (df_val["degree"] > lower_bound) & (df_val["degree"] <= upper_bound)
        ]
        hidden_states_upper = df_val[df_val["degree"] > upper_bound]
        quantiles = []
        for chunk in [hidden_states_lower, hidden_states_middle, hidden_states_upper]:
            hidden_states = []
            for model in ["vanilla", "graph_prompter_hf_frozen", "graph_prompter_hf"]:
                hidden_states.append(
                    np.stack(chunk[f"{model}_hidden_states"].tolist())[:, 0, 2]
                )
            hidden_states = np.concatenate(hidden_states)
            hidden_states = np.expand_dims(pca.transform(hidden_states), axis=0)
            quantiles.append(hidden_states)

        colors = cm.rainbow(np.linspace(0, 1, 3))  # type: ignore
        scatter_legends = self._scatter_plot_over_low_dim_reps(
            quantiles,
            markers=[["x"], ["x"], ["x"]],
            colors=[
                [colors[0]],
                [colors[1]],
                [colors[2]],
            ],
            alpha=0.7,
        )
        save_path = "./images/cls_hidden_states_degree.png" if save_plot else None
        self._title_figure_save(
            "CLS Hidden States of Node-Pairs with Grouped by Movie Degree",
            fig_size,
            fig_dpi,
            scatter_legends,
            ["lower quantile", "middle quantile", "upper_quantil"],
            save_path,
        )

    def plot_kges(
        self,
        samples=1,
        fig_size: Tuple[int, int] = (8, 8),
        fig_dpi: int = 200,
        save_plot: bool = True,
    ):
        assert samples >= 1
        self._preconditions_split_and_fig_size(samples, fig_size, fig_dpi)
        all_dfs = []
        for split, model in itertools.product(
            ["test", "val"],
            ["graph_prompter_hf_frozen", "graph_prompter_hf"],
        ):
            df, _, _ = self.load_df(split, model, [])
            df = df[
                ["model", "split", "source_id", "target_id", "hidden_states", "labels"]
            ]
            all_dfs.append(df)
        df = pd.concat(all_dfs)

        # compute cosine similarity between kges
        df_val = df[df["split"] == "val"]
        df_val_frozen_hidden_states_with_edges = np.stack(
            df[(df["model"] == "graph_prompter_hf_frozen") & (df["labels"] == 1)][
                "hidden_states"
            ]
        )
        df_val_hidden_states_with_edges = np.stack(
            df[(df["model"] == "graph_prompter_hf") & (df["labels"] == 1)][
                "hidden_states"
            ]
        )
        average_frozen_cosine_similarity_with_edges = self.__average_cosine_similarity(
            df_val_frozen_hidden_states_with_edges[:, 4, 0],
            df_val_frozen_hidden_states_with_edges[:, 5, 0],
        )
        average_cosine_similarity_with_edges = self.__average_cosine_similarity(
            df_val_hidden_states_with_edges[:, 4, 0],
            df_val_hidden_states_with_edges[:, 5, 0],
        )
        df_val_frozen_hidden_states_without_edges = np.stack(
            df[(df["model"] == "graph_prompter_hf_frozen") & (df["labels"] == 0)][
                "hidden_states"
            ]
        )
        df_val_hidden_states_without_edges = np.stack(
            df[(df["model"] == "graph_prompter_hf") & (df["labels"] == 0)][
                "hidden_states"
            ]
        )
        average_frozen_cosine_similarity_without_edges = (
            self.__average_cosine_similarity(
                df_val_frozen_hidden_states_without_edges[:, 4, 0],
                df_val_frozen_hidden_states_without_edges[:, 5, 0],
            )
        )
        average_cosine_similarity_without_edges = self.__average_cosine_similarity(
            df_val_hidden_states_without_edges[:, 4, 0],
            df_val_hidden_states_without_edges[:, 5, 0],
        )
        print(
            f"Average cosine similarity of frozen GraphPrompterHF model with edges is {average_frozen_cosine_similarity_with_edges}."
        )
        print(
            f"Average cosine similarity of end-to-end GraphPrompterHF model with edges is {average_cosine_similarity_with_edges}.",
        )
        print(
            f"Average cosine similarity of frozen GraphPrompterHF model without edges is {average_frozen_cosine_similarity_without_edges}.",
        )
        print(
            f"Average cosine similarity of end-to-end GraphPrompterHF model without edges is {average_cosine_similarity_without_edges}.",
        )

        df["degree"] = df.groupby("target_id")["target_id"].transform("count")
        df_test = df[df["split"] == "test"]
        df_val = self.group_by_source_target_ids(df[df["split"] == "val"])
        df_val = df_val.sample(samples)

        # produce plot for model separated
        df_val_with_edges = df_val[df_val["labels"] == 1]
        df_val_without_edges = df_val[df_val["labels"] == 0]
        hidden_states = np.stack(df_test["hidden_states"].tolist())
        user_kges = hidden_states[:, 4, 0]
        movie_kges = hidden_states[:, 5, 0]
        hidden_states = np.concatenate([user_kges, movie_kges])

        pca = self.fit_pca(hidden_states)

        graph_prompter_hf_frozen_user_hidden_states_with_edges = np.stack(
            df_val_with_edges["graph_prompter_hf_frozen_hidden_states"]
        )[:, 4, 0]
        graph_prompter_hf_user_hidden_states_with_edges = np.stack(
            df_val_with_edges["graph_prompter_hf_hidden_states"]
        )[:, 4, 0]
        graph_prompter_hf_frozen_movie_hidden_states_with_edges = np.stack(
            df_val_with_edges["graph_prompter_hf_frozen_hidden_states"]
        )[:, 5, 0]
        graph_prompter_hf_movie_hidden_states_with_edges = np.stack(
            df_val_with_edges["graph_prompter_hf_hidden_states"]
        )[:, 5, 0]
        graph_prompter_hf_frozen_user_hidden_states_with_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_frozen_user_hidden_states_with_edges),
            axis=0,
        )
        graph_prompter_hf_user_hidden_states_with_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_user_hidden_states_with_edges), axis=0
        )
        graph_prompter_hf_frozen_movie_hidden_states_with_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_frozen_movie_hidden_states_with_edges),
            axis=0,
        )
        graph_prompter_hf_movie_hidden_states_with_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_movie_hidden_states_with_edges), axis=0
        )

        graph_prompter_hf_frozen_user_hidden_states_without_edges = np.stack(
            df_val_without_edges["graph_prompter_hf_frozen_hidden_states"]
        )[:, 4, 0]
        graph_prompter_hf_user_hidden_states_without_edges = np.stack(
            df_val_without_edges["graph_prompter_hf_hidden_states"]
        )[:, 4, 0]
        graph_prompter_hf_frozen_movie_hidden_states_without_edges = np.stack(
            df_val_without_edges["graph_prompter_hf_frozen_hidden_states"]
        )[:, 5, 0]
        graph_prompter_hf_movie_hidden_states_without_edges = np.stack(
            df_val_without_edges["graph_prompter_hf_hidden_states"]
        )[:, 5, 0]
        graph_prompter_hf_frozen_user_hidden_states_without_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_frozen_user_hidden_states_without_edges),
            axis=0,
        )
        graph_prompter_hf_user_hidden_states_without_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_user_hidden_states_without_edges), axis=0
        )
        graph_prompter_hf_frozen_movie_hidden_states_without_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_frozen_movie_hidden_states_without_edges),
            axis=0,
        )
        graph_prompter_hf_movie_hidden_states_without_edges = np.expand_dims(
            pca.transform(graph_prompter_hf_movie_hidden_states_without_edges), axis=0
        )
        hidden_states = [
            graph_prompter_hf_frozen_user_hidden_states_with_edges,
            graph_prompter_hf_user_hidden_states_with_edges,
            graph_prompter_hf_frozen_movie_hidden_states_with_edges,
            graph_prompter_hf_movie_hidden_states_with_edges,
            graph_prompter_hf_frozen_user_hidden_states_without_edges,
            graph_prompter_hf_user_hidden_states_without_edges,
            graph_prompter_hf_frozen_movie_hidden_states_without_edges,
            graph_prompter_hf_movie_hidden_states_without_edges,
        ]

        colors = cm.rainbow(np.linspace(0, 1, 4))  # type: ignore
        scatter_legends = self._scatter_plot_over_low_dim_reps(
            hidden_states,
            markers=[["."], ["."], ["x"], ["x"], ["*"], ["*"], ["+"], ["+"]],
            colors=[
                [colors[0]],
                [colors[1]],
                [colors[0]],
                [colors[1]],
                [colors[2]],
                [colors[3]],
                [colors[2]],
                [colors[3]],
            ],
            alpha=0.7,
        )

        for user_kges, movie_kges in zip(
            [
                graph_prompter_hf_frozen_user_hidden_states_with_edges,
                graph_prompter_hf_user_hidden_states_with_edges,
                graph_prompter_hf_frozen_user_hidden_states_without_edges,
                graph_prompter_hf_user_hidden_states_without_edges,
            ],
            [
                graph_prompter_hf_frozen_movie_hidden_states_with_edges,
                graph_prompter_hf_movie_hidden_states_with_edges,
                graph_prompter_hf_frozen_movie_hidden_states_without_edges,
                graph_prompter_hf_movie_hidden_states_without_edges,
            ],
        ):
            line_segments = [
                ((user_kge[0], user_kge[1]), (movie_kge[0], movie_kge[1]))
                for user_kge, movie_kge in zip(user_kges[0, :10], movie_kges[0, :10])
            ]
            # Add lines as a LineCollection
            lc = LineCollection(line_segments, colors="gray", alpha=0.7)
            # plt.gca().add_collection(lc)

        save_path = "./images/kge_hidden_states.png" if save_plot else None
        self._title_figure_save(
            "KGEs hidden states",
            fig_size,
            fig_dpi,
            scatter_legends,
            [
                "frozen User KGEs with Edges",
                "User KGEs with Edges",
                "frozen Movie KGEs with Edges",
                "Movie KGEs with Edges",
                "frozen User KGEs without Edges",
                "User KGEs without Edges",
                "frozen Movie KGEs without Edges",
                "Movie KGEs without Edges",
            ],
            save_path,
            max_columns=2,
        )

    def __average_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.float32:
        """
        Compute the average cosine similarity along the first dimension of two tensors.
        By ChatGPT
        Returns:
            The average cosine similarity over dimension 0.
        """
        similarities = []
        for i in range(a.shape[0]):
            slice1 = a[i]
            slice2 = b[i]
            similarity = self.__cosine_similarity(slice1, slice2)
            similarities.append(similarity)

        # Average the similarities
        return np.mean(similarities)

    def __cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute the cosine similarity between two numpy arrays.
        From ChatGPT
        """
        # Ensure both arrays are 1D
        a = np.ravel(a)
        b = np.ravel(b)

        # Compute the dot product and norms
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return np.array(0.0)

        return dot_product / (norm_a * norm_b)

    def plot_cls_true_vs_non_true(
        self,
        samples=1,
        fig_size: Tuple[int, int] = (8, 8),
        fig_dpi: int = 200,
        split: Optional[str] = None,
        save_path: Optional[str | Path] = None,
    ):
        df = self._preconditions_split_and_fig_size(samples, fig_size, fig_dpi, split)
        existing_df = df[df["labels"] == 1]
        non_existing_df = df[df["labels"] == 0]
        existing_df = existing_df.sample(samples)
        non_existing_df = non_existing_df.sample(samples)
        existing_low_dim_reps = self.produce_low_dim_reps_over_layer(
            -1, [0], existing_df
        )
        non_existing_low_dim_reps = self.produce_low_dim_reps_over_layer(
            -1, [0], non_existing_df
        )
        colors = cm.rainbow(np.linspace(0, 1, 2))  # type: ignore
        scatter_legends = []
        scatter_legends = self._scatter_plot_over_low_dim_reps(
            existing_low_dim_reps,
            markers=[["o"], ["X"], ["v"]],
            colors=[[colors[0]], [colors[0]], [colors[0]]],
        )
        scatter_legends.extend(
            self._scatter_plot_over_low_dim_reps(
                non_existing_low_dim_reps,
                markers=[["o"], ["X"], ["v"]],
                colors=[[colors[1]], [colors[1]], [colors[1]]],
            )
        )
        self._title_figure_save(
            "CLS Token Embeddings in last Layer: Existing vs Non-Existing",
            fig_size,
            fig_dpi,
            scatter_legends,
            [
                "vanilla existing",
                "input embeds replace existing",
                "vanilla non-existing",
                "input embeds replace non-existing",
            ],
            save_path,
        )

    def plot_user_movie_embeddings_in_second_layer_true_vs_non_true(
        self,
        samples=1,
        fig_size: Tuple[int, int] = (8, 8),
        fig_dpi: int = 200,
        split: Optional[str] = None,
        save_paths: Optional[List[str | Path]] = None,
    ):
        if save_paths:
            assert len(save_paths) == 3
        else:
            save_path = None
        df = self._preconditions_split_and_fig_size(samples, fig_size, fig_dpi, split)
        existing_edges_df = df[df["labels"] == 1].sample(samples)
        non_existing_edges_df = df[df["labels"] == 0].sample(samples)
        low_dim_reps_existing = self.produce_low_dim_reps_over_layer(
            1, [2, 4, 6], existing_edges_df
        )
        low_dim_reps_non_existing = self.produce_low_dim_reps_over_layer(
            1, [2, 4, 6], non_existing_edges_df
        )
        low_dim_reps = np.concatenate(
            [low_dim_reps_existing, low_dim_reps_non_existing]
        )
        colors = cm.rainbow(np.linspace(0, 1, 2))  # type: ignore
        print(
            low_dim_reps.shape,
            low_dim_reps_non_existing.shape,
            low_dim_reps_non_existing.shape,
        )
        scatter_legends = self._scatter_plot_over_low_dim_reps(
            low_dim_reps[[0, 3]],
            markers=[["o", "X", "v"], ["o", "X", "v"]],
            colors=[
                [colors[0], colors[0], colors[0]],
                [colors[1], colors[1], colors[1]],
            ],
        )
        token_labels = [
            "existing user",
            "existing title",
            "existing genres",
            "non-existing user",
            "non-existing title",
            "non-existing genres",
        ]
        if save_paths:
            save_path = save_paths[0]
        self._title_figure_save(
            "Vanilla User, Title, Genres Token Embeddings in Second Layer existing vs non exsting",
            fig_size,
            fig_dpi,
            scatter_legends,
            token_labels,
            save_path,
        )
        scatter_legends = self._scatter_plot_over_low_dim_reps(
            low_dim_reps[[1, 4]],
            markers=[["o", "X", "v"], ["o", "X", "v"]],
            colors=[
                [colors[0], colors[0], colors[0]],
                [colors[1], colors[1], colors[1]],
            ],
        )
        if save_paths:
            save_path = save_paths[1]
        self._title_figure_save(
            "Prompt User, Title, Genres Token Embeddings in Second Layer existing vs non exsting",
            fig_size,
            fig_dpi,
            scatter_legends,
            token_labels,
            save_path,
        )
        scatter_legends = self._scatter_plot_over_low_dim_reps(
            low_dim_reps[[2, 5]],
            markers=[["o", "X", "v"], ["o", "X", "v"]],
            colors=[
                [colors[0], colors[0], colors[0]],
                [colors[1], colors[1], colors[1]],
            ],
        )
        if save_paths:
            save_path = save_paths[2]
        self._title_figure_save(
            "Input Embeds Replace User, Title, Genres Token Embeddings in Second Layer existing vs non exsting",
            fig_size,
            fig_dpi,
            scatter_legends,
            token_labels,
            save_path,
        )

    def nearest_neighbor(self, point, points):
        distances = np.linalg.norm(points - point, axis=1)
        return np.argmin(distances)

    def filter_by_singular_source(self, df: pd.DataFrame, min_size=25) -> pd.DataFrame:
        count_ids = df["source_id"].value_counts().rename("count_ids")
        count_ids = count_ids[count_ids["count_ids"] >= min_size]
        if not len(count_ids) > 0:  # type: ignore
            raise Exception("There are not enough source_ids matching over all models.")
        singular_source = rd.choice(count_ids["source_id"].unique())  # type: ignore
        return df[df["source_id"] == singular_source]

    def filter_by_singular_target(self, df: pd.DataFrame, min_size=25) -> pd.DataFrame:
        count_ids = df["target_id"].value_counts().rename("count_ids")
        count_ids = count_ids[count_ids["count_ids"] >= min_size]
        if not len(count_ids) > 0:  # type: ignore
            raise Exception("There are not enough targets matching over all models.")
        singular_target = rd.choice(count_ids["target_id"].unique())  # type: ignore
        return df[df["target_id"] == singular_target]


class MovieLensExplainabilityModule(ExplainabilityModule):
    def __init__(self, llm_df: pd.DataFrame, init_pcas=False) -> None:
        super().__init__(
            llm_df,
            ["cls", "user", "sep1", "movie", "sep2", "title", "sep3", "genres", "sep4"],
            ["user embedding", "sep5", "movie embedding", "sep6"],
            init_pcas=init_pcas,
        )
