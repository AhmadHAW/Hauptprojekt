import json
from typing import List, Tuple, Optional, Callable
import random as rd
from pathlib import Path
import ast

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import joblib
from matplotlib.collections import PathCollection

from dataset_manager.kg_manager import ROOT
from llm_manager.vanilla.config import VANILLA_TOKEN_DICT
from llm_manager.graph_prompter_hf.config import GRAPH_PROMPTER_TOKEN_DICT


class ExplainabilityModule:
    def __init__(
        self,
    ) -> None:
        self.llm_df = pd.read_csv(f"{ROOT}/llm/dataset.csv")
        self.vanilla_path = f"{ROOT}/llm/vanilla"
        self.graph_prompter_hf_path = f"{ROOT}/llm/graph_prompter_hf"
        self.graph_prompter_hf_frozen_path = f"{ROOT}/llm/graph_prompter_hf_frozen"
        self.vanilla_xai_artifact_root = f"{self.vanilla_path}/xai_artifacts"
        self.graph_prompter_hf_xai_artifact_root = (
            f"{self.graph_prompter_hf_path}/xai_artifacts"
        )
        self.graph_prompter_hf_frozen_xai_artifact_root = (
            f"{self.graph_prompter_hf_frozen_path}/xai_artifacts"
        )
        self.vanilla_training_path = (
            f"{self.vanilla_path}/training/checkpoint-2210/trainer_state.json"
        )
        self.graph_prompter_hf_training_path = (
            f"{self.graph_prompter_hf_path}/training/checkpoint-4420/trainer_state.json"
        )
        self.graph_prompter_hf_frozen_training_path = f"{self.graph_prompter_hf_frozen_path}/training/checkpoint-2210/trainer_state.json"

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
        graph_prompter_frozen_losses = [vanilla_losses[-1]]
        graph_prompter_frozen_losses.extend(
            [
                entry["loss"]
                for entry in graph_prompter_hf_frozen_training_process["log_history"]
                if "loss" in entry
            ]
        )
        graph_prompter_losses = [graph_prompter_frozen_losses[-1]]
        graph_prompter_losses.extend(
            [
                entry["loss"]
                for entry in graph_prompter_hf_training_process["log_history"]
                if "loss" in entry
            ]
        )
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
            (line,) = plt.plot(steps, losses, label=label)
            color = line.get_color()
            min_loss = min(losses)
            min_loss_index = steps[losses.index(min_loss)]
            plt.plot(
                min_loss_index,
                min_loss,
                "o",
                color=color,
                label=f"Min {label}: {min_loss:.4f}",
                markersize=8,
            )

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
        # Extract the loss values
        vanilla_eval_accuracy = [
            entry["eval_accuracy"]
            for entry in vanilla_training_process["log_history"]
            if "eval_accuracy" in entry
        ]
        graph_prompter_frozen_eval_accuracy = [vanilla_eval_accuracy[-1]]
        graph_prompter_frozen_eval_accuracy.extend(
            [
                entry["eval_accuracy"]
                for entry in graph_prompter_hf_frozen_training_process["log_history"]
                if "eval_accuracy" in entry
            ]
        )
        graph_prompter_eval_accuracy = [graph_prompter_frozen_eval_accuracy[-1]]
        graph_prompter_eval_accuracy.extend(
            [
                entry["eval_accuracy"]
                for entry in graph_prompter_hf_training_process["log_history"]
                if "eval_accuracy" in entry
            ]
        )
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

        def plot(accuracies: List[float], epochs: list[float], label: str):
            # Plot the loss curve
            (line,) = plt.plot(epochs, accuracies, label=label)
            color = line.get_color()
            max_accuracy = max(accuracies)
            max_accuracy_index = epochs[accuracies.index(max_accuracy)]
            plt.plot(
                max_accuracy_index,
                max_accuracy,
                "o",
                color=color,
                label=f"max {label}: {max_accuracy:.4f}",
                markersize=8,
            )

        plot(vanilla_eval_accuracy, vanilla_epochs, vanilla_label)
        plot(
            graph_prompter_frozen_eval_accuracy,
            graph_prompter_frozen_epochs,
            graph_prompter_frozen_label,
        )
        plot(graph_prompter_eval_accuracy, graph_prompter_epochs, graph_prompter_label)
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

    def __str_to_nparray(self, seq: str) -> np.ndarray:
        return np.array(ast.literal_eval(seq))

    def plot_attention_maps(
        self,
        split: str,
        masks: List[Tuple] | Tuple,
        model: str = "vanilla",
        filter: Optional[Callable] = None,
        weight_coef: int | float = 15,
        fig_dpi: int = 100,
        fig_size: Tuple[int, int] = (8, 8),
        save_plot: bool = True,
    ) -> None:
        assert weight_coef > 0
        assert fig_dpi > 0
        assert fig_size[0] > 0
        assert fig_size[1] > 0
        if model == "vanilla":
            base_path_to_csv = f"{self.vanilla_xai_artifact_root}/{split}_{{}}.csv"
            token_type_dict = VANILLA_TOKEN_DICT
        elif model == "graph_prompter_hf_frozen":
            base_path_to_csv = (
                f"{self.graph_prompter_hf_frozen_xai_artifact_root}/{split}_{{}}.csv"
            )
            token_type_dict = GRAPH_PROMPTER_TOKEN_DICT
        else:
            base_path_to_csv = (
                f"{self.graph_prompter_hf_xai_artifact_root}/{split}_{{}}.csv"
            )
            token_type_dict = GRAPH_PROMPTER_TOKEN_DICT
        token_type_dict_reverse = {v: k for k, v in token_type_dict.items()}

        if isinstance(masks, Tuple):
            df = pd.read_csv(
                base_path_to_csv.format(masks),
                converters={"attention_maps": self.__str_to_nparray},
            )
        else:
            dfs: List[pd.DataFrame] = []
            for mask in masks:
                dfs.append(
                    pd.read_csv(
                        base_path_to_csv.format(mask),
                        converters={"attention_maps": self.__str_to_nparray},
                    )
                )
            df = pd.concat(dfs)
        if filter is not None:
            df = df[df.apply(filter, axis=1)]
        attention_maps = np.stack(df["attention_maps"].tolist())
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
            f"Attention map of model {model}, split {split} and mask(s) {masks if len(masks) > 0 else 'None'}"
        )
        if save_plot:
            plt.savefig(f"./images/attentipn_map_{model}_{split}_{masks}.png")
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

    def init_pcas_for_model(
        self,
        hidden_states: np.ndarray,
        token_labels: List[str],
        pca_target_base_path_template: Optional[str] = None,
    ) -> List[List[PCA]]:
        hidden_states = np.transpose(hidden_states, (1, 0, 2, 3))  # type: ignore
        pcas: List[List[PCA]] = []
        for layer_num, layer in enumerate(hidden_states):
            pcas_on_layer: List[PCA] = []
            pcas.append(pcas_on_layer)
            for idx, label in enumerate(token_labels):
                pca = PCA(n_components=2)  # Adjust number of components as needed
                pcas_on_layer.append(pca)
                position_hidden_states = layer[:, idx]
                pca.fit_transform(position_hidden_states)
                if pca_target_base_path_template:
                    pca_target_path = pca_target_base_path_template.format(
                        label, layer_num
                    )
                    joblib.dump(pca, pca_target_path)
        return pcas

    def init_pcas(self, split="train"):
        df = self.llm_df[self.llm_df["split"] == split]
        vanilla_hidden_states = np.stack(df["vanilla_hidden_states"].tolist())
        graph_prompter_hf_hidden_states = np.stack(
            df["graph_prompter_hf_hidden_states"].tolist()
        )
        self.vanilla_pcas = self.init_pcas_for_model(
            vanilla_hidden_states, self.vanilla_tokens
        )
        self.graph_prompter_hf_pcas = self.init_pcas_for_model(
            graph_prompter_hf_hidden_states,
            self.vanilla_tokens + self.additional_embedding_tokens,
        )

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
        split: Optional[str] = None,
    ) -> pd.DataFrame:
        assert samples >= 1
        assert fig_size[0] >= 1
        assert fig_size[1] >= 1
        assert fig_dpi >= 1
        plt.rcParams["figure.figsize"] = fig_size
        plt.rcParams["figure.dpi"] = fig_dpi  # 200 e.g. is really fine, but slower
        df = self.llm_df
        if split:
            df = df[df["split"] == split]
        return df

    def produce_low_dim_reps_over_layer(
        self, layer: int, token_index: List[int], df: pd.DataFrame
    ) -> np.ndarray:
        assert layer < len(self.vanilla_pcas)
        assert max(token_index) < len(self.vanilla_pcas[0])
        vanilla_pcas = self.get_list_index_from_list(
            self.vanilla_pcas[layer], token_index
        )
        graph_prompter_hf_pcas = self.get_list_index_from_list(
            self.graph_prompter_hf_pcas[layer], token_index
        )
        vanilla_hidden_states = np.stack(df["vanilla_hidden_states"].tolist())[
            :, layer, token_index
        ].transpose(1, 0, 2)  # shape (token positions, dataset size, embedding size)
        graph_prompter_hf_hidden_states = np.stack(
            df["graph_prompter_hf_hidden_states"].tolist()
        )[:, layer, token_index].transpose(1, 0, 2)
        low_dim_reps_vanilla = self._forward_hidden_states_to_pcas(
            vanilla_hidden_states, vanilla_pcas
        )
        low_dim_reps_embedding = self._forward_hidden_states_to_pcas(
            graph_prompter_hf_hidden_states, graph_prompter_hf_pcas
        )
        return np.stack([low_dim_reps_vanilla, low_dim_reps_embedding])

    def _title_figure_save(
        self,
        title: str,
        fig_size: Tuple[int, int],
        fig_dpi: int,
        scatter_legends: List[PathCollection],
        token_labels: List[str],
        save_path: Optional[str | Path],
    ) -> None:
        plt.legend(
            scatter_legends,
            token_labels,
            scatterpoints=1,
            loc="upper right",
            ncol=3,
            fontsize=8,
        )
        plt.title(title)
        plt.figure(figsize=fig_size, dpi=fig_dpi)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def _scatter_plot_over_low_dim_reps(
        self, low_dim_reps: np.ndarray, markers: List[List[str]], colors: List[List]
    ) -> List[PathCollection]:
        """
        This method expects shapes in form of: [model, positions]
        """
        assert len(low_dim_reps) == len(markers)
        assert len(markers) == len(colors)
        assert len(low_dim_reps[0]) == len(markers[0])
        assert len(markers[0]) == len(colors[0])
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
                    )
                )
        return scatter_legends

    def plot_cls_embeddings(
        self,
        samples=1,
        fig_size: Tuple[int, int] = (8, 8),
        fig_dpi: int = 200,
        split: Optional[str] = None,
        save_path: Optional[str | Path] = None,
    ):
        assert samples >= 1
        df = self._preconditions_split_and_fig_size(samples, fig_size, fig_dpi, split)
        df = df.sample(samples)

        low_dim_reps = self.produce_low_dim_reps_over_layer(-1, [0], df)
        colors = cm.rainbow(np.linspace(0, 1, 3))  # type: ignore
        scatter_legends = self._scatter_plot_over_low_dim_reps(
            low_dim_reps,
            markers=[["o"], ["o"], ["o"]],
            colors=[[colors[0]], [colors[1]], [colors[2]]],
        )
        self._title_figure_save(
            "CLS Tokens Embeddings in last Layer",
            fig_size,
            fig_dpi,
            scatter_legends,
            ["vanilla cls", "input embeds replace cls"],
            save_path,
        )

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
