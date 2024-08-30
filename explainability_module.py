from dataset_manager import MovieLensManager

from typing import List, Tuple, Optional
import random as rd
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import joblib
from matplotlib.collections import PathCollection


class ExplainabilityModule:
    def __init__(
        self,
        llm_df: pd.DataFrame,
        vanilla_tokens: List[str],
        additional_embedding_tokens: List[str],
        init_pcas: bool = False,
    ) -> None:
        self.llm_df = llm_df
        self.vanilla_tokens = vanilla_tokens
        self.additional_embedding_tokens = additional_embedding_tokens
        if init_pcas:
            self.init_pcas()

    def plot_attention_graph(
        self,
        attentions: torch.Tensor,
        token_labels: List[str],
        title: str,
        weight_coef: int | float = 5,
        fig_dpi: int = 100,
        fig_size: Tuple[int, int] = (8, 8),
        save_path: Optional[str | Path] = None,
    ) -> None:
        assert weight_coef > 0
        assert fig_dpi > 0
        assert fig_size[0] > 0
        assert fig_size[1] > 0

        plt.rcParams["figure.figsize"] = fig_size
        plt.rcParams["figure.dpi"] = fig_dpi  # 200 e.g. is really fine, but slower
        plt.figure(figsize=fig_size, dpi=fig_dpi)
        # Create an undirected graph
        G = nx.Graph()
        labels = {}
        attentions = torch.mean(attentions, dim=0).permute((2, 0, 1))
        for layer, attentions_ in enumerate(attentions):
            for from_, inner in enumerate(attentions_):
                from_name = f"{token_labels[from_]}_{layer}"
                if layer == 0:
                    labels[from_name] = token_labels[from_]
                G.add_node(from_name, name=from_name, layer=layer)
                for to_, weight in enumerate(inner):
                    to_name = f"{token_labels[to_]}_{layer+1}"
                    G.add_node(to_name, name=to_name, layer=layer + 1)
                    G.add_edge(from_name, to_name, weight=weight)

        pos = nx.multipartite_layout(G, subset_key="layer")
        semantic_datapoints = token_labels.copy()
        semantic_datapoints.reverse()
        for node, (x, y) in pos.items():
            y = semantic_datapoints.index(node.split("_")[0])
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
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_all_attention_graphs(
        self, save_paths: Optional[List[str]] = None, weight_coef: int = 10
    ):
        self.plot_attention_graph(
            torch.Tensor(self.llm_df["vanilla_attentions"].tolist()),
            self.vanilla_tokens,
            "Vanilla Model Attentions Plot",
            weight_coef=weight_coef,
            save_path=save_paths[0] if save_paths else None,
        )
        self.plot_attention_graph(
            torch.Tensor(self.llm_df["prompt_attentions"].tolist()),
            self.vanilla_tokens + self.additional_embedding_tokens,
            "Prompt Model Attentions Plot",
            weight_coef=weight_coef,
            save_path=save_paths[1] if save_paths else None,
        )
        self.plot_attention_graph(
            torch.Tensor(self.llm_df["input_embeds_replace_attentions"].tolist()),
            self.vanilla_tokens + self.additional_embedding_tokens,
            "Input Embeds Replace Model Attentions Plot",
            weight_coef=weight_coef,
            save_path=save_paths[2] if save_paths else None,
        )

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
        prompt_hidden_states = np.stack(df["prompt_hidden_states"].tolist())
        input_embeds_replace_hidden_states = np.stack(
            df["input_embeds_replace_hidden_states"].tolist()
        )
        self.vanilla_pcas = self.init_pcas_for_model(
            vanilla_hidden_states, self.vanilla_tokens
        )
        self.prompt_pcas = self.init_pcas_for_model(
            prompt_hidden_states, self.vanilla_tokens + self.additional_embedding_tokens
        )
        self.input_embeds_replace_pcas = self.init_pcas_for_model(
            input_embeds_replace_hidden_states,
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
        prompt_pcas = self.get_list_index_from_list(
            self.prompt_pcas[layer], token_index
        )
        input_embeds_replace_pcas = self.get_list_index_from_list(
            self.input_embeds_replace_pcas[layer], token_index
        )
        vanilla_hidden_states = np.stack(df["vanilla_hidden_states"].tolist())[
            :, layer, token_index
        ].transpose(1, 0, 2)  # shape (token positions, dataset size, embedding size)
        prompt_hidden_states = np.stack(df["prompt_hidden_states"].tolist())[
            :, layer, token_index
        ].transpose(1, 0, 2)
        input_embeds_replace_hidden_states = np.stack(
            df["input_embeds_replace_hidden_states"].tolist()
        )[:, layer, token_index].transpose(1, 0, 2)
        low_dim_reps_vanilla = self._forward_hidden_states_to_pcas(
            vanilla_hidden_states, vanilla_pcas
        )
        low_dim_reps_prompt = self._forward_hidden_states_to_pcas(
            prompt_hidden_states, prompt_pcas
        )
        low_dim_reps_embedding = self._forward_hidden_states_to_pcas(
            input_embeds_replace_hidden_states, input_embeds_replace_pcas
        )
        return np.stack(
            [low_dim_reps_vanilla, low_dim_reps_prompt, low_dim_reps_embedding]
        )

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
            ["vanilla cls", "prompt cls", "input embeds replace cls"],
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
        print(len(df), len(existing_df), len(non_existing_df))
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
                "prompt existing",
                "input embeds replace existing",
                "vanilla non-existing",
                "prompt non-existing",
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


class HiddenStatesPlotter:
    def __init__(
        self,
        vanilla_df: pd.DataFrame,
        prompt_df: pd.DataFrame,
        embedding_df: pd.DataFrame,
        vanilla_pcas: List[List[PCA]],
        prompt_pcas: List[List[PCA]],
        embedding_pcas: List[List[PCA]],
    ) -> None:
        self.vanilla_df = vanilla_df
        self.prompt_df = prompt_df
        self.embedding_df = embedding_df
        self.vanilla_pcas = vanilla_pcas
        self.prompt_pcas = prompt_pcas
        self.embedding_pcas = embedding_pcas
