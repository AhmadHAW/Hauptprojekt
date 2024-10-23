import os
from typing import Optional, List

from pandas import DataFrame, Series
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd

from dataset_manager import ROOT


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, output_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, output_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Define a 2-layer GNN computation graph.
        # Use a *single* `ReLU` non-linearity in-between.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    # Our final classifier applies the dot-product between source and destination
    # node embeddings to derive edge-level predictions:
    def forward(
        self, x_source: Tensor, x_target: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_source = x_source[edge_label_index[0]]
        edge_feat_target = x_target[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_source * edge_feat_target).sum(dim=-1)


class Model(torch.nn.Module):
    """
    From Tutorial:
    We are now ready to create our heterogeneous GNN.
    The GNN is responsible for learning enriched node representations from the surrounding subgraphs,
    which can be then used to derive edge-level predictions.
    For defining our heterogenous GNN, we make use of nn.SAGEConv and the nn.to_hetero() function,
    which transforms a GNN defined on homogeneous graphs to be applied on heterogeneous ones.

    In addition, we define a final link-level classifier, which simply takes both node embeddings of the link we are trying to predict,
    and applies a dot-product on them.

    As sources do not have any node-level information,
    we choose to learn their features jointly via a torch.nn.Embedding layer.
    In order to improve the expressiveness of target features, we do the same for target nodes,
    and simply add their shallow embeddings to the pre-defined genre features.
    """

    def __init__(self, hidden_channels, output_channels, data):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for sources and targets:
        self.target_lin = torch.nn.Linear(20, hidden_channels)
        self.source_emb = torch.nn.Embedding(data["source"].num_nodes, hidden_channels)
        self.target_emb = torch.nn.Embedding(data["target"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels, output_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.model_path = f"{ROOT}/gnn/model_{{}}.pth".format(output_channels)

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "source": self.source_emb(data["source"].node_id),
            "target": self.target_lin(data["target"].x)
            + self.target_emb(data["target"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["source"],
            x_dict["target"],
            data["source", "edge", "target"].edge_label_index,
        )

        return pred

    def forward_without_classifier(self, data: HeteroData) -> Tensor:
        x_dict = {
            "source": self.source_emb(data["source"].node_id),
            "target": self.target_lin(data["target"].x)
            + self.target_emb(data["target"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        return self.gnn(x_dict, data.edge_index_dict)


class GraphRepresentationGenerator:
    """
    The GraphRepresentationGenerator manages and trains a GNN model. A GNN model consists of an encoder and classifier.
    An encoder is a Grap Convolutional Network (GCN) with a 2-layer GNN computation graph and a single ReLU activation function in between.
    A classifier applies the dot-product between source and destination node embeddings to derive edge-level predictions.
    In addition to training, validating and saving the model,
    the GraphRepresentationGenerator provides a callback function for generating embeddings for given edges.
    Among other things, this function can help to generate non-existent edges on the fly.
    """

    def __init__(
        self,
        data: HeteroData,
        gnn_train_data: HeteroData,
        gnn_val_data: HeteroData,
        gnn_test_data: HeteroData,
        force_recompute: bool = False,
        kge_dimension: Optional[int] = None,
        hidden_channels: int = 64,
        device: Optional[str] = None,
    ) -> None:
        """
        The constructor of the GraphRepresentationGenerator initializes the model in the corresponding dimensions, the optimizer for the training and data set splits.
        Parameters
        __________
        data:                   HeteroData
                                The full gnn dataset with the splits "train", "val" and "test"
        force_recompute:        bool
                                Whether to force reloading and recomputing models.
                                Default False -> Loads and computes only if missing.
        kge_dimension:    int
                                output dimension of the gnn.
                                Default 4
        """
        data = data.clone()
        if not kge_dimension:
            kge_dimension = hidden_channels
        self.model_path = f"{ROOT}/gnn/model_{kge_dimension}.pth"
        self.model = Model(
            hidden_channels=hidden_channels,
            output_channels=kge_dimension,
            data=data,
            force_recompute=force_recompute,
        )
        # load if there is a trained model and not force_recompute
        if os.path.isfile(self.model_path) and not force_recompute:
            print("loading pretrained model")
            self.model.load_state_dict(torch.load(self.model_path))
        self.device = torch.device("cpu") if not device else torch.device(device)
        print(f"Device: '{self.device}'")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.gnn_train_data = gnn_train_data
        self.gnn_val_data = gnn_val_data
        self.gnn_test_data = gnn_test_data
        self.kge_dimension = kge_dimension
        self.source_embedding_path = (
            f"{ROOT}/gnn/model_{kge_dimension}_source_embedding_split_{{}}.pth"
        )
        self.target_embedding_path = (
            f"{ROOT}/gnn/model_{kge_dimension}_target_embedding_split_{{}}.pth"
        )
        self.force_recompute = force_recompute

    def train_model(self, data, epochs, batch_size=64):
        """
        From Tutorial
        We are now ready to create a mini-batch loader that will generate subgraphs that can be used as input into our GNN.
        While this step is not strictly necessary for small-scale graphs,
        it is absolutely necessary to apply GNNs on larger graphs that do not fit onto GPU memory otherwise.
        Here, we make use of the loader.LinkNeighborLoader which samples multiple hops from both ends of a link and creates a subgraph from it.
        Here, edge_label_index serves as the "seed links" to start sampling from.

        Training our GNN is then similar to training any PyTorch model.
        We move the model to the desired device, and initialize an optimizer that takes care of adjusting model parameters
        via stochastic gradient descent.

        The training loop then iteedge over our mini-batches, applies the forward computation of the model,
        computes the loss from ground-truth labels and obtained predictions (here we make use of binary cross entropy),
        and adjusts model parameters via back-propagation and stochastic gradient descent.
        """

        # Define seed edges:
        edge_label_index = data["source", "edge", "target"].edge_label_index
        edge_label = data["source", "edge", "target"].edge_label

        # In the first hop, we sample at most 20 neighbors.
        # In the second hop, we sample at most 10 neighbors.
        # In addition, during training, we want to sample negative edges on-the-fly with
        # a ratio of 2:1.
        train_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[20, 10],
            neg_sampling_ratio=2.0,
            edge_label_index=(("source", "edge", "target"), edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=True,
        )
        for epoch in range(1, epochs + 1):
            total_loss = total_examples = 0
            for sampled_data in tqdm.tqdm(train_loader):
                self.optimizer.zero_grad()
                # Move `sampled_data` to the respective `device`
                sampled_data.to(self.device)
                # Run `forward` pass of the model
                pred = self.model(sampled_data)
                ground_truth = sampled_data["source", "edge", "target"].edge_label
                # Apply binary cross entropy
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

                loss.backward()
                self.optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
        torch.save(self.model.to(device="cpu").state_dict(), self.model_path)
        self.model.to(self.device)

    def __link_neighbor_sampling(self, data, source_ids, target_ids):
        edge_label_index = torch.tensor([source_ids, target_ids])
        edge_label = torch.ones(len(source_ids))

        loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[20, 10],
            edge_label_index=(("source", "edge", "target"), edge_label_index),
            edge_label=edge_label,
            batch_size=int(len(source_ids) / 4),
            shuffle=False,
        )
        all_sampled_data = []
        for sampled_data in tqdm.tqdm(loader, disable=True):
            all_sampled_data.append(sampled_data)
        del loader
        return all_sampled_data

    def get_embedding(self, data: HeteroData, source_id: int, target_id: int):
        """
        This method is used as a callback function and produces the KGE from given source node and target node.
        For that a subgraph of the neighbohood is generated and then applied to the GNN. Afterwards only the embeddings of
        given source and target nodes are returned.
        """
        sampled_data = self.__link_neighbor_sampling(data, source_id, target_id)
        assert isinstance(sampled_data, HeteroData)
        assert sampled_data is not None
        sampled_data.to(self.device)  # type: ignore
        embeddings = self.model.forward_without_classifier(sampled_data)
        source_node_id_index = (
            (sampled_data["source"].n_id == source_id).nonzero(as_tuple=True)[0].item()
        )
        target_node_id_index = (
            (sampled_data["target"].n_id == target_id).nonzero(as_tuple=True)[0].item()
        )
        source_embedding = embeddings["source"][source_node_id_index]  # type: ignore
        target_embedding = embeddings["target"][target_node_id_index]  # type: ignore
        return source_embedding, target_embedding

    def get_embeddings(
        self, data: HeteroData, source_ids: List[int], target_ids: List[int]
    ):
        """
        This method is used as a callback function and produces the KGE from given source node and target node.
        For that a subgraph of the neighbohood is generated and then applied to the GNN. Afterwards only the embeddings of
        given source and target nodes are returned.
        """
        batch_sampled_data = self.__link_neighbor_sampling(data, source_ids, target_ids)
        all_source_node_embeddings = []
        all_target_node_embeddings = []
        for sampled_data in batch_sampled_data:
            assert isinstance(sampled_data, HeteroData)
            assert sampled_data is not None
            # sampled_data.to(self.device)  # type: ignore
            embeddings = self.model.forward_without_classifier(sampled_data)
            edge_label_index = sampled_data[
                "source", "edge", "target"
            ].edge_label_index  # Shape: [2, num_edges]

            # Extract source and target node embeddings from the computed embeddings
            source_node_embeddings = embeddings["source"][edge_label_index[0]]
            all_source_node_embeddings.append(source_node_embeddings)
            target_node_embeddings = embeddings["target"][edge_label_index[1]]
            all_target_node_embeddings.append(target_node_embeddings)
        all_source_node_embeddings = torch.concat(all_source_node_embeddings)
        all_target_node_embeddings = torch.concat(all_target_node_embeddings)
        # Return the embeddings for the entire batch
        return all_source_node_embeddings, all_target_node_embeddings

    def __get_embedding(self, row: Series) -> Series:
        split = row["split"]
        data = (
            self.gnn_train_data
            if split == "train"
            else self.gnn_val_data
            if split == "val"
            else self.gnn_test_data
            if split == "test"
            else self.gnn_train_data
        )
        source_id = row["source_id"]
        target_id = row["target_id"]
        source_embedding, target_embedding = self.get_embedding(
            data, source_id, target_id
        )
        row = Series(
            {
                "source_embedding": source_embedding.to("cpu").detach().tolist(),
                "target_embedding": target_embedding.to("cpu").detach().tolist(),
            }
        )
        return row

    def get_saved_embeddings(self, model: str) -> Optional[DataFrame]:
        source_path = f"{ROOT}/gnn/{model}_source_embedding.pt"
        if not os.path.exists(source_path):
            source_embeddings = []
            target_embeddings = []
            for split in ["train", "test", "val"]:
                source_path = f"{ROOT}/gnn/{model}_source_embedding_{split}.pth"
                target_path = f"{ROOT}/gnn/{model}_target_embedding_{split}.pth"
                if not os.path.exists(source_path) or not os.path.exists(source_path):
                    return None
                else:
                    source_embeddings.append(torch.load(source_path))
                    target_embeddings.append(torch.load(target_path))
                row = DataFrame(
                    {
                        "source_embedding": torch.stack(source_embeddings).tolist(),
                        "target_embedding": torch.stack(target_embeddings).tolist(),
                    }
                )
            return row
        source_embeddings = torch.load(source_path)
        target_path = f"{ROOT}/gnn/{model}_target_embedding.pt"
        if not os.path.exists(target_path):
            return None
        target_embeddings = torch.load(target_path)
        row = DataFrame(
            {
                "source_embedding": source_embeddings.to("cpu").detach().tolist(),
                "target_embedding": target_embeddings.to("cpu").detach().tolist(),
            }
        )
        return row

    def generate_embeddings(self, llm_df: DataFrame) -> DataFrame:
        """
        This method passes all edges (source - target) to the GNN to produce source and target embeddings.
        Parameters
        __________
        llm_df:         DataFrame
                        The dataframe of the dataset with all NL features
        """

        # produce the embeddings for all edges
        print(f"Computing embeddings for embedding dimension {self.kge_dimension}.")
        df_splits = []
        for split in ["train", "val", "test"]:
            df_split = llm_df.loc[llm_df["split"] == split][["source_id", "target_id"]]
            source_ids = df_split["source_id"].to_list()
            target_ids = df_split["target_id"].to_list()
            data = (
                self.gnn_train_data
                if split == "train"
                else self.gnn_val_data
                if split == "val"
                else self.gnn_test_data
                if split == "test"
                else self.gnn_train_data
            )
            source_embeddings, target_embeddings = self.get_embeddings(
                data, source_ids, target_ids
            )
            df_split["source_embedding"] = source_embeddings.to("cpu").detach().tolist()
            df_split["target_embedding"] = target_embeddings.to("cpu").detach().tolist()
            df_splits.append(df_split)

        df_splits = pd.concat(df_splits)
        llm_df = llm_df.merge(df_splits, on=["source_id", "target_id"])[
            ["source_embedding", "target_embedding"]
        ]

        return llm_df

    def validate_model(self, data, batch_size=64):
        # Define the validation seed edges:
        edge_label_index = data["source", "edge", "target"].edge_label_index
        edge_label = data["source", "edge", "target"].edge_label

        val_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[20, 10],
            edge_label_index=(("source", "edge", "target"), edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=False,
        )
        preds = []
        ground_truths = []
        for sampled_data in tqdm.tqdm(val_loader):
            with torch.no_grad():
                sampled_data = sampled_data.to(self.device)
                preds.append(self.model(sampled_data))
                ground_truths.append(
                    sampled_data["source", "edge", "target"].edge_label
                )

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        auc = roc_auc_score(ground_truth, pred)
        print()
        print(f"Validation AUC: {auc:.4f}")

    def save_model(self):
        torch.save(self.model.to(device="cpu").gnn.state_dict(), self.model_path)
