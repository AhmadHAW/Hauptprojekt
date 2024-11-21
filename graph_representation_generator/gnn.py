import torch
from torch import Tensor
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData

from dataset_manager.kg_manager import ROOT


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
        # print(data["source"].node_id.device)
        # print(data["target"].x.device)
        # print(data["target"].node_id.device)
        x_dict = {
            "source": self.source_emb(data["source"].node_id),
            "target": self.target_lin(data["target"].x)
            + self.target_emb(data["target"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        return self.gnn(x_dict, data.edge_index_dict)
