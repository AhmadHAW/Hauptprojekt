import os
import joblib
from typing import Optional

from pandas import DataFrame
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

from movie_lens_loader import MovieLensLoader, GNN_PATH

GNN_MODEL_PATH = f"{GNN_PATH}/model_{{}}.pth"


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
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class Model(torch.nn.Module):
    '''
    From Tutorial:
    We are now ready to create our heterogeneous GNN.
    The GNN is responsible for learning enriched node representations from the surrounding subgraphs,
    which can be then used to derive edge-level predictions.
    For defining our heterogenous GNN, we make use of nn.SAGEConv and the nn.to_hetero() function,
    which transforms a GNN defined on homogeneous graphs to be applied on heterogeneous ones.

    In addition, we define a final link-level classifier, which simply takes both node embeddings of the link we are trying to predict,
    and applies a dot-product on them.

    As users do not have any node-level information,
    we choose to learn their features jointly via a torch.nn.Embedding layer.
    In order to improve the expressiveness of movie features, we do the same for movie nodes,
    and simply add their shallow embeddings to the pre-defined genre features.
    '''
    def __init__(self, hidden_channels, output_channels, data, force_recompute = False):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels, output_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.model_path = GNN_MODEL_PATH.format(output_channels)

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )

        return pred
    
    def forward_without_classifier(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        return self.gnn(x_dict, data.edge_index_dict)
    
class GNNTrainer():
    '''
    The GNNTrainer manages and trains a GNN model. A GNN model consists of an encoder and classifier.
    An encoder is a Grap Convolutional Network (GCN) with a 2-layer GNN computation graph and a single ReLU activation function in between.
    A classifier applies the dot-product between source and destination node embeddings to derive edge-level predictions.
    In addition to training, validating and saving the model,
    the GNNTrainer provides a callback function for generating embeddings for given edges.
    Among other things, this function can help to generate non-existent edges on the fly.
    '''
    def __init__(self, data: HeteroData, force_recompute: bool = False, kge_dimension: Optional[int] = None, hidden_channels: int = 64) -> None:
        '''
        The constructor of the GNNTrainer initializes the model in the corresponding dimensions, the optimizer for the training and data set splits.
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
        '''
        data = data.clone()
        if not kge_dimension:
            kge_dimension = hidden_channels
        self.model_path = GNN_MODEL_PATH.format(kge_dimension)
        self.model = Model(hidden_channels=hidden_channels, output_channels = kge_dimension, data=data, force_recompute = force_recompute)
        #load if there is a trained model and not force_recompute
        if os.path.isfile(self.model_path) and not force_recompute:
            print("loading pretrained model")
            self.model.load_state_dict(torch.load(self.model_path))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: '{self.device}'")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.gnn_train_data = data["train"]
        self.gnn_val_data = data["val"]
        self.gnn_test_data = data["test"]
        self.kge_dimension = kge_dimension
        self.force_recompute = force_recompute

    def train_model(self, data, epochs):
        '''
        From Tutorial
        We are now ready to create a mini-batch loader that will generate subgraphs that can be used as input into our GNN.
        While this step is not strictly necessary for small-scale graphs,
        it is absolutely necessary to apply GNNs on larger graphs that do not fit onto GPU memory otherwise.
        Here, we make use of the loader.LinkNeighborLoader which samples multiple hops from both ends of a link and creates a subgraph from it.
        Here, edge_label_index serves as the "seed links" to start sampling from.

        Training our GNN is then similar to training any PyTorch model.
        We move the model to the desired device, and initialize an optimizer that takes care of adjusting model parameters
        via stochastic gradient descent.

        The training loop then iterates over our mini-batches, applies the forward computation of the model,
        computes the loss from ground-truth labels and obtained predictions (here we make use of binary cross entropy),
        and adjusts model parameters via back-propagation and stochastic gradient descent.
        '''

        # Define seed edges:
        edge_label_index = data["user", "rates", "movie"].edge_label_index
        edge_label = data["user", "rates", "movie"].edge_label

        # In the first hop, we sample at most 20 neighbors.
        # In the second hop, we sample at most 10 neighbors.
        # In addition, during training, we want to sample negative edges on-the-fly with
        # a ratio of 2:1.
        train_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[20, 10],
            neg_sampling_ratio=2.0,
            edge_label_index=(("user", "rates", "movie"), edge_label_index),
            edge_label=edge_label,
            batch_size=128,
            shuffle=True,
        )
        for epoch in range(1, epochs + 1):
            total_loss = total_examples = 0
            for sampled_data in tqdm.tqdm(train_loader):
                self.optimizer.zero_grad()
                # Move `sampled_data` to the respective `device`
                sampled_data.to(self.device)
                #Run `forward` pass of the model
                pred = self.model(sampled_data)
                ground_truth = sampled_data["user", "rates", "movie"].edge_label
                # Apply binary cross entropy
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

                loss.backward()
                self.optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
            torch.save(self.model.to(device = "cpu").state_dict(), self.model_path)

    def __link_neighbor_sampling(self, data, user_id, movie_id):
        edge_label_index = torch.tensor([[user_id], [movie_id]])
        edge_label = torch.tensor([1])

        loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[20, 10],
            edge_label_index=(("user", "rates", "movie"), edge_label_index),
            edge_label=edge_label,
            batch_size=1,
            shuffle=False,
        )
        for sampled_data in tqdm.tqdm(loader, disable=True):
            return sampled_data
        
    def get_embedding(self, data: HeteroData, user_id: int, movie_id:int):
        '''
        This method is used as a callback function and produces the KGE from given user node and movie node.
        For that a subgraph of the neighbohood is generated and then applied to the GNN. Afterwards only the embeddings of
        given user and movie nodes are returned.
        '''
        sampled_data = self.__link_neighbor_sampling(data, user_id, movie_id).to(self.device)
        embeddings = self.model.forward_without_classifier(sampled_data)
        user_node_id_index = (sampled_data['user'].n_id == user_id).nonzero(as_tuple=True)[0].item()
        movie_node_id_index = (sampled_data['movie'].n_id == movie_id).nonzero(as_tuple=True)[0].item()
        user_embedding = embeddings["user"][user_node_id_index]
        movie_embedding = embeddings["movie"][movie_node_id_index]
        return user_embedding, movie_embedding
    
    def __are_embeddings_saved(self, df):
        if not f"user_embedding_{self.kge_dimension}" in df.columns:
            return False
        if not ((df[f"user_embedding_{self.kge_dimension}"] != "") | (df[f"user_embedding_{self.kge_dimension}"].notna())).all():
            return False
        if not f"movie_embedding_{self.kge_dimension}" in df.columns:
            return False
        if not ((df[f"movie_embedding_{self.kge_dimension}"] != "") | (df[f"movie_embedding_{self.kge_dimension}"].notna())).all():
            return False

        return True

    
    def get_embeddings(self, movie_lens_loader: MovieLensLoader, force_recompute: bool = False):
        '''
        This method passes all edges (user - movie) to the GNN to produce user and movie embeddings.
        Parameters
        __________
        movie_lens_loader:          MovieLensLoader
                                    See movieLensLoader in movie_lens_loader.py
        force_recompute:            bool
                                    Whether to force reloading and recomputing datasets and values.
                                    Default False -> Loads and computes only if missing.
        '''
        def __get_embedding(row, movie_lens_loader: MovieLensLoader):
            split = row["split"]
            data = movie_lens_loader.gnn_train_data if split == "train" else movie_lens_loader.gnn_val_data if split == "val" else movie_lens_loader.gnn_test_data if split == "test" else movie_lens_loader.gnn_train_data
            user_id = row["mappedUserId"]
            movie_id = row["mappedMovieId"]
            user_embedding, movie_embedding = self.get_embedding(data, user_id, movie_id)
            row[f"user_embedding_{self.kge_dimension}"] = user_embedding
            row[f"movie_embedding_{self.kge_dimension}"] = movie_embedding
            return row
        df = movie_lens_loader.llm_df
        if not self.__are_embeddings_saved(df) or force_recompute:
            #produce the embeddings for all edges
            print(f"Computing embeddings for embedding dimension {self.kge_dimension}.")
            df = movie_lens_loader.llm_df.apply(lambda row: __get_embedding(row, movie_lens_loader), axis = 1)
            #save new embeddings      
            movie_lens_loader.replace_llm_df(df)        

        
        

    def validate_model(self, data):
        # Define the validation seed edges:
        edge_label_index = data["user", "rates", "movie"].edge_label_index
        edge_label = data["user", "rates", "movie"].edge_label

        val_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[20, 10],
            edge_label_index=(("user", "rates", "movie"), edge_label_index),
            edge_label=edge_label,
            batch_size=3 * 128,
            shuffle=False,
        )
        preds = []
        ground_truths = []
        for sampled_data in tqdm.tqdm(val_loader):
            with torch.no_grad():
                sampled_data = sampled_data.to(self.device)
                preds.append(self.model(sampled_data))
                ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        auc = roc_auc_score(ground_truth, pred)
        print()
        print(f"Validation AUC: {auc:.4f}")

    def save_model(self):
        torch.save(self.model.to(device = "cpu").gnn.state_dict(), self.model_path)