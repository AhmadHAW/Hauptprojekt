import os
import joblib

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

from movie_lens_loader import GNN_MODEL_PATH, MovieLensLoader, GNN_PATH


GNN_USER_TRAIN_PCA = f"{GNN_PATH}/pca_user_train.pkl"               
GNN_USER_TEST_PCA = f"{GNN_PATH}/pca_user_test.pkl"
GNN_USER_VAL_PCA = f"{GNN_PATH}/pca_user_val.pkl"
GNN_USER_REST_PCA = f"{GNN_PATH}/pca_user_rest.pkl"

GNN_MOVIE_TRAIN_PCA = f"{GNN_PATH}/pca_movie_train.pkl"               
GNN_MOVIE_TEST_PCA = f"{GNN_PATH}/pca_movie_test.pkl"
GNN_MOVIE_VAL_PCA = f"{GNN_PATH}/pca_movie_val.pkl"
GNN_MOVIE_REST_PCA = f"{GNN_PATH}/pca_movie_rest.pkl"

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data, force_recompute = False):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        #load if there is a trained model and not force_recompute
        if os.path.isfile(GNN_MODEL_PATH and not force_recompute):
            self.gnn.load_state_dict(torch.load(GNN_MODEL_PATH))

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
    def __init__(self, data, force_recompute = False) -> None:
        self.model = Model(hidden_channels=64, data=data, force_recompute = force_recompute)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: '{self.device}'")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.gnn_train_data = data["train"]
        self.gnn_val_data = data["val"]
        self.gnn_test_data = data["test"]
        self.force_recompute = force_recompute
        self.__load_pcas_if_exists()

    def __load_pcas_if_exists(self):
        if os.path.exists(GNN_USER_TRAIN_PCA) and not self.force_recompute:
            self.user_pca_train = joblib.load(GNN_USER_TRAIN_PCA)
        else:
            self.user_pca_train = None
        if os.path.exists(GNN_USER_VAL_PCA) and not self.force_recompute:
            self.user_pca_val = joblib.load(GNN_USER_VAL_PCA)
        else:
            self.user_pca_val = None
        if os.path.exists(GNN_USER_TEST_PCA) and not self.force_recompute:
            self.user_pca_test = joblib.load(GNN_USER_TEST_PCA)
        else:
            self.user_pca_test = None
        if os.path.exists(GNN_USER_REST_PCA) and not self.force_recompute:
            self.user_pca_rest = joblib.load(GNN_USER_REST_PCA)
        else:
            self.user_pca_rest = None

        if os.path.exists(GNN_MOVIE_TRAIN_PCA) and not self.force_recompute:
            self.movie_pca_train = joblib.load(GNN_MOVIE_TRAIN_PCA)
        else:
            self.movie_pca_train = None
        if os.path.exists(GNN_MOVIE_VAL_PCA) and not self.force_recompute:
            self.movie_pca_val = joblib.load(GNN_MOVIE_VAL_PCA)
        else:
            self.movie_pca_val = None
        if os.path.exists(GNN_MOVIE_TEST_PCA) and not self.force_recompute:
            self.movie_pca_test = joblib.load(GNN_MOVIE_TEST_PCA)
        else:
            self.movie_pca_test = None
        if os.path.exists(GNN_MOVIE_REST_PCA) and not self.force_recompute:
            self.movie_pca_rest = joblib.load(GNN_MOVIE_REST_PCA)
        else:
            self.movie_pca_rest = None

    def train_model(self, data, epochs):
        # In the first hop, we sample at most 20 neighbors.
        # In the second hop, we sample at most 10 neighbors.
        # In addition, during training, we want to sample negative edges on-the-fly with
        # a ratio of 2:1.
        # We can make use of the `loader.LinkNeighborLoader` from PyG:

        # Define seed edges:
        edge_label_index = data["user", "rates", "movie"].edge_label_index
        edge_label = data["user", "rates", "movie"].edge_label

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

                sampled_data.to(self.device)
                pred = self.model(sampled_data)
                ground_truth = sampled_data["user", "rates", "movie"].edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

                loss.backward()
                self.optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
            torch.save(self.model.gnn.state_dict(), GNN_MODEL_PATH)

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
        
    def get_embedding(self, data, user_id, movie_id, add_pca = False, split = None):
        sampled_data = self.__link_neighbor_sampling(data, user_id, movie_id)
        embeddings = self.model.forward_without_classifier(sampled_data)
        user_node_id_index = (sampled_data['user'].n_id == user_id).nonzero(as_tuple=True)[0].item()
        movie_node_id_index = (sampled_data['movie'].n_id == movie_id).nonzero(as_tuple=True)[0].item()
        user_embedding = embeddings["user"][user_node_id_index]
        movie_embedding = embeddings["movie"][movie_node_id_index]
        if add_pca:
            assert split
            if split == "train":
                user_pca = self.user_pca_train
                movie_pca = self.movie_pca_train
            elif split == "val":
                user_pca = self.user_pca_val
                movie_pca = self.movie_pca_val
            elif split == "test":
                user_pca = self.user_pca_test
                movie_pca = self.movie_pca_test
            else:
                user_pca = self.user_pca_rest
                movie_pca = self.movie_pca_rest

            pca_2_user_embedding = user_pca.fit_transform(user_embedding.reshape(1, -1))
            pca_2_movie_embedding = movie_pca.fit_transform(movie_embedding.reshape(1, -1))
        else:
            pca_2_user_embedding = None
            pca_2_movie_embedding = None
        return user_embedding, movie_embedding, pca_2_user_embedding, pca_2_movie_embedding
    
    def get_embeddings(self, movie_lens_loader: MovieLensLoader):
        '''
        This method first passes all edges (user - movie) to the GNN to produce user and movie embeddings.
        Then all embeddings are passed to a PCA and are reduced to 2 dimensions.
        '''
        def __get_embedding(row, movie_lens_loader: MovieLensLoader):
            split = row["split"]
            data = movie_lens_loader.gnn_train_data if split == "train" else movie_lens_loader.gnn_val_data if split == "val" else movie_lens_loader.gnn_test_data if split == "test" else movie_lens_loader.gnn_train_data
            user_id = row["mappedUserId"]
            movie_id = row["mappedMovieId"]
            user_embedding, movie_embedding, _, _ = self.get_embedding(data, user_id, movie_id)
            row["user_embedding"] = user_embedding.detach().tolist()
            row["movie_embedding"] = movie_embedding.detach().tolist()
            return row
        
        #produce the embeddings for all edges
        df = movie_lens_loader.llm_df.apply(lambda row: __get_embedding(row, movie_lens_loader), axis = 1)
        #compress the embeddings of all user embeddings to two dimensions
        df[f"pca_2_user_embedding"] = ""
        for split in ["train", "test", "dev", "rest"]:
            condition = df['split'] == split
            user_embeddings = list(df[df["split"] == split]["user_embedding"].values)
            if split == "train":
                if not self.user_pca_train or self.force_recompute:
                    self.user_pca_train = PCA(n_components=2)  # Reduce to 2 dimensions
                pca = self.user_pca_train
            if split == "val":
                if not self.user_pca_val or self.force_recompute:
                    self.user_pca_val = PCA(n_components=2)  # Reduce to 2 dimensions
                pca = self.user_pca_val
            if split == "test":
                if not self.user_pca_test or self.force_recompute:
                    self.user_pca_test = PCA(n_components=2)  # Reduce to 2 dimensions
                pca = self.user_pca_test
            if split == "rest":
                if not self.user_pca_rest or self.force_recompute:
                    self.user_pca_rest = PCA(n_components=2)  # Reduce to 2 dimensions
                pca = self.user_pca_rest
            pca_2_user_embeddings = pca.fit_transform(user_embeddings).squeeze().tolist()
            df.loc[condition, 'pca_2_user_embedding'] = pca_2_user_embeddings

        #compress the embeddings of all movie embeddings to two dimensions
        df[f"pca_2_movie_embedding"] = ""
        for split in ["train", "test", "dev", "rest"]:
            condition = df['split'] == split
            movie_embeddings = list(df[df["split"] == split]["movie_embedding"].values)
            if split == "train":
                if not self.movie_pca_train or self.force_recompute:
                    self.movie_pca_train = PCA(n_components=2)  # Reduce to 2 dimensions
                pca = self.movie_pca_train
            if split == "val":
                if not self.movie_pca_val or self.force_recompute:
                    self.movie_pca_val = PCA(n_components=2)  # Reduce to 2 dimensions
                pca = self.movie_pca_val
            if split == "test":
                if not self.movie_pca_test or self.force_recompute:
                    self.movie_pca_test = PCA(n_components=2)  # Reduce to 2 dimensions
                pca = self.movie_pca_test
            if split == "rest":
                if not self.movie_pca_rest or self.force_recompute:
                    self.movie_pca_rest = PCA(n_components=2)  # Reduce to 2 dimensions
                pca = self.movie_pca_rest
            pca_2_movie_embeddings = pca.fit_transform(movie_embeddings).squeeze().tolist()
            df.loc[condition, 'pca_2_movie_embedding'] = pca_2_movie_embeddings
        
            
        
        # Save all PCAs models
        self.__save_all_pcas()

        movie_lens_loader.replace_llm_df(df)

    def __save_all_pcas(self):
        joblib.dump(self.user_pca_train, GNN_USER_TRAIN_PCA)
        joblib.dump(self.user_pca_test, GNN_USER_TEST_PCA)
        joblib.dump(self.user_pca_val, GNN_USER_VAL_PCA)
        joblib.dump(self.user_pca_rest, GNN_USER_REST_PCA)

        joblib.dump(self.movie_pca_train, GNN_MOVIE_TRAIN_PCA)
        joblib.dump(self.movie_pca_test, GNN_MOVIE_TEST_PCA)
        joblib.dump(self.movie_pca_val, GNN_MOVIE_VAL_PCA)
        joblib.dump(self.movie_pca_rest, GNN_MOVIE_REST_PCA)
        
        

        
        

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
        torch.save(self.model.gnn.state_dict(), GNN_MODEL_PATH)