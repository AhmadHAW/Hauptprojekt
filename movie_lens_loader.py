import os
import torch
from torch_geometric.data import download_url, extract_zip
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict

def row_to_prompt_datapoint(row):
    user_id = row["mappedUserId"]
    title = row["title"]
    genres = row["genres"]
    user_embedding = row["user_embedding"]
    movie_embedding = row["movie_embedding"]
    prompt = f"user: {user_id}, title: {title}, genres: {genres} user embedding: {user_embedding}, movie embedding: {movie_embedding}"
    return prompt

def row_to_vanilla_datapoint(row):
    user_id = row["mappedUserId"]
    title = row["title"]
    genres = row["genres"]
    prompt = f"user: {user_id}, title: {title}, genres: {genres}"
    return prompt

ROOT = "./data"                                                     #The root path where models and datasets are saved at.

GNN_PATH = f"{ROOT}/gnn"                                            #The path, where gnn datasets and models are saved at.
GNN_TRAIN_DATASET_PATH = f"{GNN_PATH}/train"                        #The path where the gnn training dataset is saved at.
GNN_TEST_DATASET_PATH = f"{GNN_PATH}/test"                          #The path where the gnn test dataset is saved at.
GNN_VAL_DATASET_PATH = f"{GNN_PATH}/dev"                            #The path where the gnn validation dataset is saved at.
GNN_COMPLETE_DATASET_PATH = f"{GNN_PATH}/complete"                  #The path where the complete gnn dataset is saved at.
GNN_MODEL_PATH = f"{GNN_PATH}/model.pth"                            #The path where the trained gnn model is saved at.

LLM_PATH = f"{ROOT}/llm"                                            #The path, where LLM datasets and models are saved at.
LLM_DATASET_PATH = f"{LLM_PATH}/dataset.csv"                        #The path where the LLM dataset is saved at.
LLM_PROMPT_PATH = f"{LLM_PATH}/prompt"
LLM_VANILLA_PATH = f"{LLM_PATH}/vanilla"
LLM_PROMPT_TRAINING_PATH = f"{LLM_PROMPT_PATH}/training"            #The path where the LLM training outputs are saved at.
LLM_VANILLA_TRAINING_PATH = f"{LLM_VANILLA_PATH}/training"          #The path where the LLM training outputs are saved at.
LLM_PROMPT_BEST_MODEL_PATH = f"{LLM_PROMPT_TRAINING_PATH}/best"     #The path where the best trained LLM model is saved at.
LLM_VANILLA_BEST_MODEL_PATH = f"{LLM_VANILLA_TRAINING_PATH}/best"   #The path where the best trained LLM model is saved at.
LLM_PROMPT_DATASET_PATH = f"{LLM_PATH}/prompt_dataset"                      #The path where the huggingface prompt dataset (tokenized) is saved at.
LLM_VANILLA_DATASET_PATH = f"{LLM_PATH}/vanilla_dataset"                    #The path where the huggingface vanilla dataset (tokenized) is saved at.

SUB_DIRS = [GNN_PATH, LLM_PATH, LLM_PROMPT_PATH, LLM_VANILLA_PATH, LLM_PROMPT_TRAINING_PATH, LLM_VANILLA_TRAINING_PATH]


class MovieLensLoader():
    '''
    The MovieLensLoader manages the original graph data set and the pre-processed data sets for GNN and LLM.
    '''

    def __init__(self, force_recompute: bool = False) -> None:
        '''
        The constructor allows general settings, like forcing to reload even tho there are datasets present.
        The preprocessing of the dataset can be read in detail in the original gnn link prediction tutorial of
        torch geometrics (https://colab.research.google.com/drive/1xpzn1Nvai1ygd_P5Yambc_oe4VBPK_ZT?usp=sharing#scrollTo=vit8xKCiXAue)

        Parameters
        __________
        force_recompute:bool
                        Whether to force reloading and recomputing datasets and values.
                        Default False -> Loads and computes only if missing.

        '''
        if not self.__data_present() or force_recompute:
            self.__create_dirs()            #Create dirs if not exist
            self.__download_dataset()       #download dataset if missing or forced to reload
            self.__map_to_unique()          #preprocessing step, where all node ids are mapped to a range of ids
            self.__generate_hetero_data()   #generate Torch Geometric HeteroDataset
            self.__split_data()             #split dataset into train, dev and test
            self.__generate_llm_dataset()   #generate pandas dataframe with prompts and labels for LLM
            self.__split_llm_dataset()      #split llm dataset according to the HeteroDataset split into train, dev, test and rest
        else:
            self.__load_datasets_from_disk()

    def __data_present(self) -> bool:
        '''Returns True, if GNN_TRAIN_DATASET_PATH, GNN_TEST_DATASET_PATH, GNN_VAL_DATASET_PATH, LLM_DATASET_PATH are files.'''
        return os.path.isfile(GNN_TRAIN_DATASET_PATH) and os.path.isfile(GNN_TEST_DATASET_PATH) and os.path.isfile(GNN_VAL_DATASET_PATH) and os.path.isfile(LLM_DATASET_PATH)

    def __create_dirs(self) -> None:
        for directory in SUB_DIRS:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def __load_datasets_from_disk(self):
        self.gnn_train_data = torch.load(GNN_TRAIN_DATASET_PATH)
        self.gnn_val_data = torch.load(GNN_VAL_DATASET_PATH)
        self.gnn_test_data = torch.load(GNN_TEST_DATASET_PATH)
        self.data = torch.load(GNN_COMPLETE_DATASET_PATH)
        self.llm_df = pd.read_csv(LLM_DATASET_PATH)


    def __download_dataset(self):
        '''See torch geometric tutorial'''
        movies_path = f'{ROOT}/ml-latest-small/movies.csv'
        ratings_path = f'{ROOT}/ml-latest-small/ratings.csv'
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        extract_zip(download_url(url, ROOT), ROOT)
        self.movies_df = pd.read_csv(movies_path, index_col='movieId')
        self.movies_llm_df = pd.read_csv(movies_path, index_col='movieId')
        self.movies_llm_df['genres'] = self.movies_llm_df['genres'].apply(lambda genres: list(genres.split("|")))
        self.genres = self.movies_df['genres'].str.get_dummies('|')
        self.movie_feat = torch.from_numpy(self.genres.values).to(torch.float)
        self.ratings_df = pd.read_csv(ratings_path)

    def __map_to_unique(self):
        '''See torch geometric tutorial'''
        # Create a mapping from unique user indices to range [0, num_user_nodes):
        self.unique_user_id = self.ratings_df['userId'].unique()
        self.unique_user_id = pd.DataFrame(data={
            'userId': self.unique_user_id,
            'mappedUserId': pd.RangeIndex(len(self.unique_user_id)),
        })
        # Create a mapping from unique movie indices to range [0, num_movie_nodes):
        self.unique_movie_id = pd.DataFrame(data={
            'movieId': self.movies_df.index,
            'mappedMovieId': pd.RangeIndex(len(self.movies_df)),
        })
        self.unique_llm_movie_id = pd.DataFrame(data={
            'mappedMovieId': pd.RangeIndex(len(self.movies_llm_df)),
            'title': self.movies_llm_df["title"],
            'genres': self.movies_llm_df["genres"]
        }).reset_index()

        # Perform merge to obtain the edges from users and movies:
        self.ratings_user_id = pd.merge(self.ratings_df['userId'], self.unique_user_id,
                                    left_on='userId', right_on='userId', how='left')
        self.ratings_user_id = torch.from_numpy(self.ratings_user_id['mappedUserId'].values)
        self.ratings_movie_id = pd.merge(self.ratings_df['movieId'], self.unique_movie_id,
                                    left_on='movieId', right_on='movieId', how='left')
        self.ratings_movie_id = torch.from_numpy(self.ratings_movie_id['mappedMovieId'].values)

        # With this, we are ready to construct our `edge_index` in COO format
        # following PyG semantics:
        self.edge_index_user_to_movie = torch.stack([self.ratings_user_id, self.ratings_movie_id], dim=0)
        assert self.edge_index_user_to_movie.size() == (2, 100836)
        self.llm_df = self.ratings_df.merge(self.unique_user_id, on="userId").merge(self.unique_llm_movie_id, on="movieId")[["mappedUserId", "mappedMovieId", "title", "genres"]]


    def __generate_hetero_data(self):
        '''See torch geometric tutorial'''
        self.data = HeteroData()

        # Save node indices:
        self.data["user"].node_id = torch.arange(len(self.unique_user_id))
        self.data["movie"].node_id = torch.arange(len(self.movies_df))

        # Add the node features and edge indices:
        self.data["movie"].x = self.movie_feat
        self.data["user", "rates", "movie"].edge_index = self.edge_index_user_to_movie

        # We also need to make sure to add the reverse edges from movies to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        transform = T.ToUndirected()
        self.data = transform(self.data)

        assert self.data.node_types == ["user", "movie"]
        assert self.data.edge_types == [("user", "rates", "movie"),
                                ("movie", "rev_rates", "user")]
        assert self.data["user"].num_nodes == 610
        assert self.data["user"].num_features == 0
        assert self.data["movie"].num_nodes == 9742
        assert self.data["movie"].num_features == 20
        assert self.data["user", "rates", "movie"].num_edges == 100836
        assert self.data["movie", "rev_rates", "user"].num_edges == 100836

    def __split_data(self):
        '''See torch geometric tutorial'''
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=("user", "rates", "movie"),
            rev_edge_types=("movie", "rev_rates", "user"),
            
        )
        self.gnn_train_data, self.gnn_val_data, self.gnn_test_data = transform(self.data)
        torch.save(self.gnn_train_data, GNN_TRAIN_DATASET_PATH)
        torch.save(self.gnn_val_data, GNN_VAL_DATASET_PATH)
        torch.save(self.gnn_test_data, GNN_TEST_DATASET_PATH)
        torch.save(self.data, GNN_COMPLETE_DATASET_PATH)
    
    def __generate_llm_dataset(self):
        '''
        This method produces prompts given the user id, movie id, movie title and genres.
        '''
        print("generate llm dataset...")
        self.llm_df["prompt"] = self.llm_df.apply(row_to_vanilla_datapoint, axis=1)

    def __is_in_split(self, edge_index, user_id, movie_id) -> bool:
        '''
        This methods returns the True if the current gnn dataset split contains the edge between given user and movie.
        '''
        test_tensor = torch.Tensor([user_id,movie_id])
        return len(torch.nonzero(torch.all(edge_index==test_tensor, dim=1)))>0
    
    def __find_split(self, row, train_edge_index, val_edge_index, test_edge_index) -> str:
        '''
        Returns the split train, test, val or rest if the given edge is found in either gnn dataset split.
        the datapoint is assigned to train if present or if overlapping between val and test in a 50/50 proportion. 
        '''
        user_id = row["mappedUserId"]
        movie_id = row["mappedMovieId"]
        if self.__is_in_split(train_edge_index, user_id, movie_id):
            split = "train"
        elif self.__is_in_split(val_edge_index, user_id, movie_id):
            split = "val" if self.__last == "test" else "test"
            self.__last = split
        elif self.__is_in_split(test_edge_index, user_id, movie_id):
            split = "test"
        else:
            split = "rest"
        return split

    def __split_llm_dataset(self):
        '''
        This method assigns all datapoints in the LLM dataset to the same split as they are found in the gnn dataset split.
        '''
        train_edge_index = self.gnn_train_data["user", "rates", "movie"]["edge_index"].T
        test_edge_index = self.gnn_val_data["user", "rates", "movie"]["edge_index"].T
        val_edge_index = self.gnn_test_data["user", "rates", "movie"]["edge_index"].T
        self.__last = "test"
        print("splitting LLM dataset")
        self.llm_df["split"] = self.llm_df.apply(lambda row: self.__find_split(row, train_edge_index, val_edge_index, test_edge_index), axis = 1)
        self.llm_df.to_csv(LLM_DATASET_PATH, index=False)

    def add_graph_embeddings(self, get_embedding_cb):
        '''
        This method computes the user node embedding and movie node embedding of all edges.
        '''
        def __add_graph_embeddings(row, get_embedding_cb):
            split = row["split"]
            user_id = row["mappedUserId"]
            movie_id = row["mappedMovieId"]
            data = self.gnn_train_data if split == "train" else self.gnn_val_data if split == "val" else self.gnn_test_data if split == "test" else self.data
            user_embedding, movie_embedding = get_embedding_cb(data, user_id, movie_id)
            row["user_embedding"] = user_embedding.detach().tolist()
            row["movie_embedding"] = movie_embedding.detach().tolist()
            return row

        self.llm_df = self.llm_df.apply(lambda row: __add_graph_embeddings(row, get_embedding_cb), axis = 1)
        self.llm_df.to_csv(LLM_DATASET_PATH, index=False)
    
    def __dataset_from_df(self, df):
        dataset_train = Dataset.from_pandas(df[df["split"] == "train"])
        dataset_val = Dataset.from_pandas(df[df["split"] == "val"])
        dataset_test = Dataset.from_pandas(df[df["split"] == "test"])
        return DatasetDict({
            "train": dataset_train,
            "val": dataset_val,
            "test": dataset_test,
        })


    def generate_prompt_embedding_dataset(self, tokenize_function = None, force_recompute = False):
        if os.path.exists(LLM_PROMPT_DATASET_PATH) and not force_recompute:
            dataset = datasets.load_from_disk(LLM_PROMPT_DATASET_PATH)
        else:
            assert "user_embedding" in self.llm_df
            assert "movie_embedding" in self.llm_df
            assert self.llm_df["user_embedding"].notna().all()
            assert self.llm_df["movie_embedding"].notna().all()
            llm_df = self.llm_df.copy(deep = True)
            llm_df["labels"] = 1
            llm_df["prompt"] = self.llm_df.apply(row_to_prompt_datapoint, axis=1)
            dataset = self.__dataset_from_df(llm_df)
            if tokenize_function:
                dataset = dataset.map(tokenize_function, batched = True)
            dataset.save_to_disk(LLM_PROMPT_DATASET_PATH)
        return dataset
    
    def generate_vanilla_dataset(self, tokenize_function = None, force_recompute = False):
        if os.path.exists(LLM_VANILLA_DATASET_PATH) and not force_recompute:
            dataset = datasets.load_from_disk(LLM_VANILLA_DATASET_PATH)
        else:
            llm_df = self.llm_df.copy(deep = True)
            llm_df["labels"] = 1
            llm_df["prompt"] = self.llm_df.apply(row_to_vanilla_datapoint, axis=1)
            dataset = self.__dataset_from_df(llm_df)
            if tokenize_function:
                dataset = dataset.map(tokenize_function, batched = True)
            dataset.save_to_disk(LLM_VANILLA_DATASET_PATH)
        return dataset
    
    def sample_prompt_datapoint(self, split = "val", get_embedding_cb = None, existing = True, tokenize_function = None):
        df = self.llm_df[self.llm_df["split"] == split]
        if existing:
            random_row = df.sample(1).iloc[0]
        else:
            assert get_embedding_cb
            dataset = self.gnn_train_data if split == "train" else self.gnn_val_data if split == "val" else self.gnn_test_data if split == "test" else self.data
            existing = True
            while existing:
                user_id = df["mappedUserId"].sample(1).iloc[0]
                random_row = df.sample(1).iloc[0]
                movie_id = random_row["mappedMovieId"]
                existing = ((self.llm_df["mappedMovieId"] == movie_id) & (self.llm_df["mappedUserId"] == user_id)).any()
                if not existing:
                    random_row = random_row.copy(deep= True)
                    random_row["mappedUserId"] = user_id
                    user_embedding, movie_embedding = get_embedding_cb(dataset, user_id, movie_id, split)
                    random_row["user_embedding"] = user_embedding.detach().tolist()
                    random_row["movie_embedding"] = movie_embedding.detach().tolist()
        prompt = row_to_prompt_datapoint(random_row)
        labels = 1 if existing else 0
        result = {"prompt": prompt, "labels": labels}
        if tokenize_function:
            return tokenize_function(result, return_pt = True)
        else:
            return result
        
    def sample_vanilla_datapoint(self, split = "val", existing = True, tokenize_function = None):
        df = self.llm_df[self.llm_df["split"] == split]
        if existing:
            random_row = df.sample(1).iloc[0]
        else:
            existing = True
            while existing:
                user_id = df["mappedUserId"].sample(1).iloc[0]
                random_row = df.sample(1).iloc[0]
                movie_id = random_row["mappedMovieId"]
                existing = ((self.llm_df["mappedMovieId"] == movie_id) & (self.llm_df["mappedUserId"] == user_id)).any()
                if not existing:
                    random_row = random_row.copy(deep= True)
                    random_row["mappedUserId"] = user_id
        prompt = row_to_prompt_datapoint(random_row)
        labels = 1 if existing else 0
        result = {"prompt": prompt, "labels": labels}
        if tokenize_function:
            return tokenize_function(result, return_pt = True)
        else:
            return result
        
    def replace_llm_df(self, df: pd.DataFrame):
        self.llm_df = df
        self.llm_df.to_csv(LLM_DATASET_PATH, index = False)
        
            


