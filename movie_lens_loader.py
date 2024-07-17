import os
from typing import List, Tuple, Callable, Optional, Dict, Union

import torch
from torch_geometric.data import download_url, extract_zip
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict

def row_to_prompt_datapoint(row, kgeg_dimension):
    user_id = row["mappedUserId"]
    title = row["title"]
    genres = row["genres"]
    user_embedding = row[f"user_embedding_{kgeg_dimension}"]
    movie_embedding = row[f"movie_embedding_{kgeg_dimension}"]
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
GNN_VAL_DATASET_PATH = f"{GNN_PATH}/val"                            #The path where the gnn validation dataset is saved at.
GNN_COMPLETE_DATASET_PATH = f"{GNN_PATH}/complete"                  #The path where the complete gnn dataset is saved at.

LLM_PATH = f"{ROOT}/llm"                                        #The path, where LLM datasets and models are saved at.
LLM_DATASET_PATH = f"{LLM_PATH}/dataset.csv"                        #The path where the LLM dataset is saved at.
LLM_MODEL_DIMENSION_PATH = f"{LLM_PATH}/{{}}"
LLM_PROMPT_PATH = f"{LLM_MODEL_DIMENSION_PATH}/prompt"
LLM_VANILLA_PATH = f"{LLM_PATH}/vanilla"
LLM_PROMPT_TRAINING_PATH = f"{LLM_PROMPT_PATH}/training"            #The path where the LLM training outputs are saved at.
LLM_VANILLA_TRAINING_PATH = f"{LLM_VANILLA_PATH}/training"          #The path where the LLM training outputs are saved at.
LLM_PROMPT_BEST_MODEL_PATH = f"{LLM_PROMPT_TRAINING_PATH}/best"     #The path where the best trained LLM model is saved at.
LLM_VANILLA_BEST_MODEL_PATH = f"{LLM_VANILLA_TRAINING_PATH}/best"   #The path where the best trained LLM model is saved at.
LLM_PROMPT_DATASET_PATH = f"{LLM_MODEL_DIMENSION_PATH}/prompt_dataset"                      #The path where the huggingface prompt dataset (tokenized) is saved at.
LLM_VANILLA_DATASET_PATH = f"{LLM_PATH}/vanilla_dataset"                    #The path where the huggingface vanilla dataset (tokenized) is saved at.

SUB_DIRS = [GNN_PATH, LLM_VANILLA_PATH]


class MovieLensLoader():
    '''
    The MovieLensLoader manages the original graph data set and the pre-processed data sets for GNNs and LLMs.
    '''

    def __init__(self, force_recompute: bool = False, kge_dimensions: List[int] = [4]) -> None:
        '''
        The constructor allows general settings, like forcing to reload even tho there are datasets present.
        The preprocessing of the dataset can be read in detail in the original gnn link prediction tutorial of
        torch geometrics (https://colab.research.google.com/drive/1xpzn1Nvai1ygd_P5Yambc_oe4VBPK_ZT?usp=sharing#scrollTo=vit8xKCiXAue)

        Parameters
        __________
        force_recompute:            bool
                                    Whether to force reloading and recomputing datasets and values.
                                    Default False -> Loads and computes only if missing.
        kge_dimensions:             List[int]
                                    List of kge_dimensions that we expect to train models with.
                                    Default List[4]


        '''
        if not self.__data_present() or force_recompute:
            #Create dir system if not exist
            self.__create_dirs(kge_dimensions)
            #download dataset if missing or forced to reload
            movies_df, movies_llm_df, movie_feat, ratings_df = self.__download_dataset()
            #preprocessing step, where all node ids are mapped to a range of ids
            self.llm_df = self.__map_to_unique(movies_df, movies_llm_df, ratings_df)
            #generate Torch Geometric HeteroDataset
            self.data = self.__generate_hetero_data(movies_df, movie_feat)  
            #split HeteroDataset dataset into train, dev and test
            self.__split_data()
            #generate pandas dataframe with prompts and labels for LLM
            self.__generate_llm_dataset()
            #split llm dataset according to the HeteroDataset split into train, dev, test and rest
            self.__split_llm_dataset()
        else:
            self.__load_datasets_from_disk()

    def __data_present(self) -> bool:
        '''Returns True, if GNN_TRAIN_DATASET_PATH, GNN_TEST_DATASET_PATH, GNN_VAL_DATASET_PATH, LLM_DATASET_PATH are files.'''
        return os.path.isfile(GNN_TRAIN_DATASET_PATH) and os.path.isfile(GNN_TEST_DATASET_PATH) and os.path.isfile(GNN_VAL_DATASET_PATH) and os.path.isfile(LLM_DATASET_PATH)

    def __create_dirs(self, kge_dimensions: int) -> None:
        '''
        Create dir system if not exist
        '''
        for kge_dimension in kge_dimensions:
            directory = LLM_PROMPT_PATH.format(kge_dimension)
            if not os.path.exists(directory):
                os.makedirs(directory)
        for directory in SUB_DIRS:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def __load_datasets_from_disk(self):
        self.gnn_train_data = torch.load(GNN_TRAIN_DATASET_PATH)
        self.gnn_val_data = torch.load(GNN_VAL_DATASET_PATH)
        self.gnn_test_data = torch.load(GNN_TEST_DATASET_PATH)
        self.data = torch.load(GNN_COMPLETE_DATASET_PATH)
        self.llm_df = pd.read_csv(LLM_DATASET_PATH)


    def __download_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''Downloads the https://files.grouplens.org/datasets/movielens/ml-latest-small.zip dataset and
        extracts the content into ROOT//ml-latest-small/'''
        movies_path = f'{ROOT}/ml-latest-small/movies.csv'
        ratings_path = f'{ROOT}/ml-latest-small/ratings.csv'
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        extract_zip(download_url(url, ROOT), ROOT)
        movies_df = pd.read_csv(movies_path, index_col='movieId')
        movies_llm_df = pd.read_csv(movies_path, index_col='movieId')
        movies_llm_df['genres'] = movies_llm_df['genres'].apply(lambda genres: list(genres.split("|")))
        genres = movies_df['genres'].str.get_dummies('|')
        movie_feat = torch.from_numpy(genres.values).to(torch.float)
        ratings_df = pd.read_csv(ratings_path)
        return movies_df, movies_llm_df, movie_feat, ratings_df


    def __map_to_unique(self, movies_df: pd.DataFrame, movies_llm_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        '''Maps the IDs from the data set to a compact ID series.
        Then merge the data points so that <userID, movieID, title, genres] quadruples are created.
        '''
        # Create a mapping from unique user indices to range [0, num_user_nodes):
        self.unique_user_id = ratings_df['userId'].unique()
        self.unique_user_id = pd.DataFrame(data={
            'userId': self.unique_user_id,
            'mappedUserId': pd.RangeIndex(len(self.unique_user_id)),
        })
        # Create a mapping from unique movie indices to range [0, num_movie_nodes):
        self.unique_movie_id = pd.DataFrame(data={
            'movieId': movies_df.index,
            'mappedMovieId': pd.RangeIndex(len(movies_df)),
        })
        self.unique_llm_movie_id = pd.DataFrame(data={
            'mappedMovieId': pd.RangeIndex(len(movies_llm_df)),
            'title': movies_llm_df["title"],
            'genres': movies_llm_df["genres"]
        }).reset_index()

        # Perform merge to obtain the edges from users and movies:
        ratings_user_id = pd.merge(ratings_df['userId'], self.unique_user_id,
                                    left_on='userId', right_on='userId', how='left')
        ratings_user_id = torch.from_numpy(ratings_user_id['mappedUserId'].values)
        ratings_movie_id = pd.merge(ratings_df['movieId'], self.unique_movie_id,
                                    left_on='movieId', right_on='movieId', how='left')
        ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedMovieId'].values)

        # With this, we are ready to construct our `edge_index` in COO format
        # following PyG semantics:
        self.edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
        assert self.edge_index_user_to_movie.size() == (2, 100836)
        return ratings_df.merge(self.unique_user_id, on="userId").merge(self.unique_llm_movie_id, on="movieId")[["mappedUserId", "mappedMovieId", "title", "genres"]]


    def __generate_hetero_data(self, movies_df:pd.DataFrame, movie_feat:pd.DataFrame) -> HeteroData:
        '''(from Tutorial)
            With this, we are ready to initialize our `HeteroData` object and pass the necessary information to it.
            Note that we also pass in a `node_id` vector to each node type in order to reconstruct the original node indices from sampled subgraphs.
            We also take care of adding reverse edges to the `HeteroData` object.
            This allows our GNN model to use both directions of the edge for message passing:
        '''
        data = HeteroData()

        # Save node indices:
        data["user"].node_id = torch.arange(len(self.unique_user_id))
        data["movie"].node_id = torch.arange(len(movies_df))

        # Add the node features and edge indices:
        data["movie"].x = movie_feat
        data["user", "rates", "movie"].edge_index = self.edge_index_user_to_movie

        # We also need to make sure to add the reverse edges from movies to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        transform = T.ToUndirected()
        data = transform(data)

        assert data.node_types == ["user", "movie"]
        assert data.edge_types == [("user", "rates", "movie"),
                                ("movie", "rev_rates", "user")]
        assert data["user"].num_nodes == 610
        assert data["user"].num_features == 0
        assert data["movie"].num_nodes == 9742
        assert data["movie"].num_features == 20
        assert data["user", "rates", "movie"].num_edges == 100836
        assert data["movie", "rev_rates", "user"].num_edges == 100836
        return data

    def __split_data(self):
        '''(From Tutorial)
        ## Defining Edge-level Training Splits
        Since our data is now ready-to-be-used, we can split the ratings of users into training, validation, and test splits.
        This is needed in order to ensure that we leak no information about edges used during evaluation into the training phase.
        For this, we make use of the [`transforms.RandomLinkSplit`](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.RandomLinkSplit) transformation from PyG.
        This transforms randomly divides the edges in the `("user", "rates", "movie")` into training, validation and test edges.
        The `disjoint_train_ratio` parameter further separates edges in the training split into edges used for message passing (`edge_index`) and edges used for supervision (`edge_label_index`).
        Note that we also need to specify the reverse edge type `("movie", "rev_rates", "user")`.
        This allows the `RandomLinkSplit` transform to drop reverse edges accordingly to not leak any information into the training phase.
        '''
        # For this, we first split the set of edges into
        # training (80%), validation (10%), and testing edges (10%).
        # Across the training edges, we use 70% of edges for message passing,
        # and 30% of edges for supervision.
        # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
        # Negative edges during training will be generated on-the-fly, so we don't want to
        # add them to the graph right away.
        # Overall, we can leverage the `RandomLinkSplit()` transform for this from PyG:
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=("user", "rates", "movie"),
            rev_edge_types=("movie", "rev_rates", "user"),
            
        )
        gnn_train_data, gnn_val_data, gnn_test_data = transform(self.data)
        self.gnn_train_data: HeteroData = gnn_train_data
        self.gnn_val_data: HeteroData= gnn_val_data
        self.gnn_test_data: HeteroData = gnn_test_data
        torch.save(self.gnn_train_data, GNN_TRAIN_DATASET_PATH)
        torch.save(self.gnn_val_data, GNN_VAL_DATASET_PATH)
        torch.save(self.gnn_test_data, GNN_TEST_DATASET_PATH)
        torch.save(self.data, GNN_COMPLETE_DATASET_PATH)
    
    def __generate_llm_dataset(self) -> None:
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

    def __split_llm_dataset(self) -> None:
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

    def add_graph_embeddings(self, get_embedding_cb: Callable) -> None:
        '''
        This method computes the user node embedding and movie node embedding of all edges.
        '''
        def __add_graph_embeddings(row, get_embedding_cb: Callable) -> pd.Series:
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
    
    def __dataset_from_df(self, df: pd.DataFrame) -> DatasetDict:
        '''
        Generates the LLM datasets from the pandas dataframe.
        '''
        dataset_train = Dataset.from_pandas(df[df["split"] == "train"])
        dataset_val = Dataset.from_pandas(df[df["split"] == "val"])
        dataset_test = Dataset.from_pandas(df[df["split"] == "test"])
        return DatasetDict({
            "train": dataset_train,
            "val": dataset_val,
            "test": dataset_test,
        })


    def generate_prompt_embedding_dataset(self, tokenize_function:Callable = None, kge_dimension:int = 4, force_recompute: bool = False) -> DatasetDict:
        '''
        Generates the dataset for training the prompt model,
        by passing the tokenizer.tokenize function and
        the embedding dimension of the target prompt model.
        '''
        llm_prompt_dataset_path= LLM_PROMPT_DATASET_PATH.format(kge_dimension)
        if os.path.exists(llm_prompt_dataset_path) and not force_recompute:
            dataset = datasets.load_from_disk(llm_prompt_dataset_path)
        else:
            assert f"user_embedding_{kge_dimension}" in self.llm_df
            assert f"movie_embedding_{kge_dimension}" in self.llm_df
            assert self.llm_df[f"user_embedding_{kge_dimension}"].notna().all()
            assert self.llm_df[f"movie_embedding_{kge_dimension}"].notna().all()
            llm_df = self.llm_df.copy(deep = True)
            llm_df["labels"] = 1
            llm_df["prompt"] = self.llm_df.apply(lambda row: row_to_prompt_datapoint(row, kge_dimension), axis=1)
            dataset = self.__dataset_from_df(llm_df)
            if tokenize_function:
                dataset = dataset.map(tokenize_function, batched = True)
            dataset.save_to_disk(llm_prompt_dataset_path)
        return dataset
    
    def generate_vanilla_dataset(self, tokenize_function: Callable = None, force_recompute: bool = False) -> DatasetDict:
        '''
        Generates the dataset for training the vanilla model,
        by passing the tokenizer.tokenize function.
        '''
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
    
    def sample_prompt_datapoint(self, get_embedding_cb: Callable, split: str = "val", existing: bool = True, tokenize_function: Optional[Callable] = None, kgeg_dimension: int = 4)-> Dict[str, Union[str, int, torch.Tensor]]:
        '''
        Samples one datapoint of the prompt model dataset.
        Parameters
        __________
        get_embedding_cb:           Callable
                                    The callback function from the gnn_trainer that takes user and movie id and generates its embedding.
        split:                      str
                                    The split of the original dataset where to look for existing edges
                                    Default val
        existing:                   bool
                                    Flag if the sample edge is supposed to exist in the original dataset.
                                    Default True
        tokenize_function:          Optional[Callable]
                                    The callback function from the EncoderOnlyClassifier, that allows to encode the prompt to its input ids, if given
        kgeg_dimension:  int
                                    The kgeg_dimension of the gnn the sample is taking its embeddings from.
        Returns
        __________
        data_sample:                 Dict[str, Union[str, int, torch.Tensor]]
                                    A sample in the form {"prompt": <prompt>, "labels": <label>} if no tokenize_function was passed else
                                    {"input_ids": <input_ids>, "attention_mask": <attention_mask>, "labels": <label>}                         
        '''
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
                    user_embedding, movie_embedding = get_embedding_cb(dataset, user_id, movie_id, add_pca = True, split = split)
                    random_row["user_embedding"] = user_embedding
                    random_row["movie_embedding"] = movie_embedding
        prompt = row_to_prompt_datapoint(random_row, kgeg_dimension)
        labels = 1 if existing else 0
        result = {"prompt": prompt, "labels": labels}
        if tokenize_function:
            return tokenize_function(result, return_pt = True)
        else:
            return result
        
    def sample_vanilla_datapoint(self, split = "val", existing = True, tokenize_function = None):
        '''
        Samples one datapoint of the vanilla model dataset.
        Parameters
        __________
        split:                      str
                                    The split of the original dataset where to look for existing edges
                                    Default val
        existing:                   bool
                                    Flag if the sample edge is supposed to exist in the original dataset.
                                    Default True
        tokenize_function:          Optional[Callable]
                                    The callback function from the EncoderOnlyClassifier, that allows to encode the prompt to its input ids, if given
        Returns
        __________
        data_sample:                 Dict[str, Union[str, int, torch.Tensor]]
                                    A sample in the form {"prompt": <prompt>, "labels": <label>} if no tokenize_function was passed else
                                    {"input_ids": <input_ids>, "attention_mask": <attention_mask>, "labels": <label>}                         
        '''
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
        prompt = row_to_vanilla_datapoint(random_row)
        labels = 1 if existing else 0
        result = {"prompt": prompt, "labels": labels}
        if tokenize_function:
            return tokenize_function(result, return_pt = True)
        else:
            return result
        
    def replace_llm_df(self, df: pd.DataFrame):
        '''
        overrides the current llm dataset with the given dataset.'''
        self.llm_df = df
        self.llm_df.to_csv(LLM_DATASET_PATH, index = False)
        
            


