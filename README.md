# Visualize Knowledge Graph Embedding Influence on LLMs: A Toolsuit

by Ahmad Khalidi
As part of the Hauptprojekt in the HAW-Hamburg.

## Toolsuite Applications

The Toolsuite is designed to give the user insights of how Knowledge Graph Embeddings change the semantic behaviour of LLMs (NLP-Transformer) and provides a pipeline with which a user can:

**1.**  pre-process a suitable graph dataset and convert it into a standardized format,
**2.**  produce low-dimensional graph embeddings (KGE) of nodes,
**3.**  train multiple LLMs on the downstream task, with and without including KGEs and
**4.**  create plots over semantically meaningful related areas of attentions and hidden states during inference.

As a result of the toolsuite pipeline, the user is confronted with a the internal structures of multiple LLMs. Those LLMs were trained in different ways to include KGEs in or even omit KGEs completely and rely only on the natural language part of the dataset. This allows the user to get an understanding of how introducing KGEs to LLMs changes their semantic behaviour.

The current version only supports the downstream task: **Link Prediction**, a task in which the model must decide for a given pair of nodes whether an edge exists between them.
In addition, only the **Bert architecture** [8]  is currently supported. However, the corresponding interfaces for integrating other architectures are available.
The tool suite was designed so that **standardized artifacts** are exchanged between the steps of the pipeline, thus ensuring a loose coupling of the components.
The tool suite is demonstrated using the MovieLens dataset [7]. However, if the required preconditions are met, any other graph datasets can also be used for processing.

## Quickstart

This toolsuites end products are plots over semantically relevant areas of LLMs with and without the inclusion knowledge graph embeddings. Since producing all the neccesary steps can be time consuming, we start from the end of the pipeline and offer the preprocessed artifacts along the way. This way the user is able to get right into the action without having to train entire models and produce embeddings, attentions and hidden states. By moving backwards we offer the a portion of the tutorial to teach how to customize the current steps, so that in the end the user is able to customize the entire pipeline.

### Model Names

For convenience we named the models according to their strategy to include KGEs. The **Vanilla Model** does not include KGEs. The **Prompt Model** does include KGEs in the prompt. The **Attention Model** does include KGEs in the embedding space and reached the highest perfomance in our experiements.

### Requirements Installation

If Cuda is available run ```pip install -r requirements_cuda.txt``` else ```pip install -r requirements_cpu.txt```. If the torch or cuda version in the requirements.txt do not match, update them.

### Huggingface Login

Until we have solved the licensing issues of this works added data on the original MovieLens Dataset, the access to the artifacts is sitting behind an access wall. Contact <Ahmad.Khalidi@Haw-Hamburg.de> if you want access and state your business.

If you have already granted access to the dataset, run:
```huggingface-cli login``` in a terminal and pass your access token to your account.

### Attentions Plot

Attentions of the LLM offer a great way to understand where the information for the decision process actually came from.
First we import the MovieLens dataset manager (dm) and explainability module (exp):

```python
from dataset_manager import MovieLensManager
from explainability_module import MovieLensExplainabilityModule
```

Then we load the dataset from huggingface hub and init the explainability module with it:

```python
df = MovieLensManager.load_dataset_from_hub("AhmadPython/MovieLens_KGE")
exp = MovieLensExplainabilityModule(df)
```

And print its content in a way, that we display all columns with numpy content as their dtype and shape with

```python
print(exp.get_verbose_df(n = 5))
```

|   | source_id | target_id | id_x | id_y | prompt_feature_title        | prompt_feature_genres                                       | labels | split | prompt                                                                                       | prompt_source_embedding | prompt_target_embedding | input_embeds_replace_source_embedding | input_embeds_replace_target_embedding | vanilla_attentions | vanilla_hidden_states | vanilla_attentions_original_shape | vanilla_hidden_states_original_shape | prompt_attentions    | prompt_hidden_states  | prompt_attentions_original_shape | prompt_hidden_states_original_shape | input_embeds_replace_attentions | input_embeds_replace_hidden_states | input_embeds_replace_attentions_original_shape | input_embeds_replace_hidden_states_original_shape |
|---|-----------|-----------|------|------|-----------------------------|-------------------------------------------------------------|--------|-------|----------------------------------------------------------------------------------------------|-------------------------|-------------------------|----------------------------|----------------------------|--------------------|-----------------------|-----------------------------------|--------------------------------------|----------------------|-----------------------|----------------------------------|-------------------------------------|----------------------|-------------------------|-------------------------------------|----------------------------------------|
| 0 | 0         | 0         | 0    | 0    | Toy Story (1995)            | ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy'] | 1      | train | 0[SEP]0[SEP]Toy Story [1995][SEP]('Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy') | float64: (4,)           | float64: (4,)           | float64: (128,)            | float64: (128,)            | float32: (9, 9, 2) | float32: (3, 9, 128)  | int64: (3,)                       | int64: (3,)                          | float32: (13, 13, 2) | float32: (3, 13, 128) | int64: (3,)                      | int64: (3,)                         | float32: (13, 13, 2) | float32: (3, 13, 128)   | int64: (3,)                         | int64: (3,)                            |
| 1 | 0         | 2         | 0    | 2    | Grumpier Old Men (1995)     | ['Comedy', 'Romance']                                       | 1      | train | 0[SEP]2[SEP]Grumpier Old Men [1995][SEP]('Comedy', 'Romance')                                | float64: (4,)           | float64: (4,)           | float64: (128,)            | float64: (128,)            | float32: (9, 9, 2) | float32: (3, 9, 128)  | int64: (3,)                       | int64: (3,)                          | float32: (13, 13, 2) | float32: (3, 13, 128) | int64: (3,)                      | int64: (3,)                         | float32: (13, 13, 2) | float32: (3, 13, 128)   | int64: (3,)                         | int64: (3,)                            |
| 2 | 0         | 5         | 0    | 5    | Heat (1995)                 | ['Action', 'Crime', 'Thriller']                             | 1      | train | 0[SEP]5[SEP]Heat [1995][SEP]('Action', 'Crime', 'Thriller')                                  | float64: (4,)           | float64: (4,)           | float64: (128,)            | float64: (128,)            | float32: (9, 9, 2) | float32: (3, 9, 128)  | int64: (3,)                       | int64: (3,)                          | float32: (13, 13, 2) | float32: (3, 13, 128) | int64: (3,)                      | int64: (3,)                         | float32: (13, 13, 2) | float32: (3, 13, 128)   | int64: (3,)                         | int64: (3,)                            |
| 3 | 0         | 43        | 0    | 43   | Seven (a.k.a. Se7en) (1995) | ['Mystery', 'Thriller']                                     | 1      | train | 0[SEP]43[SEP]Seven (a.k.a. Se7en) [1995][SEP]('Mystery', 'Thriller')                         | float64: (4,)           | float64: (4,)           | float64: (128,)            | float64: (128,)            | float32: (9, 9, 2) | float32: (3, 9, 128)  | int64: (3,)                       | int64: (3,)                          | float32: (13, 13, 2) | float32: (3, 13, 128) | int64: (3,)                      | int64: (3,)                         | float32: (13, 13, 2) | float32: (3, 13, 128)   | int64: (3,)                         | int64: (3,)                            |
| 4 | 0         | 130       | 0    | 130  | Canadian Bacon (1995)       | ['Comedy', 'War']                                           | 1      | train | 0[SEP]130[SEP]Canadian Bacon [1995][SEP]('Comedy', 'War')                                    | float64: (4,)           | float64: (4,)           | float64: (128,)            | float64: (128,)            | float32: (9, 9, 2) | float32: (3, 9, 128)  | int64: (3,)                       | int64: (3,)                          | float32: (13, 13, 2) | float32: (3, 13, 128) | int64: (3,)                      | int64: (3,)                         | float32: (13, 13, 2) | float32: (3, 13, 128)   | int64: (3,)                         | int64: (3,)                            |

As we can see this dataset contains source and target ids of nodes, in context of the MovieLens dataset **user id** and **movie id**. Then there are the natural language features, like prompt_feature_**title** and prompt_feature_**genres**. Labels *1* are existing edges, naming the user has in fact rated the movie. **Prompt** stands for the *Vanilla Prompt* The kind of prompt that does not include KGEs. Every prompt is build the same way: ```source_id[SEP]movie_id[SEP]prompt_feature_1[SEP]prompt_feature_2[SEP]...```
While the prompt for the model that includes KGEs in the prompt adds
```prompt_source_embedding[SEP]prompt_target_embedding[SEP]``` with both embeddings being of dtype float 16 with length 4.
The *input_embeds_replace_source_embedding* and *input_embeds_replace_target_embedding* are not passed in the prompt but in the input embeddings of the LLM, replacing the placeholder *[PAD]* tokens. Both are of dtype float64 and 128 length, because the Bert model this was produced with has a hidden state size of 128.

We are more interested in the *vanilla_attentions*, *prompt_attentions* and *input_embeds_replace_attentions*. These arrays give us for the respective strategy the averaged attentions over the prompt positions. So ones we plot these attentions, we will see the attentions the model puts on each individual feature in total. Other then the special tokens *[SEP]* we will also the the beginning token *[CLS]* which is added by the Bert classifier. This token is passed to the classifier header and summarizes the entire models extracted features.

We now plot the attentions over all models with:

```python
exp.plot_all_attention_graphs()
```

Vanilla Model             |  Prompt Model             |  Input Embeds Replace Model
:-------------------------:|:-------------------------:|:-------------------------:
![Vanilla Attentions](/images/Vanilla_Model_Attentions.png)  |  ![Prompt Attentions](/images/Prompt_Model_Attentions.png)|![Embedding Attentions](/images/Input_Embeds_Replace_Model_Attentions.png)

By not going to deep into the interpretation of the plots, we can see the common behaviour, where in the last layer, all features influense the *[CLS]* token the most. For more insights and interpretation feel free to check my master thesis [...] which will be released soon.

Now that we have plotted the attentions of the models, we get an idea where the information come from, that was used to predict if a user rated a movie.

The plots above were created using the [attentions notebook](/attentions.ipynb). If you want to know how to customize these plots, check out [attention customization](/attention_customization_tutorial.ipynb).

### Hidden State Plots

When it comes to generating hidden state plots, we first have to reduce the hidden states to 2 dimensions using principal component analysis.

```python
from dataset_manager import MovieLensManager
from explainability_module import MovieLensExplainabilityModule
df = MovieLensManager.load_dataset_from_hub("AhmadPython/MovieLens_KGE")
exp = MovieLensExplainabilityModule(df, init_pcas=True)
```

When initializing, we fit PCA on every feature token seperatly over the train dataset split. Then for plots we transform the filtered samples into their lower dimensions and plot them in a way, that the specific behaviour we are seeking to understand is underlined.

For example with

```python
exp.plot_cls_embeddings(samples=1000)
```

We plot the 2 dimensional embeddings of cls tokens, seperated by colour for each model.
![CLS Token Hidden States](/images/cls_token_embeddings.png)
Or we can plot all cls tokens seperating the model by shape and the label *edge (does not) exist* by color.

## Toolsuite Components

The toolsuite is divided into four main components: **Dataset Manager**, **Graph Representation Generator**, **LLM Manager** and **Explainability Module**.
The Dataset Manager preprocesses given dataset and manages its distribution and persistence.
The Graph Representation Generator manages the Graph Models to produe KGEs.
The LLM Manager enables the LLMs to process KGE empowered datasets.
Explainability Module produces manageable plots of the inner workings of the LLMs. The components are defined as following.

### Dataset Manager

#### Initialization

The Knowledge Graph Manager is initialized with a standartized form of the Knowledge Graph. First the manager genereates a dir structure for all upcomming files to be saved. The dataset is then transformed into a Torch Geometric [6] Hetero data object ("A data object describing a heterogeneous graph, holding multiple node and/or edge types in disjunct storage objects."). This data object is then split into *train*, *test* and *validation* data objects. The same split is then applied on the original standartized form of the dataset. Last, prompts are generated, that can be used by LLMs for the downstream *link prediction* task **without** including *KGEs*.

```python
def __init__(
    self,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    force_recompute: bool = False,
)
```

[link to code](dataset_manager.py#L35-L41)

Every dataset manager that inherets form the KGManger to pass the source, target and edge dataframes in the initialization process. The MovieLensManager who inherets from the KGManager generates the dataframes from the automaticaly downloaded MovieLens dataset.

#### Dataset Distributions

Every other component receives their datas from this manager but also provides new data to this manager, to be included in a standartized form. Once the KGEs are produced, they can be appended to the original standartized dataset. The KGManager manages the following 4 datasets

```python
kg_manager.data, # the fill Graph Hetero Data object 
kg_manager.gnn_train_data, # the train split of the Hetero Data object
kg_manager.gnn_val_data,
kg_manager.gnn_test_data,
kg_manager.llm_df # The LLM dataframe with labels, prompts and embeddings
kg_manager.source_df # The LLM dataframe with source_ids and their features
kg_manager.target_df # The LLM dataframe with target_ids and their features
```

[link to llm dataframes](/dataset_manager.py#L97-L104), [link to graph datasets](/dataset_manager.py#L220-L222)

#### Dataset Persistence

The standartized original form of the dataset are tabular data and saved as *csv*. Every additional Embeddings, while persisting are first transformed into torch-Tensors and saved seperatly. The saved csv do not contain any embeddings on the disk, but only in memory.

The actual code for saving the llm_df datafrane is checking first if there are torch tensors or numpy array to be expected.

```python
columns_without_embeddings = list(
    filter(lambda column: "embedding" not in column, self.llm_df.columns)
)
columns_with_embeddings = list(
    filter(lambda column: "embedding" in column, self.llm_df.columns)
)
self.llm_df[columns_without_embeddings].to_csv(LLM_DATASET_PATH, index=False)
for column in columns_with_embeddings:
    torch.save(
        torch.tensor(self.llm_df[column].tolist()), f"{GNN_PATH}/{column}.pt"
    )
```

[save_llm_df](/dataset_manager.py#L312)

While in the huggingface dataset format, nested numpy arrays are first flattened and then saved, becaue huggingface does not allow nested numpy arrays. For restoration purpuses later, we also save the original shapes of the numpy arrays.

```python
df["attentions_original_shape"] = df["attentions"].apply(
    lambda attention: attention.shape
)
df["attentions"] = df["attentions"].apply(
    lambda attention: attention.flatten()
)
df["hidden_states_original_shape"] = df["hidden_states"].apply(
    lambda hidden_states: hidden_states.shape
)
df["hidden_states"] = df["hidden_states"].apply(
    lambda attention: attention.flatten()
)
```

[from generate_huggingface_dataset](/dataset_manager.py#L550)

### Graph Representation Generator

The graph representation generators are managing the underlying graph convolutional networks. They are also providing a callback function to other components which allows them to produce KGEs for any given datapoint. There is one graph representation generators for every method we have to include KGEs into the LLM decision process (currently 2). There are multiple generators in this toolsuite, because depending on the way the KGE is included in the LLM, different embedding sizes are viable and thus different graph convolutional model output sizes.

#### Graph Convolutional Network Training

Ones the Knowledge Graph Manager is initialized the Graph Representation Generator picks up the Hetero data object and trains the graph convolutional networks on the **link prediction** task. While training, **non-existing edges are produced on the fly** with a factor of 1:1, so for every existing edge, one non-existing edge is trained on.

```python
'''
Init and training the graph representation learner
for the LLm where embeddings are passed in the prompt
'''
graph_representation_generator_prompt = GraphRepresentationGenerator(
    kg_manager.data,
    kg_manager.gnn_train_data,
    kg_manager.gnn_val_data,
    kg_manager.gnn_test_data,
    kge_dimension=PROMPT_KGE_DIMENSION,
)
graph_representation_generator_prompt.train_model(
    kg_manager.gnn_train_data, EPOCHS, BATCH_SIZE
)
```

#### Generate Embeddings

Once the datasets are ready, and the KGManager are trained, we produce the embeddings for every datapoint in the LLM dataset and assign them.

```python
prompt_embeddings = graph_representation_generator_prompt.generate_embeddings(
    kg_manager.llm_df
)
kg_manager.append_prompt_graph_embeddings(prompt_embeddings)
```

#### KGE Callback Function

This callback function provides other components to generate the KGEs for any given datapoint (existing or non-existing). The callback function takes care, that when producing the KGE, only nodes of the same split are included.

### LLM Manager

The llm manager provide the nessecary tools to include KGEs into LLMs (currently Bert only). Every Bert Classifier in this component manages its own DataLoader and tokenize procedure, depending on how KGEs are included in the *link-prediction* task. After the training, the manager produces a fixed validation dataset that is then passed to the Explainability Module.

#### Bert Classifier

Every Bert Classifier manages the way, KGEs are included into the prediction process. For that the input parameters are expanded by a semantic positional encoding and in one case by the KGEs. To expand the input parameters, all classifier hold customized DataLoader and tokenize procedure.

```python
if inputs_embeds is None:
    inputs_embeds = self.bert.embeddings(
        input_ids
    )  # we take the input embeds and later replace the positions where the padding tokens where placed as placeholder with the graph embeddings.
    assert isinstance(inputs_embeds, torch.Tensor)
if graph_embeddings is not None and len(graph_embeddings) > 0:
    if attention_mask is not None:
        mask = (
            (
                (attention_mask.to(self.device).sum(dim=1) - 1)
                .unsqueeze(1)
                .repeat((1, 2))
                - torch.tensor([3, 1], device=self.device)
            )
            .unsqueeze(2)
            .repeat((1, 1, self.config.hidden_size))
        )  # basically a mask finding the last positions between the sep tokens (reshaped so they can be used in scatter)
        inputs_embeds = inputs_embeds.to(
            self.device
        ).scatter(
            1, mask.to(self.device), graph_embeddings.to(self.device)
        )  # replace the input embeds at the place holder positions with the KGEs.
outputs = self.bert(
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    position_ids=position_ids,
    head_mask=head_mask,
    inputs_embeds=inputs_embeds,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)  # feed forward the input embeds to the attention model
```

[from](/llm_manager.py#L607-638)
[tutorial change feed forward](/customize_input_embeds_replace_model.ipynb)

#### Data Loader

The data loaders are based on the Graph Representation Learner and are able to produce non-existing edges on the fly. They also manage the KGEs and values to be passed in the correct format. A data loader can be initialized in a way, that no non-existing edges are produced on the fly.

```python
new_features = []
for feature in features:
    # Every datapoint has a chance to be replaced by a negative datapoint, based on the false_ratio.
    # The _transform_to_false_exmample methods have to be implemented by the inheriting class.
    # For the prompt classifier, every new datapoint also contains embeddings of the nodes.
    # If the false ratio is -1 this step will be skipped (for validation)
    if self.false_ratio != -1 and rd.uniform(0, 1) >= (
        1 / (self.false_ratio + 1)
    ):
        new_feature = self._transform_to_false_example()
        new_features.append(new_feature)
    else:
        new_features.append(feature)
```

[from](/llm_manager.py#L144-156)

#### Tokenize Procedure

The tokenize procedure is not only defining the mapping of prompts to input ids and attention masks, but also the generation of the semantic positional encodings. These encodings are used to summarize hidden states and attentions over the entire input positions.

```python
'''
Every tokenize function receives the example to tokenize. The result are
not only the ids, attention mask and labels, but also the
semantic positional encoding, which are basically positions of all 
features over the datapoint.
'''
def tokenize_function(self, example, return_pt=False):
    tokenized = self.tokenizer(
        example["prompt"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ...
    result = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": example["labels"],
        "semantic_positional_encoding": semantic_positional_encoding,
    }
```

[example1](/llm_manager.py#1140-1156), [example2](/llm_manager.py#L1233-1249)

#### Producing the fixed Validation dataset

The LLM Manager produces a fixed dataset with existing and non-existing edges. For every sample forwarded, the hidden states and attentions are summarized over the semantic positional encodings and then saved for later use. This results in a validation dataset that can be examined in the Explainability Module.

```python
'''
This method produces non-existing edges for the splits "val" and "test"
so that they do not have to generate non-existing edges on the fly and
make the validation process more repeadable.
'''
df = kg_manager.add_false_edges(
    1.0,
    graph_representation_generator_prompt.get_embedding, #passing the callback function for producing the embedding for any given datapoint.
    graph_representation_generator_attention.get_embedding,
    splits=["val", "test"],
)
```

### Explainability Module

The Explainability Module is used to examine the semantic behaviour of the LLM and their usage of KGEs in downstream tasks. With the previous summarization process, the attentions can be used to explain what features and KGEs the LLM takes into consideration. The hidden states give indications of how much information is stored in the internal representation of the LLMs.

#### Attentions View

This module allows to plot the attentions over all input features over all attention layers of the LLM. This allows us to see which features and KGEs have the highest influence on the downstream task. We can also summerize the attentions even more to directly compare the influence of input features and KGEs.

#### Hidden States View

This module allows the user to display the internal LLM embeddings on a 2d view, using principal component analysis. For every feature and KGE, the summarized hidden states are transformed to two dimensions. The user can examine the differences of the models with corresponding filtering, coloring and shaping of the scatter plot.

## Related Work

The Large Language Model (LLM) Transformer architecture has achieved great success in the Natural language processing field (NLP) in recent years. However, the immense success has also increased the demand on language models to refer to underlying facts in questions of knowledge and to reason in a comprehensible manner.
LLMs are often applied to purely natural language data, which do not include the structural information in which the texts may appear [1].
Structural information can be mapped as **Knowledge Graphs (KG)**, among other things.
KGs are multi-relational graphs that model knowledge from the real world in a structured way. Nodes represent entities such as names, products or locations and edges represent the relationships between these entities. In most cases, KGs are defined as a a list of triples: two entities and their relationship [i.e., ```<head entity, relation, tail entity>```](2).
There are two major strategies to include knowledge of graphs into LLMs: the parametic (embedding) approach, also known as **Knowledge Graph Embedding (KGE)** and the non-parametic approach, also known as **Retrieval-Augmented Generation (RAG)**[3].

KGE refers exclusively to the specific graph topologies of (sub-) graphs, nodes and edges in their (close) neighborhood.
![Possible Graph Structures [1]](/images/Graph-Structures.png?raw=true)

The illustration shows how strongly the structural differences between KG topologies can vary. With KGE, these complex high dimensional graph structures can be embedded on low-dimensional vectors and thus made available to the LLM for further downstream tasks [1][2][4].
We want to provide a tool to understand *the **semantic processing of natural language in LLMs** when **knowledge graph embeddings** are introduced.*

## Sources

[1] Jin, Bowen et al. “Large Language Models on Graphs: A Comprehensive Survey.” ArXiv abs/2312.02783 (2023): n. pag.

[2] Cao, Jiahang et al. “Knowledge Graph Embedding: A Survey from the Perspective of Representation Spaces.” ACM Computing Surveys 56 (2022): 1 - 42.

[3] Lewis, Patrick et al. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” ArXiv abs/2005.11401 (2020): n. pag.

[4] Wu, Lingfei et al. “Graph Neural Networks for Natural Language Processing: A Survey.” ArXiv abs/2106.06090 (2021): n. pag.

[5] Khoshraftar, Shima and Aijun An. “A Survey on Graph Representation Learning Methods.” ACM Transactions on Intelligent Systems and Technology 15 (2022): 1 - 55.

[6] Fey, Matthias and Jan Eric Lenssen. “Fast Graph Representation Learning with PyTorch Geometric.” ArXiv abs/1903.02428 (2019): n. pag.

[7] Harper, F. Maxwell et al. “The MovieLens Datasets: History and Context.” ACM Trans. Interact. Intell. Syst. 5 (2016): 19:1-19:19.

[8] Devlin, Jacob, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” North American Chapter of the Association for Computational Linguistics (2019).
