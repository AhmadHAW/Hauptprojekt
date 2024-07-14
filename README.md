#  Visualize Knowledge Graph Embedding Influence on LLMs: A Toolsuit
by Ahmad Khalidi
As part of the Hauptprojekt in the HAW-Hamburg.

The Large Language Model (LLM) Transformer architecture has achieved great success in the Natural language processing field (NLP) in recent years. However, the immense success has also increased the demand on language models to refer to underlying facts in questions of knowledge and to reason in a comprehensible manner.
LLMs are often applied to purely natural language data, which do not include the structural information in which the texts may appear [2].
Structural information can be mapped as **Knowledge Graphs (KG)**, among other things.
KGs are multi-relational graphs that model knowledge from the real world in a structured way. Nodes represent entities such as names, products or locations and edges represent the relationships between these entities. In most cases, KGs are defined as a a list of triples: two entities and their relationship (i.e., ```<head entity, relation, tail entity>```)[1].
There are two major strategies to include knowledge of graphs into LLMs: the parametic (embedding) approach, also known as **Knowledge Graph Embedding (KGE)** and the non-parametic approach, also known as **Retrieval-Augmented Generation (RAG)**[3].

RAG allows LLMs to incorporate concrete knowledge from (un-)structured sources during sequence generation by using a pre-trained retriever to extract relevant passages from the facts and make them available to the LLM.
In the context of KG, this could mean for RAG to extract the nodes and edges relevant for the task and make them available to the LLM as additional information.

KGE, on the other hand, does not include any further textual information in the prediction, but refers exclusively to the specific graph topology of the entire graph and of nodes and edges in their (close) neighborhood.
![Possible Graph Structures [2]](/images/Graph-Structures.png?raw=true)
The illustration shows how strongly the structural differences between KG topologies can vary. With KGE, these complex high dimensional graph structures can be embedded on low-dimensional vectors and thus made available to the LLM for further downstream tasks [1][2][4].

## Research Question and Experiment
While RAG is a powerful tool for incorporating factual knowledge into LLMs, in this project we focus only on the influence of KGEs on LLMs, as their semantic influence on the inner workings of LLMs has neither been sufficiently studied nor intuitively understood.
With this knowledge, the following question arises from current research: *How does the **semantic processing of natural language in LLMs** change with the **inclusion of knowledge graph embeddings**, illustrated by a simple experiment?*

In a simple experiment, we want to determine how the inclusion of low dimensional graph embeddings affects the semantic *understanding* of the LLM. To do this, we compare LLMs that have been trained **with and without** the inclusion of KGEs. We want to determine what **degree of *attention*** the trained model gives to the KGEs. In addition, we want to determine how the **semantic properties of the LLM's output embeddings** change when the KGEs are included.
To simplify matters, we now call the LLM with the inclusion of KGE the "**Prompt-Model**" and the LLM without the inclusion of KGE the "**Vanilla-Model**".

As a task for the LLMs, we set the **link prediction**: A task in which the models should determine whether an edge exists between two given nodes.
The LLMs should be able to solve the binary classification task (*exists: True/False*) with the help of a textual description of both given nodes (Vanilla-Model) and in one case in addition with the help of the given KGE (Prompt-Model).

For the sake of simplicity, we reduce the KGE length using statistical methods until the performance of the prompt model significantly beats the performance of the Vanilla-Model without requiring too large a context length. We then compare and visualize the attention ratios between the natural language features and the KGEs using the evaluation dataset.

In the last step, we visualize the output embeddings of the LLMs with the help of statistical dimension reduction and semantically meaningful unions. During the experiments, further methods of unsupervised learning are applied to the output embeddings in order to understand the semantic influence of KGEs on LLMs. However, the exact methods are only superficially defined at this stage.

## Knowledge Graph Embedding Generation (Graph Representation Learning)
So far we have only talked about the use of KGEs in LLMs, but we have not clarified how they are produced from KGs.
![Role of LLM in Graph Representation Learning](/images/Roles.png?raw=true)
In this figure, [2] describes two essential decisions that need to be made when generating KGUs.
First, graph structures can be embedded directly into a low-dimensional vector space or, for example, a **Graph Neural Network (GNN)** can be trained to produce semantically rich embeddings.
Secondly, the roles of the LLM and GNN need to be clarified. Put simply, a GNN can be used to produce KGE for downstream tasks of the LLM (LLM as Predictor), GNN and LLN can learn the respective embeddings in an interplay for use in downstream tasks (LLS as Aligner) or the LLM produces semantically rich embeddings of the text for the GNN for use in downstream tasks (LLM as Encoder).
In this project, we use a Graph Convolutional Network (GCN) to generate the KGEs. GCNs are one of the best state-of-the-art architectures for generating semantically rich KGEs, which we expect to have a strong semantic influence on the behavior of the LLM.
In addition, we choose the role of the predictor for the LLM, since its behavior is in the foreground and we can thus retain more control over it during training.

In [5], a distinction is also made between three graph embedding techniques: node embeddings, edge embeddings and subgraph embeddings.
For **node embeddings**, a mapping function $f: v_i \rightarrow R^d$ is learned so that for a graph $G = (V,E)$, where $V$ is the set of all nodes and $E$ is the set of all edges of the graph, a low-dimensional vector of dimension $d$ is created for each node $v_i$ such that $d << |V|$ and the **similarity of nodes in the graph is preserved in embedding space**.
![Example Node Embedding](/images/Node%20Embedding.PNG)
In this example, all nodes from the given graph are mapped to vectors of dimension $d=4$, with the example values for node $a$.

**Graph edge embeddings** define a mapping function $f: e_i->R^d$ for a given graph $G=(V,E)$, so that for vector dimension $d << |E|$ applies and the **similarity between edges of the graph in the embedding space is preserved**.

With **subgraph embeddings**, a mapping function is learned for a given graph $G=(V,E)$ so that $d<<|G_{sub}|$ applies to the low-dimensional embedding and the **similarity of the subgraphs is preserved**.

Edge and subgraph embeddings are often based on node embeddings [5], which is why we only use the node embedding method for simplification in this project.

I refer to the [tutorial script](https://colab.research.google.com/drive/1xpzn1Nvai1ygd_P5Yambc_oe4VBPK_ZT?usp=sharing) of [6] for a detailed implementation and description of how the applied Graph Convolutional Network generates the node embeddings from given nodes and their neighborhoods. We have only made one change to this implementation by allowing you to specify the output dimension of the embedding vector.

## Training and Inference Framework for Language Models on Graphs.
Now that we understand how KGEs are created and we defined the role of LLM, let's take a closer look at exactly how KGEs can be incorporated into the LLM training and inference process.
There is ***Graph as Sequence***, ***Graph-Empowered LLM*** and ***Graph-Aware LLM Finetuning*** for LLMs in the role of predictors. With the graph as sequence method, the KGE is embedded in the prompt. This method does not require any changes to the LLM architecture. In the Graph Empowered LLM method, the architecture of the LLM is modified so that text and KGE can be processed together. In Graph Aware LLM Finetuning, neither the input prompt nor the model architecture are changed, but the training process is adapted with the involvement of the KGEs.[2]
As before, we opt for the supposedly simplest implementation to observe the semantic influence of KGEs on LLMs using the ***graph as sequence method***.

## Dataset MovieLens
MovieLens is a non-commercial movie-recommendation dataset. Users can rate films via an [Internet service](https://movielens.org/). In its full form, the benchmark dataset consists of 25 million film ratings, 1 million tags applied to 62,000 films and 16,000 users. A smaller version consists of 100000 ratings, 3600 tags, applied to 9000 movies and 600 users.
A data point has the form ```<user, movie, rating, timestamp>```. Users are only mapped as IDs. Movies have the content <title, ID, genres>. For our research question, we are not looking at the tags for now.
As already mentioned, we have strictly adhered to the Torch Geometric Tutorial when reshaping the data set. As a result, we get the following data points for the graph model: ```<user, rates, movie>```, where user and movie only consist of the IDs.
For the LLM we get the following data points: ```<prompt, label>```, where the prompt is a natural language text consisting of *user ID, movie title, movie genres and movie ID*. The label is ```1``` (for user has rated the movie) and ```0``` (for user has not rated the movie).
The original dataset only contains existing edges. That means we have to generate new (non existing) edges during training.
## Visualizing the Semantic Embedding Space
 

## Sources
[1] Knowledge Graph Embedding: A Survey from the Perspective of Representation Spaces
[2] Large Language Models on Graphs: A Comprehensive Survey
[3] Retrieval-augmented generation for knowledge-intensive nlp tasks
[4] Graph Neural Networks for Natural Language Processing: A Survey
[5] A Survey on Graph Representation Learning Methods
[6] Fast Graph Representation Learning with {PyTorch Geometric}
[7] The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems