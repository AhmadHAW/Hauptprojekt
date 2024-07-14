#  Visualize Knowledge Graph Embedding Influence on LLMs: A Toolsuit
by Ahmad Khalidi
As part of the Hauptprojekt in the HAW-Hamburg.

The Large Language Model (LLM) Transformer architecture has achieved great success in the Natural language processing field (NLP) in recent years. However, the immense success has also increased the demand on language models to refer to underlying facts in questions of knowledge and to reason in a comprehensible manner.
LLMs are often applied to purely natural language data, which do not include the structural information in which the texts may appear [1].
Structural information can be mapped as **Knowledge Graphs (KG)**, among other things.
KGs are multi-relational graphs that model knowledge from the real world in a structured way. Nodes represent entities such as names, products or locations and edges represent the relationships between these entities. In most cases, KGs are defined as a a list of triples: two entities and their relationship (i.e., ```<head entity, relation, tail entity>```)[2].
There are two major strategies to include knowledge of graphs into LLMs: the parametic (embedding) approach, also known as **Knowledge Graph Embedding (KGE)** and the non-parametic approach, also known as **Retrieval-Augmented Generation (RAG)**[3].

RAG allows LLMs to incorporate concrete knowledge from (un-)structured sources during sequence generation by using a pre-trained retriever to extract relevant passages from the facts and make them available to the LLM.
In the context of KG, this could mean for RAG to extract the nodes and edges relevant for the task and make them available to the LLM as additional information.

KGE, on the other hand, does not include any further textual information in the prediction, but refers exclusively to the specific graph topology of the entire graph and of nodes and edges in their (close) neighborhood.
![Possible Graph Structures [1]](/images/Graph-Structures.png?raw=true)
The illustration shows how strongly the structural differences between KG topologies can vary. With KGE, these complex high dimensional graph structures can be embedded on low-dimensional vectors and thus made available to the LLM for further downstream tasks [1][2][4].

## Research Question and Experiment
While RAG is a powerful tool for incorporating factual knowledge into LLMs, in this project we focus only on the influence of KGEs on LLMs, as their semantic influence on the inner workings of LLMs has neither been sufficiently studied nor intuitively understood.
With this knowledge, the following question arises from current research: *How does the **semantic processing of natural language in LLMs** change with the **inclusion of knowledge graph embeddings**, illustrated by a simple experiment?*

In a simple experiment, we want to determine how the inclusion of low dimensional graph embeddings affects the semantic *understanding* of the LLM. To do this, we compare LLMs that have been trained **with and without** the inclusion of KGEs. We want to determine what **degree of *attention*** the trained model gives to the KGEs. In addition, we want to determine how the **semantic properties of the LLM's output embeddings** change when the KGEs are included.
To simplify matters, we now call the LLM with the inclusion of KGE the "**Prompt-Model**" and the LLM without the inclusion of KGE the "**Vanilla-Model**".

The task for the LLMs is **link prediction**: A task in which the models should determine whether an edge exists between two given nodes.
The LLMs should be able to solve the binary classification task (*exists: True/False*) with the help of a textual description of both given nodes (Vanilla-Model) and in one case in addition with the help of the given KGE (Prompt-Model).

## Knowledge Graph Embedding Generation (Graph Representation Learning)
So far we have only talked about the use of KGEs in LLMs, but we have not clarified how they are produced from KGs.
![Role of LLM in Graph Representation Learning](images/Roles.PNG?raw=true)
In this figure, [1] describes two essential decisions that need to be made when generating KGUs.
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
There is ***Graph as Sequence***, ***Graph-Empowered LLM*** and ***Graph-Aware LLM Finetuning*** for LLMs in the role of predictors. With the graph as sequence method, the KGE is embedded in the prompt. This method does not require any changes to the LLM architecture. In the Graph Empowered LLM method, the architecture of the LLM is modified so that text and KGE can be processed together. In Graph Aware LLM Finetuning, neither the input prompt nor the model architecture are changed, but the training process is adapted with the involvement of the KGEs.[1]
As before, we opt for the supposedly simplest implementation to observe the semantic influence of KGEs on LLMs using the ***graph as sequence method***.

## Dataset MovieLens
MovieLens is a non-commercial movie-recommendation dataset. Users can rate films via an [Internet service](https://movielens.org/). In its full form, the benchmark dataset consists of 25 million film ratings, 1 million tags applied to 62,000 films and 16,000 users. A smaller version consists of 100000 ratings, 3600 tags, applied to 9000 movies and 600 users.
A data point has the form ```<user, movie, rating, timestamp>```. Users are only mapped as IDs. Movies have the content <title, ID, genres>. For our research question, we are not looking at the tags for now.
As already mentioned, we have strictly adhered to the Torch Geometric Tutorial when reshaping the data set. As a result, we get the following data points for the graph model: ```<user, rates, movie>```, where user and movie only consist of the IDs.
For the LLM we get the following data points: ```<prompt, label>```, where the prompt is a natural language text consisting of *user ID, movie title, movie genres and movie ID*. The label is ```1``` (for user has rated the movie) and ```0``` (for user has not rated the movie).
The original dataset only contains existing edges. That means we have to generate new (non existing) edges during training and inference.
## Visualizing the Semantic Embedding Space
We have described how KGEs are generated and how we transfer them to the LLM in the training and inference process. Now let's take a look at how we want to measure and visualize the changes in the semantic understanding of the LLM.
First, let's look at the general performance of the vanilla LLM and the prompt LLMs. We check whether adding the KGEs leads to an improvement in accuracy. To do this, we look at the training process and then use the evaluation dataset to produce the cofusion matrices of the models.

Then we calculate the positional encodings of all semantically meaningful passages in the prompt, such as user ID, movie ID, movie title, genres, user embedding and movie embedding.
With the help of these positional encodings, we produce two views, one view of the attentions in the output layer and one view of the embeddings in the output layer.

We expect to observe a shift from the natural language content to the KGEs in the view on the attentions. We may also see a difference in attention for user groups that are strongly or weakly connected (have rated many/few movies).

In the view of the embeddings in the output layer, we can reduce the dimensions with Principal Component Analysis and thus produce a two-dimensional vector space. Here we try to compare the distance and proximity of semantically interesting subgroups more precisely.


## Sources
[1] Jin, Bowen et al. “Large Language Models on Graphs: A Comprehensive Survey.” ArXiv abs/2312.02783 (2023): n. pag.
[2] Cao, Jiahang et al. “Knowledge Graph Embedding: A Survey from the Perspective of Representation Spaces.” ACM Computing Surveys 56 (2022): 1 - 42.
[3] Lewis, Patrick et al. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” ArXiv abs/2005.11401 (2020): n. pag.
[4] Wu, Lingfei et al. “Graph Neural Networks for Natural Language Processing: A Survey.” ArXiv abs/2106.06090 (2021): n. pag.
[5] Khoshraftar, Shima and Aijun An. “A Survey on Graph Representation Learning Methods.” ACM Transactions on Intelligent Systems and Technology 15 (2022): 1 - 55.
[6] Fey, Matthias and Jan Eric Lenssen. “Fast Graph Representation Learning with PyTorch Geometric.” ArXiv abs/1903.02428 (2019): n. pag.
[7] Harper, F. Maxwell et al. “The MovieLens Datasets: History and Context.” ACM Trans. Interact. Intell. Syst. 5 (2016): 19:1-19:19.