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
[2] describes that KGEs can be translated directly from their graph structure into an embedding or that a **graph neural network (GNN)** can be used for this purpose.
## Installation Process
Install most requirements using:
```pip install -r requirements.txt```
also install

```torch-sparse```
```torch-scatter```
These libraries have strict requirements tho and the correct versions had to be specified manually (in my case).
I refer you to the [tutorial](https://colab.research.google.com/drive/1xpzn1Nvai1ygd_P5Yambc_oe4VBPK_ZT?usp=sharing#scrollTo=rwgNwoa26Eja) by torch geometrics, which I used as a template. This tutorial is also a good starting point to understand Graph Representation Learning in general.

## Overview
A top down view on the overall process of Graph Representation Learning in LLMs as implemented in this project.
![Overview of the transformation- and trainingsprocesses.](/images/Overview.png?raw=true)
1.    The original (toy) dataset has user nodes and movie nodes, while the nodes are connected with edges from users to movies, if a user watched and rated the movie. In the first step we remove the rating and map new ids, so the graph is ready for link prediction.
2.    In the second step we train a GCNN on link prediction. The GCNN has to decide if any given edge exists in the original graph.
3.    With the original dataset and the dataset split we trained the GCNN on, we generate an NLP dataset, where the prompts for the LLM are the sum of the information we have about the user (only id) and the movie (title, genres and id). Again, the task of the LLM is to predict, if any given edge exists.
4.    Next with the trained GCNN, we produce node embeddings for every user and movie in their respective edge and split. For the user and movie embeddings for every split we fit a PCA and reduce the dimensionality to two. We end up with 8 fitted PCAs (2 for each split: train, val, test, rest)
5.    The reduced embeddings are now added to the prompt as a text, so the resulting dataset includes the graph embeddings.
6.    Two LLMs are now trained on predicting if a given edge exists, one on the dataset with embeddings and one without (vanilla).
## Playground
Follow the playground for experiments.

## Sources
[1] Knowledge Graph Embedding: A Survey from the Perspective of Representation Spaces
[2] Large Language Models on Graphs: A Comprehensive Survey
[3] Retrieval-augmented generation for knowledge-intensive nlp tasks
[4] Graph Neural Networks for Natural Language Processing: A Survey