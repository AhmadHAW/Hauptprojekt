#  Visualize Graph Representation Learning Influence on LLMs: A Toolsuit
by Ahmad Khalidi
As part of the Hauptprojekt in the HAW-Hamburg.

In this project, we look at the behavior of a Bert model when we pass the embeddings of Graph Convolutional Networks (GCNN) in the prompt.

The process is as follows: On a sample data set we train a GCNN on the task of link prediction. The structure of the GCNN allows the embeddings of individual nodes to be extracted during inference. We then reduce these embeddings with Principal Component Analysis (PCA) to obtain a low-dimensional and compact form of the embeddings. Two Bert models are then trained on the link prediction task (with and without including the embeddings). We then analyze the performance of the two models in comparison. We also visualize the attention header of the models. We try to confirm the direct influence of the embeddings on the LLM. Finally, we visualize and compare the output embeddings of the Bert models using PCA.

## Set Up
Install most requirements using:
```pip install -r requirements.txt```
also install

```torch-sparse```
```torch-scatter```

These libraries have strict requirements tho and the correct versions had to be specified manually (in my case).
I refer you to the [tutorial](https://colab.research.google.com/drive/1xpzn1Nvai1ygd_P5Yambc_oe4VBPK_ZT?usp=sharing#scrollTo=rwgNwoa26Eja) by torch geometrics, which I used as a template. This tutorial is also a good starting point to understand Graph Representation Learning in general.

## Dataset Transformation and Trainingsprocesses
![Overview of the transformation- and trainingsprocesses.](https://github.com/AhmadHAW/Hauptprojekt/blob/main/images/Overview.png?raw=true)
1.    The original (toy) dataset has user nodes and movie nodes, while the nodes are connected with edges from users to movies, if a user watched and rated the movie. In the first step we remove the rating and map new ids, so the graph is ready for link prediction.
2.    In the second step we train a GCNN on link prediction. The GCNN has to decide if any given edge exists in the original graph.
3.    With the original dataset and the dataset split we trained the GCNN on, we generate an NLP dataset, where the prompts for the LLM are the sum of the information we have about the user (only id) and the movie (title, genres and id). Again, the task of the LLM is to predict, if any given edge exists.
4.    Next with the trained GCNN, we produce node embeddings for every user and movie in their respective edge and split. For the user and movie embeddings for every split we fit a PCA and reduce the dimensionality to two. We end up with 8 fitted PCAs (2 for each split: train, val, test, rest)
5.    The reduced embeddings are now added to the prompt as a text, so the resulting dataset includes the graph embeddings.
6.    Two LLMs are now trained on predicting if a given edge exists, one on the dataset with embeddings and one without (vanilla).

## Playground
Follow the playground for experiments.
