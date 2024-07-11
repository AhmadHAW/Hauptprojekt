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


## Playground
Follow the playground for experiments.