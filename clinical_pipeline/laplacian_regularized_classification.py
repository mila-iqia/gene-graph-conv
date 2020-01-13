"""
This file also needs the TCGA benchmark project available at https://github.com/mandanasmi/TCGA_Benchmark

"""

import meta_dataloader.TCGA
import sklearn.model_selection
import sys
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from torch import Tensor
import networkx as nx
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from genegraphconv.data.gene_graphs import StringDBGraph, HetIOGraph, FunCoupGraph, HumanNetV2Graph, GeneManiaGraph, \
    RegNetGraph

graph_initializer_list = [StringDBGraph, HetIOGraph, FunCoupGraph, HumanNetV2Graph, GeneManiaGraph, RegNetGraph]
graph_names_list = ["stringdb", "hetio", "funcoup", "humannet", "genemania", "regnet"]

graph_index = 0  # Chose a graph in the list ny its index

# for graph_index in range(6):

########################################################################################################################
# Evaluate simple classification pipeline on a specific task
########################################################################################################################

task = meta_dataloader.TCGA.TCGATask(('PAM50Call_RNAseq', 'BRCA'))  # ('_EVENT', 'LUNG'))

########################################################################################################################
# Load a graph, get laplacian and data and get a consistent column ordering
########################################################################################################################

graph = graph_initializer_list[graph_index](
    datastore="/Users/paul/Desktop/user1/PycharmProjects/gene-graph-conv/genegraphconv/data")
graph_name = graph_names_list[graph_index]

# Sets of nodes
graph_genes = list(graph.nx_graph.nodes)
dataset_genes = task.gene_ids
intersection_genes = list(set(graph_genes).intersection(dataset_genes))

# Get subggraph
subgraph = graph.nx_graph.subgraph(intersection_genes)
subgraph_genes = list(subgraph.nodes)

# Get Laplacian of the subgraph
L = torch.tensor(nx.laplacian_matrix(subgraph).todense()).float()

# Get matrix with the columns in the same order as Laplacian
X = pd.DataFrame(task._samples, columns=dataset_genes)[subgraph_genes].to_numpy()
y = task._labels

########################################################################################################################
# Prepare data
########################################################################################################################

# Turn it into a binary classification (all against type 2)
y = [int(i == 2) for i in y]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,
                                                                            y,
                                                                            stratify=y,
                                                                            train_size=0.8,
                                                                            shuffle=True,
                                                                            random_state=0)


########################################################################################################################
# Define model
########################################################################################################################


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


batch_size = 64
epochs = 70
Lambda = 1
learning_rate = 0.00001

model = LogisticRegression(X.shape[1], 1)
criterion = torch.nn.BCEWithLogitsLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_set = TensorDataset(Tensor(X_train), Tensor(y_train))
test_set = TensorDataset(Tensor(X_test), Tensor(y_test))

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)

########################################################################################################################
# Train
########################################################################################################################

train_loss_list = []
test_loss_list = []
test_acc_list = []
cpt = 0

for epoch in range(int(epochs)):
    # train
    for i, (data, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs[:, 0], label) + Lambda * torch.matmul(model.linear.weight,
                                                              torch.matmul(L, model.linear.weight.T))
        loss.backward()
        train_loss_list.append((cpt, loss.item()))
        cpt += 1
        optimizer.step()

    # test
    for i, (data, label) in enumerate(test_dataloader):
        # Only one batch
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs[:, 0], label)
        test_loss_list.append((cpt, loss.item()))
        test_acc_list.append((cpt, (accuracy_score(label, (outputs.detach() > 0).int()))))


train_loss_list = np.array(train_loss_list)
test_loss_list = np.array(test_loss_list)
# Save
np.save("/Users/paul/PycharmProjects/TCGA_Benchmark/results/prediction_pipeline_losses/train_loss_list_" + graph_name,
        train_loss_list)
np.save("/Users/paul/PycharmProjects/TCGA_Benchmark/results/prediction_pipeline_losses/test_loss_list_" + graph_name,
        test_loss_list)
np.save("/Users/paul/PycharmProjects/TCGA_Benchmark/results/prediction_pipeline_losses/test_acc_list_" + graph_name,
        test_acc_list)

# Plot
# plt.ylim(0, 2)
# plt.plot(train_loss_list[:, 0], train_loss_list[:, 1], label="train")
# plt.plot(test_loss_list[:, 0], test_loss_list[:, 1], label="test")
# plt.legend()
# plt.show()
