"""
This file also needs the TCGA benchmark project available at https://github.com/mandanasmi/TCGA_Benchmark

"""

import meta_dataloader.TCGA
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from torch import Tensor
from torch import nn
import networkx as nx
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import math
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from genegraphconv.data.gene_graphs import StringDBGraph, HetIOGraph, FunCoupGraph, HumanNetV2Graph, GeneManiaGraph, \
    RegNetGraph


########################################################################################################################
# Define model
########################################################################################################################

class MaskedNetwork(torch.nn.Module):
    """
    One fully connected masked by adj matrix and then sigmoid and scalar product with the vector having 1 in every
    component
    """
    def __init__(self, input_dim, output_dim, adjacency_matrix=None):
        super(MaskedNetwork, self).__init__()
        self.weight = Parameter(torch.Tensor(input_dim, input_dim))
        self.bias = None  # Parameter(torch.Tensor(input_dim))
        self.adj = adjacency_matrix
        self.nonlin = torch.nn.ReLU()  # torch.nn.Sigmoid()
        self.second_layer = nn.Linear(input_dim, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.adj is not None:
            # return self.second_layer(self.sig(F.linear(x, self.weight * self.adj, self.bias)))
            return self.second_layer(self.nonlin(F.linear(x, self.weight * self.adj)))
        else:
            # return self.second_layer(self.sig(F.linear(x, self.weight, self.bias)))
            return self.second_layer(self.nonlin(F.linear(x, self.weight)))

########################################################################################################################
# Get adjacency matrix and corresponding data with right column ordering
########################################################################################################################


def getdata(graph_index, datastore="/network/home/bertinpa/Documents/gene-graph-conv/genegraphconv/data",
            covered_genes=None):
    """
    :param datastore: /Users/paul/Desktop/user1/PycharmProjects/gene-graph-conv/genegraphconv/data (local)
                    /network/home/bertinpa/Documents/gene-graph-conv/genegraphconv/data (server)
    :return:
    """
    if graph_names_list[graph_index] == "landmark":
        graph_name = graph_names_list[graph_index]
        print("Training with the", graph_names_list[graph_index], "graph")
        landmarkgene_path = "/network/home/bertinpa/Documents/gene-graph-conv/genegraphconv/data/datastore" \
                            "/random_landmark_genes_seed0.npy"
        landmark_genes = np.load(landmarkgene_path)
        is_landmark = [int(i in landmark_genes) for i in covered_genes]
        adj_matrix = np.array([is_landmark for i in covered_genes])
        adj_matrix = adj_matrix + adj_matrix.T + np.identity(len(covered_genes))
        adj_matrix = torch.Tensor(adj_matrix.astype(bool).astype(int))
        if torch.cuda.is_available():
            adj_matrix = adj_matrix.cuda()

    elif graph_names_list[graph_index]:
        print("Training with the", graph_names_list[graph_index], "graph")
        graph = graph_initializer_list[graph_index](datastore=datastore)
        graph_name = graph_names_list[graph_index]

        # Restrict to covered genes only
        if covered_genes is None:
            graph_genes = list(graph.nx_graph.nodes)
            dataset_genes = task.gene_ids
            covered_genes = list(set(graph_genes).intersection(dataset_genes))

        # Get subggraph
        subgraph = graph.nx_graph.subgraph(covered_genes)
        # subgraph_genes = list(subgraph.nodes)

        # Get Adjacency Matrix of the subgraph
        adj_matrix = torch.Tensor(np.array(nx.adjacency_matrix(subgraph, nodelist=covered_genes).todense()))
        adj_matrix += torch.eye(adj_matrix.shape[0])  # add diagonal
        if torch.cuda.is_available():
            adj_matrix = adj_matrix.cuda()

    else:
        print("Training without graph")
        graph_name = "none"
        adj_matrix = None

    # Get matrix with the columns in the same order as adjacency matrix
    X = pd.DataFrame(task._samples, columns=task.gene_ids)[covered_genes].to_numpy()
    y = np.array(task._labels)

    return graph_name, X, y, adj_matrix

########################################################################################################################
# Training pipeline
########################################################################################################################


def train(savedir="/network/home/bertinpa/Documents/TCGA_Benchmark/results/prediction_pipeline_losses4", fold=0,
          plot=False):
    train_loss_list = []
    test_loss_list = []
    test_pred_list = []
    cpt = 0

    for epoch in range(int(epochs)):
        print("\t\tEpoch", epoch, "over", epochs)
        # train
        for i, (data, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            outputs = model(data)
            loss = criterion(outputs[:, 0], label)
            loss.backward()
            train_loss_list.append((cpt, loss.item()))
            cpt += 1
            optimizer.step()

        # test
        for i, (data, label) in enumerate(test_dataloader):
            # Only one batch
            optimizer.zero_grad()
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            outputs = model(data)
            loss = criterion(outputs[:, 0], label)
            test_loss_list.append((cpt, loss.item()))
            label_pred = outputs.detach().cpu()
            test_pred_list.append((cpt, label.cpu(), label_pred.numpy()))
            # print("\t\t", np.bincount(np.array(label_pred)[:, 0]))
            # test_acc_list.append((cpt, (accuracy_score(label.cpu(), label_pred))))

    train_loss_list = np.array(train_loss_list)
    test_loss_list = np.array(test_loss_list)
    # Save
    np.save(os.path.join(savedir, "train_loss_list_" + graph_name + "_" + str(fold)), train_loss_list)
    np.save(os.path.join(savedir, "test_loss_list_" + graph_name + "_" + str(fold)), test_loss_list)
    np.save(os.path.join(savedir, "test_pred_list_" + graph_name + "_" + str(fold)), test_pred_list)

    # Plot
    if plot:
        plt.ylim(0, 2)
        plt.plot(train_loss_list[:, 0], train_loss_list[:, 1], label="train")
        plt.plot(test_loss_list[:, 0], test_loss_list[:, 1], label="test")
        plt.legend()
        plt.show()

########################################################################################################################
# Main
########################################################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for prediction pipeline')
    parser.add_argument('learning_rate', type=float)
    parser.add_argument(
        'savedir',
        default="/network/home/bertinpa/Documents/TCGA_Benchmark/results/prediction_pipeline_losses_test")

    args = parser.parse_args()

    batch_size = 32
    epochs = 100
    learning_rate = args.learning_rate
    savedir = args.savedir

    print("learning rate", learning_rate, "saving in", savedir)

    ####################################################################################################################
    # Evaluate simple classification pipeline on a specific task
    ####################################################################################################################

    task = meta_dataloader.TCGA.TCGATask(('PAM50Call_RNAseq', 'BRCA'))
    covered_genes = np.load("/network/home/bertinpa/Documents/TCGA_Benchmark/data/covered_genes.npy")

    ####################################################################################################################
    # List of graphs
    ####################################################################################################################

    graph_initializer_list = [StringDBGraph, HetIOGraph, FunCoupGraph, HumanNetV2Graph, GeneManiaGraph]  # RegNetGraph]
    graph_names_list = ["stringdb", "hetio", "funcoup", "humannet", "genemania", "landmark", None]  # "regnet"

    # graph_index = 0  # Chose a graph in the list ny its index

    for graph_index in range(7):

        ################################################################################################################
        # Get data
        ################################################################################################################

        graph_name, X, y, M = getdata(graph_index, covered_genes=covered_genes)

        print("data of shape", X.shape)

        # import pdb
        # pdb.set_trace()

        ################################################################################################################
        # Prepare data + train test split
        ################################################################################################################

        # Turn it into a binary classification (all against type 2)
        y = np.array([int(i == 2) for i in y])

        # # Usual train test split
        # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,
        #                                                                             y,
        #                                                                             stratify=y,
        #                                                                             train_size=0.8,
        #                                                                             shuffle=True,
        #                                                                             random_state=0)

        # Cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fold = 0

        for train_index, test_index in skf.split(X, y):
            fold += 1
            print("\tComputing fold", fold)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = MaskedNetwork(X.shape[1], X.shape[1], M)
            if torch.cuda.is_available():
                model = model.cuda()

            # Loss
            criterion = torch.nn.BCEWithLogitsLoss()

            # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            train_set = TensorDataset(Tensor(X_train), Tensor(y_train))
            test_set = TensorDataset(Tensor(X_test), Tensor(y_test))

            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)

            ############################################################################################################
            # Train
            ############################################################################################################

            train(savedir=savedir,
                  fold=fold, plot=False)
            # /Users/paul/Desktop/user1/PycharmProjects/
            # /network/home/bertinpa/Documents/
