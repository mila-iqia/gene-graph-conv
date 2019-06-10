#!/usr/bin/env python
# coding: utf-8
import os
import copy
import time
import sys
import pickle
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import itertools
from torch.autograd import Variable
import sklearn, sklearn.model_selection, sklearn.metrics
import numpy as np
from scipy import sparse
from models.mlp import MLP
from models.gcn import GCN
from models.slr import SLR
from models.utils import *
from data import datasets
from data.gene_graphs import GeneManiaGraph
from data.utils import record_result
from orion.client import report_results
from argparser import parse_args


def main(argv=None):
    opt = parse_args(argv)
    dataset = datasets.TCGADataset()
    dataset.df = dataset.df - dataset.df.mean(axis=0)

    gene_graph = GeneManiaGraph()
    search_num_genes=[50, 100, 200, 300, 500, 1000, 2000, 4000, 8000, 16300]
    test_size=300
    cuda = torch.cuda.is_available()
    exp = []
    for num_genes in search_num_genes:
        start_time = time.time()
        gene = "RPL4"
        model = GCN(cuda=cuda, dropout=opt.dropout, num_layer=opt.num_layer, channels=opt.channels, embedding=opt.embedding, aggregation=opt.aggregation, lr=opt.lr, agg_reduce=opt.agg_reduce)
        dataset.labels = dataset.df[gene].where(dataset.df[gene] > 0).notnull().astype("int")
        dataset.labels = dataset.labels.values if type(dataset.labels) == pd.Series else dataset.labels
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels, stratify=dataset.labels, train_size=opt.train_size, test_size=opt.test_size, random_state=opt.seed)
        if num_genes == 16300:
            neighbors = gene_graph.nx_graph
        else:
            neighbors = gene_graph.bfs_sample_neighbors(gene, num_genes)

        X_train = X_train[list(neighbors.nodes)].copy()
        X_test = X_test[list(neighbors.nodes)].copy()
        X_train[gene] = 1
        X_test[gene] = 1
        adj = sparse.csr_matrix(nx.to_numpy_matrix(neighbors))
        model.fit(X_train, y_train, adj=adj)

        y_hat = model.predict(X_test)
        y_hat = np.argmax(y_hat, axis=1)
        auc = sklearn.metrics.roc_auc_score(y_test, np.asarray(y_hat).flatten())
        del model
        exp.append(auc)
    report_results([{"name": "auc", "type": "objective", "value": np.array(exp).mean()}])

if __name__ == '__main__':
    main(sys.argv[1:])
