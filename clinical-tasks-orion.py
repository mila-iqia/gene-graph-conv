#!/usr/bin/env python
# coding: utf-8

import time
import os
import sys
import copy
import pickle
import networkx as nx
import pandas as pd
import numpy as np

import itertools
import sklearn
import torch
import datetime
import matplotlib, matplotlib.pyplot as plt
from collections import defaultdict
from scipy import sparse

from torch.autograd import Variable
from data import datasets
from data.gene_graphs import GeneManiaGraph, RegNetGraph
from data.utils import record_result
import meta_dataloader.TCGA as TCGA
from meta_dataloader.TCGA import TCGAMeta
from meta_dataloader.utils import stratified_split
from models.mlp import MLP
from models.gcn import GCN
from models.slr import SLR
from models.utils import *
from orion.client import report_results
from argparser import parse_args

def main(argv=None):
    opt = parse_args(argv)

    tasks = TCGAMeta(download=True, preload=True)
    task = tasks[113]

    # Setup the results dictionary
    filename = "experiments/results/clinical-tasks.pkl"
    try:
        results = pickle.load(open(filename, "rb"), encoding='latin1')
        print("Loaded Checkpointed Results")
    except Exception as e:
        print(e)
        results = pd.DataFrame(columns=['task', 'acc_metric', 'model', 'graph', 'trial', 'train_size', 'time_elapsed'])
        print("Created a New Results Dictionary")

    train_size = 50
    trials = 3
    cuda = True
    exp = []

    for trial in range(trials):
        model = GCN(cuda=cuda, dropout=opt.dropout, num_layer=opt.num_layer, channels=opt.channels, embedding=opt.embedding, aggregation=opt.aggregation, lr=opt.lr, agg_reduce=opt.agg_reduce, seed=trial)
        task._samples = task._samples - task._samples.mean(axis=0)
        task._samples = task._samples / task._samples.var()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(task._samples, task._labels, stratify=task._labels, train_size=train_size, test_size=len(task._labels) - train_size)
        adj = sparse.csr_matrix(nx.to_numpy_matrix(GeneManiaGraph().nx_graph))
        model.fit(X_train, y_train, adj=adj)

        y_hat = []
        for chunk in get_every_n(X_test, 10):
            y_hat.extend(np.argmax(model.predict(chunk), axis=1).numpy())

        exp.append(model.metric(y_test, y_hat))
        print(exp)
    report_results([{"name": "acc_metric", "type": "objective", "value": np.array(exp).mean()}])

if __name__ == '__main__':
    main(sys.argv[1:])
