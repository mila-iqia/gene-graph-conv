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
from models.mlp import MLP
from models.gcn import GCN
from models.slr import SLR
from models.utils import *

tasks = TCGAMeta(download=True)
graphs = {"genemania": GeneManiaGraph()}

# Setup the results dictionary
filename = "experiments/results/clinical-tasks.pkl"
try:
    results = pickle.load(open(filename, "rb"), encoding='latin1')
    print("Loaded Checkpointed Results")
except Exception as e:
    print(e)
    results = pd.DataFrame(columns=['task', 'acc_metric', 'model', 'graph', 'seed', 'train_size', 'time_elapsed'])
    print("Created a New Results Dictionary")

train_size = 50
test_size = 200
trials = 1
cuda = True
models = [
              #GCN(name="GCN_lay20_chan32_emb32_dropout_pool_kmeans", cuda=cuda, dropout=True, num_layer=4, channels=32, embedding=32, prepool_extralayers=5, pooling="kmeans"),
              #GCN(name="GCN_lay20_chan32_emb32_dropout_pool_hierarchy", cuda=cuda, dropout=True, num_layer=4, channels=32, embedding=32, prepool_extralayers=5, pooling="hierarchy"),
              #GCN(name="GCN_lay20_chan32_emb32_dropout_pool_random", cuda=cuda, dropout=True,num_layer=4, channels=32, embedding=32, prepool_extralayers=5, pooling="random"),
              # GCN(name="GCN_lay3_chan64_emb32_dropout_agg_hierarchy_reduce", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, aggregation="hierarchy", lr=0.001),
              # GCN(name="GCN_lay3_chan64_emb32_dropout_agg_hierarchy_reduce_1.2", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, aggregation="hierarchy", agg_reduce=1.2, lr=0.001),
              # GCN(name="GCN_lay3_chan64_emb32_dropout_agg_hierarchy_reduce_1.5", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, aggregation="hierarchy", agg_reduce=1.5, lr=0.001),
              # GCN(name="GCN_lay3_chan64_emb32_dropout_agg_hierarchy_reduce_2", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, aggregation="hierarchy", agg_reduce=2, lr=0.001),
              # GCN(name="GCN_lay3_chan64_emb32_dropout_agg_hierarchy_reduce_3", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, aggregation="hierarchy", agg_reduce=3, lr=0.001),
              # GCN(name="GCN_lay3_chan64_emb32_dropout_agg_hierarchy_reduce_5", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, aggregation="hierarchy", agg_reduce=5, lr=0.001),
              # GCN(name="GCN_lay3_chan64_emb32_dropout_agg_hierarchy_reduce_10", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, aggregation="hierarchy", agg_reduce=10, lr=0.001),
              # GCN(name="GCN_lay3_chan64_emb32_dropout_pool_hierarchy", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, pooling="hierarchy"),
              GCN(name="GCN_lay3_chan64_emb32_dropout", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, agg_reduce=2, lr=0.001),
              #MLP(name="MLP_lay2_chan512_dropout", cuda=cuda, dropout=True, num_layer=2, channels=512),
              #MLP(name="MLP_lay2_chan512", cuda=cuda, dropout=False, num_layer=2, channels=512),
              #SLR(name="SLR_lambda1_l11", cuda=cuda)
             ]

# Create the set of all experiment ids and see which are left to do
columns = ["task", "graph", "model", "seed", "train_size"]
all_exp_ids = [x for x in itertools.product(["".join(task.id) for task in tasks], graphs.keys(), [model.name for model in models], range(trials), [train_size])]
all_exp_ids = pd.DataFrame(all_exp_ids, columns=columns)
all_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in all_exp_ids.itertuples(name=None)]
results_exp_ids = results[columns].copy()
results_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in results_exp_ids.itertuples(name=None)]
intersection_ids = all_exp_ids.index.intersection(results_exp_ids.index)
todo = all_exp_ids.drop(intersection_ids).to_dict(orient="records")

print("todo: " + str(len(todo)))
print("done: " + str(len(results)))


for row in todo:
    start_time = time.time()
    print(len(results))
    graph_name = row["graph"]
    seed = row["seed"]
    model = [copy.deepcopy(model) for model in models if model.name == row["model"]][0]
    task = [copy.deepcopy(task) for task in tasks if "".join(task.id) == row["task"]][0]
    task._samples = task._samples - task._samples.mean(axis=0)
    task._samples = task._samples / task._samples.var()

    experiment = {
        "task": "".join(task.id),
        "model": model.name,
        "graph": graph_name,
        "seed": seed,
        "train_size": train_size,
    }
    print(experiment)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(task._samples, task._labels, stratify=task._labels, train_size=train_size, test_size=test_size)
    X_train = X_train.copy()
    X_test = X_test.copy()
    gene_graph = graphs[graph_name]
    adj = sparse.csr_matrix(nx.to_numpy_matrix(gene_graph.nx_graph))
    model.fit(X_train, y_train, adj=adj)

    y_hat = []
    for chunk in get_every_n(X_test, 10):
        y_hat.extend(np.argmax(model.predict(chunk), axis=1).numpy())
    experiment["acc_metric"] = model.metric(y_test, y_hat)
    experiment["time_elapsed"] = str(time.time() - start_time)
    experiment["cuda"] = cuda
    print(experiment)
    results = record_result(results, experiment, filename)
