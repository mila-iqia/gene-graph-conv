#!/usr/bin/env python
# coding: utf-8
import os
import copy
import time
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


dataset = datasets.TCGADataset()
dataset.df = dataset.df - dataset.df.mean(axis=0)

# Setup the results dictionary
filename = "experiments/results/fig-5.pkl"
try:
    results = pickle.load(open(filename, "rb"), encoding='latin1')
    print("Loaded Checkpointed Results")
except Exception as e:
    results = pd.DataFrame(columns=['auc', 'gene', 'model', 'num_genes', 'seed', 'train_size', 'time_elapsed'])
    print("Created a New Results Dictionary")

gene_graph = GeneManiaGraph()

search_num_genes=[16300]#[50, 100, 200, 300, 500, 1000, 2000, 4000, 8000, 16300]
test_size=300
search_train_size=[50]
cuda=True
trials=3
search_genes = ["RPL4"]
models = [
             # GCN(name="GCN_lay20_chan32_emb32_dropout_pool_hierarchy", cuda=cuda, dropout=True, num_layer=4, channels=32, embedding=32, prepool_extralayers=5, aggregation="hierarchy"),
              GCN(name="GCN_lay3_chan16_emb32_dropout_agg_hierarchy_prepool_1", cuda=cuda, dropout=True, num_layer=3, channels=16, embedding=32, prepool_extralayers=1, aggregation="hierarchy"),
              GCN(name="GCN_lay3_chan64_emb32_dropout_agg_hierarchy_reduce_5_schedule", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, aggregation="hierarchy", agg_reduce=5, lr=0.001, patience=15, scheduler=True), 
              GCN(name="GCN_lay3_chan64_emb32_dropout_agg_hierarchy_reduce_5", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, aggregation="hierarchy", agg_reduce=5),
              GCN(name="GCN_lay3_chan64_emb32_dropout_agg_hierarchy", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, aggregation="hierarchy"),
              #GCN(name="GCN_lay20_chan32_emb32_dropout_pool_random", cuda=cuda, num_layer=4, channels=32, embedding=32, prepool_extralayers=5, aggregation="random"),
              GCN(name="GCN_lay3_chan64_emb32_dropout", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32),
              MLP(name="MLP_lay2_chan512_dropout", cuda=cuda, dropout=True, num_layer=2, channels=512),
              MLP(name="MLP_lay2_chan512", cuda=cuda, dropout=False, num_layer=2, channels=512),
              SLR(name="SLR_lambda1_l11", cuda=cuda)
             ]

# Create the set of all experiment ids and see which are left to do
model_names = [model.name for model in models]
columns = ["gene", "model", "num_genes", "train_size", "seed"]
all_exp_ids = [x for x in itertools.product(search_genes, model_names, search_num_genes, search_train_size, range(trials))]
all_exp_ids = pd.DataFrame(all_exp_ids, columns=columns)
all_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in all_exp_ids.itertuples(name=None)]
results_exp_ids = results[columns].copy()
results_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in results_exp_ids.itertuples(name=None)]
intersection_ids = all_exp_ids.index.intersection(results_exp_ids.index)
todo = all_exp_ids.drop(intersection_ids).to_dict(orient="records")
print("todo: " + str(len(todo)))
print("done: " + str(len(results)))



for row in todo:
    print(row)
    start_time = time.time()
    gene = row["gene"]
    model_name = row["model"]
    seed = row["seed"]
    num_genes = row["num_genes"] if row["num_genes"] < 10000 else 16300
    train_size = row["train_size"]
    model = [copy.deepcopy(model) for model in models if model.name == row["model"]][0]
    experiment = {
        "gene": gene,
        "model": model.name,
        "num_genes": num_genes,
        "train_size": train_size,
        "seed": seed,
    }

    dataset.labels = dataset.df[gene].where(dataset.df[gene] > 0).notnull().astype("int")
    dataset.labels = dataset.labels.values if type(dataset.labels) == pd.Series else dataset.labels
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels, stratify=dataset.labels, train_size=train_size, test_size=test_size, random_state=seed)
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

    x_test = Variable(torch.FloatTensor(np.expand_dims(X_test.values, axis=2)), requires_grad=False).float()
    if cuda:
        x_test = x_test.cuda()

    y_hat = []
    for chunk in get_every_n(x_test, 10):
        y_hat.extend(model.predict(chunk)[:,1].data.cpu().numpy().tolist())
    auc = sklearn.metrics.roc_auc_score(y_test, np.asarray(y_hat).flatten())
    del model
    experiment["auc"] = auc
    experiment["time_elapsed"] = str(time.time() - start_time)
    results = record_result(results, experiment, filename)
    print(experiment)
