#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from models.models import MLP
from data import datasets
from data.gene_graphs import GeneManiaGraph, RegNetGraph
from data.utils import record_result
from data.clinical.datasets import TCGADataset, Task
from data.clinical import taskloader
from data.clinical import split_dataset

from models.models import MLP, GCN, SLR

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


tcga = TCGADataset()
task_ids = taskloader.get_all_tasks(tcga)


# In[3]:


tasks = [Task(tcga, task_id, limit=1000) for task_id in task_ids]


# In[4]:


graphs = {"genemania": GeneManiaGraph()}


# In[5]:


# Setup the results dictionary
filename = "experiments/results/clinical-tasks.pkl"
try:
    results = pickle.load(open(filename, "rb"), encoding='latin1')
    print("Loaded Checkpointed Results")
except Exception as e:
    print(e)
    results = pd.DataFrame(columns=['task', 'auc', 'model', 'graph', 'seed', 'train_size', 'time_elapsed'])
    print("Created a New Results Dictionary")


# In[10]:


train_size = 50
test_size = 200
trials = 3
cuda = False
models = [
              #GCN(name="GCN_lay20_chan32_emb32_dropout_pool_kmeans", cuda=cuda, dropout=True, num_layer=4, channels=32, embedding=32, prepool_extralayers=5, pooling="kmeans"),
              GCN(name="GCN_lay20_chan32_emb32_dropout_pool_hierarchy", cuda=cuda, dropout=True, num_layer=4, channels=32, embedding=32, prepool_extralayers=5, pooling="hierarchy"),
              #GCN(name="GCN_lay20_chan32_emb32_dropout_pool_random", cuda=cuda, dropout=True,num_layer=4, channels=32, embedding=32, prepool_extralayers=5, pooling="random"),
              GCN(name="GCN_lay3_chan64_emb32_dropout_pool_hierarchy", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32, pooling="hierarchy"),
              GCN(name="GCN_lay3_chan64_emb32_dropout", cuda=cuda, dropout=True, num_layer=3, channels=64, embedding=32),
              MLP(name="MLP_lay2_chan512_dropout", cuda=cuda, dropout=True, num_layer=2, channels=512),
              MLP(name="MLP_lay2_chan512", cuda=cuda, dropout=False, num_layer=2, channels=512),
              #SLR(name="SLR_lambda1_l11", cuda=cuda)
             ]


# In[11]:


# Create the set of all experiment ids and see which are left to do
columns = ["task", "graph", "model", "seed", "train_size"]
all_exp_ids = [x for x in itertools.product([task.id for task in tasks], graphs.keys(), [model.name for model in models], range(trials), [train_size])]
all_exp_ids = pd.DataFrame(all_exp_ids, columns=columns)
all_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in all_exp_ids.itertuples(name=None)]
results_exp_ids = results[columns].copy()
results_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in results_exp_ids.itertuples(name=None)]
intersection_ids = all_exp_ids.index.intersection(results_exp_ids.index)
todo = all_exp_ids.drop(intersection_ids).to_dict(orient="records")

print("todo: " + str(len(todo)))
print("done: " + str(len(results)))


# In[12]:


def get_every_n(a, n=2):
    for i in range(a.shape[0] // 2):
        yield a[2*i:2*(i+1)]


# In[ ]:


for row in todo:
    start_time = time.time()
    import pdb; pdb.set_trace()
#    if len(results) % 10 == 0:
    print(len(results))
    graph_name = row["graph"]
    seed = row["seed"]
    model = [copy.deepcopy(model) for model in models if model.name == row["model"]][0]
    task = [copy.deepcopy(task) for task in tasks if task.id == row["task"]][0]
    test_size = min(test_size, len(task.labels - train_size))
    experiment = {
        "task": task.id,
        "model": model.name,
        "graph": graph_name,
        "seed": seed,
        "train_size": train_size,
    }
    print(experiment)
    try:
        X_train, X_test, y_train, y_test = sklearn.model_selection.            train_test_split(task.get_data(), task.labels, stratify=task.labels, 
                             train_size=train_size, test_size=test_size)
    except ValueError as e:
        print(e)
        results = record_result(results, experiment, filename)
        continue

    X_train = X_train.copy()
    X_test = X_test.copy()
    gene_graph = graphs[graph_name]
    adj = np.asarray(nx.to_numpy_matrix(gene_graph.nx_graph))
    try:
        model.fit(X_train, y_train.astype("uint8"), adj=adj)
        x_test = Variable(torch.FloatTensor(np.expand_dims(X_test.values, axis=2)), requires_grad=False).float()
        if cuda:
            x_test = x_test.cuda()

        y_hat = []
        for chunk in get_every_n(x_test, 10):
            y_hat.extend(model.predict(chunk)[:,1].data.cpu().numpy().tolist())
        auc = sklearn.metrics.roc_auc_score(y_test, np.asarray(y_hat).flatten())
        model.best_model = None # cleanup
        experiment["auc"] = auc
        experiment["time_elapsed"] = str(time.time() - start_time)
        experiment["cuda"] = cuda
        print(experiment)
        results = record_result(results, experiment, filename)
    except Exception as e:
        print(e)
    


# In[12]:


results.groupby(["task", "model"]).mean()


# In[14]:


(results.groupby(["model"]).mean(), results.groupby(["model"]).var())


# In[19]:


len(results.task.unique())


# In[26]:


subset = results

q = subset.groupby(['model'])['auc']
print(q.mean())


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (7.5, 3.6)
plot_train_size = 50

subset = results

q = subset.groupby(['model'])['auc']

stderr = []
mean = []
labels = []
for model in set(subset.model):
    labels.append(model)
    mean.append(q.mean()[model])
    stderr.append(q.std()[model]/np.sqrt(q.count()[model]))

freq_series = pd.Series.from_array(mean)

plt.figure(figsize=(12, 8))
fig = freq_series.plot(kind='bar')

plt.bar(x=range(len(set(subset.model))), height=mean, yerr=stderr)
fig.set_xticklabels(labels)
plt.xticks(rotation=-80)

fig.set_ylim((0.5, 0.9))


# In[ ]:




