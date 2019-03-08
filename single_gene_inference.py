import pickle
import argparse
import traceback

import pandas as pd
import numpy as np

import itertools
import sklearn
import torch

from models.mlp import MLP
from data import datasets
from data.gene_graphs import GeneManiaGraph, RegNetGraph, HumanNetV2Graph, FunCoupGraph
from data.utils import record_result

parser = argparse.ArgumentParser()
parser.add_argument('--graph', default=None, type=str, help='Which graph to evaluate? Default - no graph, all nodes used')
parser.add_argument('--results', default='all_nodes', type=str, help='Name of file to save results to. Default - all_nodes')
args = parser.parse_args()

# Setup the results dictionary
filename = "./experiments/results/{}.pkl".format(args.results)
try:
    with open(filename, 'rb') as f:
        results = pickle.load(f, encoding='latin1')
    print("Loaded Checkpointed Results")
except Exception as e:
    print(e)
    results = pd.DataFrame(columns=['auc', 'gene', 'model', 'graph', 'seed', 'train_size', 'error'])
    print("Created a New Results Dictionary - {}".format(filename))


train_size = 50
test_size = 1000
trials = 3
cuda = torch.cuda.is_available()

graph_dict = {"regnet": RegNetGraph, "genemania": GeneManiaGraph, 
              "humannetv2":HumanNetV2Graph, "funcoup":FunCoupGraph}

# Select graph and set variables
if args.graph:
    # Check graph arg is valid
    assert args.graph in graph_dict.keys()
    graph_name = args.graph
    gene_graph = graph_dict[graph_name]()
    is_first_degree = True
else:
    is_first_degree = False
    graph_name = "all_nodes"

# Read in data
dataset = datasets.TCGADataset()

# Create list of the genes to perform inference on
# If assessing first-degree neighbours, then train only for those genes 
# that are there in the graph 
which_genes = dataset.df.columns.tolist()
if args.graph:
    which_genes = set(gene_graph.nx_graph.nodes).intersection(which_genes)


# Create the set of all experiment ids and see which are left to do
columns = ["gene", "seed"]
all_exp_ids = [x for x in itertools.product(which_genes, range(trials))]
all_exp_ids = pd.DataFrame(all_exp_ids, columns=columns)
all_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in all_exp_ids.itertuples(name=None)]
results_exp_ids = results[columns].copy()
results_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in results_exp_ids.itertuples(name=None)]
intersection_ids = all_exp_ids.index.intersection(results_exp_ids.index)
todo = all_exp_ids.drop(intersection_ids).to_dict(orient="records")

print("todo: " + str(len(todo)))
print("done: " + str(len(results)))


for row in todo:
    if len(results) % 10 == 0:
        print(len(results))
    gene = row["gene"]
    seed = row["seed"]
    model = MLP(column_names=dataset.df.columns, dropout=False, cuda=cuda, metric=sklearn.metrics.roc_auc_score)

    experiment = {
        "gene": gene,
        "model": "Basic_MLP",
        "graph": graph_name,
        "seed": seed,
        "train_size": train_size,
    }
    dataset.labels = dataset.df[gene].where(dataset.df[gene] > 0).notnull().astype("int")
    dataset.labels = dataset.labels.values if type(dataset.labels) == pd.Series else dataset.labels

    try:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels, stratify=dataset.labels, 
                             train_size=train_size, test_size=test_size)
    except ValueError:
        results = record_result(results, experiment, filename)
        continue
    if is_first_degree:
        neighbors = list(gene_graph.first_degree(gene)[0])
        neighbors = [n for n in neighbors if n in X_train.columns.values]
        X_train = X_train.loc[:, neighbors].copy()
        X_test = X_test.loc[:, neighbors].copy()
    else:
        X_train = X_train.copy()
        X_test = X_test.copy()
    X_train[gene] = 1
    X_test[gene] = 1
    try:
        model.fit(X_train, y_train)
        if cuda:
            X_test = X_test.cuda()
        y_hat = model.predict(X_test)
        auc = sklearn.metrics.roc_auc_score(y_test, np.argmax(y_hat, axis=1))
        experiment["auc"] = auc
        model.best_model = None # cleanup
        del model
        torch.cuda.empty_cache()
    except Exception:
        tb = traceback.format_exc()
        experiment['error'] = tb
    
    results = record_result(results, experiment, filename)
