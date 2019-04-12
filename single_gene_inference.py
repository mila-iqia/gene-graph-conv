import pickle
import argparse
import traceback

import pandas as pd
import numpy as np

import itertools
import sklearn
import torch

from models.mlp import MLP
from data.datasets import TCGADataset, GTexDataset, GEODataset
from data.gene_graphs import GeneManiaGraph, RegNetGraph,HumanNetV1Graph, HumanNetV2Graph, \
    FunCoupGraph, HetIOGraph, StringDBGraph
from data.utils import record_result
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--graph', default=None, type=str,
                    help='Which graph to evaluate? Default - no graph, all nodes used')
parser.add_argument('--dataset', default='tcga', type=str,
                    help='Gene expression data to be used. Default - TCGA. Valid options are tcga, gtex, geo')
parser.add_argument('--edge', default=None, type=str,
                    help='Edge type for HetIO graph [interaction, regulation, covariation] and Stringdb ['
                         'neighborhood, fusion, cooccurence, coexpression, experimental, database, textmining, all]')
parser.add_argument('--results', default='all_nodes', type=str,
                    help='Name of file to save results to. Default - all_nodes')
parser.add_argument('--seed', default=0, type=int, help='Seed for training')
parser.add_argument('--rand', default=False, type=bool, help='Randomize graph ?')

args = parser.parse_args()
print(args)
seed = args.seed
randomize = args.rand

# Setup the results dictionary
filename = "./experiments/results/{}_seed{}.pkl".format(args.results, seed)
try:
    with open(filename, 'rb') as f:
        results = pickle.load(f, encoding='latin1')
    print("Loaded Checkpointed Results")
except Exception as e:
    print(e)
    results = pd.DataFrame(columns=['auc', 'gene', 'model', 'graph', 'seed', 'train_size', 'error'])
    print("Created a New Results Dictionary - {}".format(filename))


train_size = 2000
test_size = 1000
cuda = torch.cuda.is_available()

graph_dict = {"regnet": RegNetGraph, "genemania": GeneManiaGraph, "humannetv1": HumanNetV1Graph,
              "humannetv2": HumanNetV2Graph, "funcoup": FunCoupGraph,
              "hetio": HetIOGraph, "stringdb": StringDBGraph, "landmark": None}

# Select graph and set variables
if args.graph:
    # Check graph arg is valid
    assert args.graph in graph_dict.keys()
    if args.graph == 'landmark':
        landmark_genes = np.load("data/datastore/landmarkgenes.npy")
        is_first_degree = True
        is_landmark = True
        graph_name = "landmark"
    else:
        graph_name = args.graph
        if graph_name == 'hetio' or graph_name == 'stringdb':
            gene_graph = graph_dict[graph_name](graph_type=args.edge, randomize=randomize)
        else:
            gene_graph = graph_dict[graph_name](randomize=randomize)
        is_first_degree = True
        is_landmark = False
else:
    is_first_degree = False
    is_landmark = False
    graph_name = "all_nodes"

# Read in data
try:
    assert args.dataset in ['tcga', 'gtex', 'geo']
    if args.dataset == 'tcga':
        dataset = TCGADataset()
    elif args.dataset == 'gtex':
        dataset = GTexDataset()
    elif args.dataset == 'geo':
        dataset = GEODataset(file_path='/network/data1/genomics/D-GEX/bgedv2.hdf5', 
                             seed=seed, load_full=False, nb_examples=40000)

except Exception:
    tb = traceback.format_exc()
    print(tb)
    print("Please enter a valid argument for the dataset. Valid options are tcga, gtex and geo")
    import sys
    sys.exit()

# Create list of the genes to perform inference on
# If assessing first-degree neighbours, then train only for those genes 
# that are there in the graph 
which_genes = dataset.df.columns.tolist()
if args.graph and args.graph != 'landmark':
    which_genes = set(gene_graph.nx_graph.nodes).intersection(which_genes)
if args.graph and args.graph == 'landmark':
    landmark_genes = [x for x in which_genes if x in landmark_genes]
    print("Number of covered landmark genes", len(landmark_genes))

print("Number of covered genes", len(which_genes))

# Create the set of all experiment ids and see which are left to do
columns = ["gene"]
all_exp_ids = [x for x in itertools.product(which_genes)]
all_exp_ids = pd.DataFrame(all_exp_ids, columns=columns)
all_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in all_exp_ids.itertuples(name=None)]
results_exp_ids = results[columns].copy()
results_exp_ids.index = ["-".join(map(str, tup[1:])) for tup in results_exp_ids.itertuples(name=None)]
intersection_ids = all_exp_ids.index.intersection(results_exp_ids.index)
todo = all_exp_ids.drop(intersection_ids).sort_values(by='gene').to_dict(orient="records")

print("todo: " + str(len(todo)))
print("done: " + str(len(results)))

name = "{}_seed{}.pkl".format(args.results, seed)
for row in tqdm(todo, desc=name):
#    if len(results) % 100 == 0:
#        print("\nRemaining: {}".format(len(results)))
    gene = row["gene"]
    seed = seed
    model = MLP(column_names=dataset.df.columns, num_layer=1, dropout=False,
                train_valid_split=0.5, cuda=cuda, metric=sklearn.metrics.roc_auc_score,
                channels=16, batch_size=10, lr=0.0007, weight_decay=0.00000001,
                verbose=False, patience=5, num_epochs=10, seed=seed,
                full_data_cuda=True, evaluate_train=False)

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
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels,
                                                                                    stratify=dataset.labels,
                                                                                    train_size=train_size,
                                                                                    test_size=test_size,
                                                                                    random_state=seed)

    except ValueError:
        results = record_result(results, experiment, filename)
        continue
    if is_first_degree:
        if is_landmark:
            neighbors = landmark_genes
        else:
            neighbors = list(gene_graph.first_degree(gene)[0])
            neighbors = [n for n in neighbors if n in X_train.columns.values]
        X_train = X_train.loc[:, neighbors].copy()
        X_test = X_test.loc[:, neighbors].copy()
    else:
        X_train = X_train.copy()
        X_test = X_test.copy()

    try:
        X_train[gene] = 1
        X_test[gene] = 1
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        auc = sklearn.metrics.roc_auc_score(y_test, np.argmax(y_hat, axis=1))
        experiment["auc"] = auc
        model.best_model = None  # cleanup
        del model
        torch.cuda.empty_cache()
    except Exception:
        tb = traceback.format_exc()
        experiment['error'] = tb

    results = record_result(results, experiment, filename)