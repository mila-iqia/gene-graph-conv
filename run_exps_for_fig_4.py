# This file generates the data for Figure #5 from the paper https://arxiv.org/pdf/1806.06975.pdf

import pickle
import argparse
import os
import networkx as nx
import pandas as pd


from models.ml_methods import MLMethods
import data
import data.gene_datasets
from data.graph import Graph, get_hash

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bucket-idx', default=0, type=int, help='which bucket is this.')
    parser.add_argument('--num-buckets', default=0, type=int, help='How many buckets are there?')
    parser.add_argument('--exp-name', default=0, type=str, help='which exp dir is this.')
    parser.add_argument('--graph', default=0, type=str, help='which graph is this.')
    return parser


def parse_args(argv):
    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv
    return opt


def main(argv=None):
    opt = parse_args(argv)
    tcgatissue = data.gene_datasets.TCGATissue()
    graph = Graph()
    graph.load_graph(get_hash(opt.graph))
    nx_graph = nx.from_numpy_matrix(graph.adj)
    mapping = dict(zip(range(0, len(tcgatissue.df.columns)), tcgatissue.df.columns))
    nx_graph = nx.relabel_nodes(g, mapping)

    results_file = "experiments/results/" + opt.exp_name + '/results-' + str(opt.bucket_idx) + '.pkl'
    try:
        results = pickle.load(open(results_file, "r"))
    except Exception:
        results = {"df": pd.DataFrame(columns=['auc', 'gene_name', 'model', 'num_genes', 'seed', 'train_size'])}

    tcgatissue = data.gene_datasets.TCGATissue()
    tcgatissue.df = tcgatissue.df - tcgatissue.df.mean()
    bucket_size = tcgatissue.df.shape[-1] / opt.num_buckets
    start = opt.bucket_idx - 1 * bucket_size
    end = (opt.bucket_idx) * bucket_size

    df = tcgatissue.df.copy(deep=True)
    genes_to_iter = tcgatissue.df.iloc[:, start:end].columns.difference(results['df']['gene_name'].unique())
    num_all_genes = len(tcgatissue.df.columns)

    methods = [{'key': 'MLP', 'method': MLMethods("MLP", dropout=False, cuda=False)}, ]

    for gene in genes_to_iter:
        tcgatissue.df = df[:]
        method_comparison(results, tcgatissue, methods, gene=gene, num_genes=50, trials=3, train_size=50, test_size=1000, file_to_write=results_file, nx_graph=nx_graph)
        tcgatissue.df = df[:]
        method_comparison(results, tcgatissue, methods, gene=gene, num_genes=num_all_genes, trials=3, train_size=50, test_size=1000, file_to_write=results_file, nx_graph=nx_graph)


def method_comparison(results, dataset, methods, gene, num_genes, trials, train_size, test_size, file_to_write=None, nx_graph=None):
    mean = dataset.df[gene].mean()
    dataset.labels = [1 if x > mean else 0 for x in dataset.df[gene]]
    if num_genes != len(dataset.df.columns):
        neighbors = set([gene])
        neighbors = neighbors.union(set(nx_graph.neighbors(gene)))
        dataset.df = dataset.df[list(neighbors)]

    dataset.df[gene] = 1
    dataset.data = dataset.df.as_matrix()
    neighborhood = None
    for method in methods:
        for seed in range(trials):
            already_done = results["df"][(results["df"].gene_name == gene) &
                                         (results["df"].model == model['key']) &
                                         (results["df"].num_genes == num_genes) &
                                         (results["df"].seed == seed) &
                                         (results["df"].train_size == train_size)].shape[0] > 0

            if already_done:
                print "already done:", method['key'], num_genes, seed
                continue
            print "doing:", method['key'], num_genes, seed
            result = method['method'].loop(dataset=dataset, seed=seed, train_size=train_size, test_size=test_size, adj=neighborhood)

            experiment = {
                "gene_name": gene,
                "model": model['key'],
                "num_genes": num_genes,
                "seed": seed,
                "train_size": train_size,
                "auc": result
                }

            results["df"] = results["df"].append(experiment, ignore_index=True)
            print results

            dir = "/".join(file_to_write.split('/')[0:-1])
            if not os.path.isdir(dir):
                os.makedirs(dir)
            pickle.dump(results, open(file_to_write, "wb"))


if __name__ == '__main__':
    main()
