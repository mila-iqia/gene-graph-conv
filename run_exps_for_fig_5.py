# This file generates the data for Figure #5 from the paper https://arxiv.org/pdf/1806.06975.pdf
import pickle
import argparse
import os
import networkx as nx
import pandas as pd
import numpy as np

from models.ml_methods import MLMethods
import data
import data.gene_datasets
from data.graph import Graph, sample_neighbors


def main(argv=None):
    # Generates the data for figure #5
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', type=str)
    parser.add_argument('--graph-path', type=str)
    parser.add_argument('--tcgatissue-full-path', type=str)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--cuda', action="store_true", help='If we want to run on gpu.')
    opt = parser.parse_args(argv)

    data_dir = '/'.join(opt.tcgatissue_full_path.split('/')[:-1])
    data_file = opt.tcgatissue_full_path.split('/')[-1]
    tcgatissue = data.gene_datasets.TCGATissue(data_dir=data_dir, data_file=data_file)

    opt.seed = 0
    opt.nb_class = None
    opt.nb_examples = None
    opt.nb_nodes = None
    opt.graph = "genemania"
    opt.dataset = tcgatissue
    opt.add_self = True
    opt.norm_adj = True
    opt.add_connectivity = False
    opt.pool_graph = "ignore"
    genes = ["RPL13", "HLA-B", "S100A9", "IFIT1", "RPL5", "RPS31", "ZFP82", "IL5", "DLGAP2"]
    if opt.gene:
        genes = [opt.gene]


    graph = Graph()
    if opt.graph_path:
        graph.load_graph(opt.graph_path)
    else:
        graph.load_graph(data.graph.get_hash(opt.graph))
    nx_graph = nx.from_numpy_matrix(graph.adj)
    mapping = dict(zip(range(0, len(tcgatissue.df.columns)), tcgatissue.df.columns))
    nx_graph = nx.relabel_nodes(nx_graph, mapping)

    methods = [
        {'key': 'CGN_lay3_chan64_emb32_dropout', 'method': MLMethods("CGN", num_layer=3, num_channel=64, add_emb=32, cuda=opt.cuda)},
        {'key': 'MLP_lay2_chan512_dropout', 'method': MLMethods("MLP", dropout=True, num_layer=2, num_channel=512, cuda=opt.cuda)},
        {'key': 'MLP_lay2_chan512', 'method': MLMethods("MLP", dropout=False, num_layer=2, num_channel=512, cuda=opt.cuda)},
        {'key': 'SLR_lambda1_l11', 'method': MLMethods("SLR", cuda=opt.cuda)},
        ]

    for gene in genes:
        df = tcgatissue.df.copy(deep=True)

        if not os.path.exists("results_" + gene + ".pkl"):
            empty_df = pd.DataFrame(columns=['auc','gene_name', 'model', 'num_genes', 'seed', 'train_size'])
            results = {"df": empty_df}
        else:
            results = pickle.load(open("results_" + gene + ".pkl", "r"))

        tcgatissue.df = df[:]
        method_comparison(results, tcgatissue, methods, gene=gene,
                          search_num_genes=[50, 100,200,300,500,1000,2000,4000,8000,16000],
                          trials=opt.trials,
                          search_train_size=[50],
                          test_size=1000,
                          nx_graph=nx_graph,
                          graph_name=opt.graph)


def method_comparison(results, dataset, methods, gene, search_num_genes, trials, search_train_size, test_size, nx_graph, graph_name):

    dataset = data.gene_datasets.TCGATissue()
    dataset.df = dataset.df - dataset.df.mean()

    mean = dataset.df[gene].mean()
    dataset.labels = [1 if x > mean else 0 for x in dataset.df[gene]]
    full_df = dataset.df.copy(deep=True)

    for train_size in search_train_size:
        for ex in search_num_genes:

            num_genes = ex
            num_genes = np.min([num_genes, dataset.df.shape[1]])
            print ex, num_genes

            neighbors = sample_neighbors(nx_graph, gene, num_genes, include_self=True)
            print "neighbors", len(neighbors), "train_size", train_size

            dataset.df = dataset.df[list(neighbors)]
            dataset.df[gene] = 1
            dataset.data = dataset.df.as_matrix()

            neighborhood = np.asarray(nx.to_numpy_matrix(nx.Graph(nx_graph.subgraph(neighbors))))
            for method in methods:
                for seed in range(trials):
                    already_done = results["df"][(results["df"].gene_name == gene) &
                                                 (results["df"].model == method['key']) &
                                                 (results["df"].num_genes == num_genes) &
                                                 (results["df"].seed == seed) &
                                                 (results["df"].train_size == train_size)]

                    if already_done.shape[0] > 0:
                        print("already done:", method['key'], "num_genes", num_genes, "train_size",
                              train_size, "seed", seed)
                        continue
                    print("doing:", method['key'], "num_genes", num_genes, "train_size",
                          train_size, "seed", seed)

                    result = method['method'].loop(dataset=dataset,
                                                   seed=seed,
                                                   train_size=train_size,
                                                   test_size=test_size,
                                                   adj=neighborhood,
                                                   graph_name=graph_name)

                    experiment = {"gene_name": gene,
                                  "model": method['key'],
                                  "num_genes": num_genes,
                                  "seed": seed,
                                  "train_size": train_size,
                                  "auc": result
                                 }

                    results["df"] = results["df"].append(experiment, ignore_index=True)
                    exp_dir = "experiments/results/fig-5/"
                    if not os.path.isdir(exp_dir):
                        os.makedirs(exp_dir)
                    results_file = exp_dir + "results_" + str(gene) + '.pkl'
                    pickle.dump(results, open(results_file, "wb"))
            dataset.df = full_df

if __name__ == '__main__':
    main()
