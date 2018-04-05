import sys
import sklearn, sklearn.model_selection, sklearn.metrics, sklearn.linear_model, sklearn.neural_network, sklearn.tree
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from utils import get_second_degree


def infer_gene(method, genes, gene_to_infer, g):
    labels = [1 if x > genes[gene_to_infer].mean() else 0 for x in genes[gene_to_infer]]

    # TODO: fix this hack. There are some genes
    if sum(labels) < 2:
        labels[0] = 1

    temp_genes = genes.drop(gene_to_infer, axis=1)

    first_degree_neighbors = temp_genes.loc[:, list(g.neighbors(gene_to_infer))].dropna(axis=1)
    second_degree_neighbors = temp_genes.loc[:, list(get_second_degree(gene_to_infer, g))].dropna(axis=1)

    full_results = method(temp_genes, labels, 10, 1000)

    if len(first_degree_neighbors.columns) == 0:
        first_degree_results = (.5, .0)
    else:
        first_degree_results = method(first_degree_neighbors, labels, 10, 1000)

    if len(second_degree_neighbors.columns) == 0:
        second_degree_results = (.5, .0)
    else:
        second_degree_results = method(second_degree_neighbors, labels, 10, 1000)

    first_degree_diff = full_results[0] - first_degree_results[0]
    second_degree_diff = full_results[0] - second_degree_results[0]

    data = {"gene_name": gene_to_infer,
            "auc": full_results[0],
            "std": full_results[1],
            "first_degree_auc": first_degree_results[0],
            "first_degree_std": first_degree_results[1],
            "first_degree_diff": first_degree_diff,
            "second_degree_diff": second_degree_diff,
            "second_degree_auc": second_degree_results[0],
            "second_degree_std": second_degree_results[1],
           }
    return pd.DataFrame(data, [0])

def infer_all_genes(method, genes, g):
    results = pd.DataFrame(columns=["gene_name",
                                    "auc",
                                    "std",
                                    "first_degree_auc",
                                    "first_degree_std",
                                    "second_degree_auc",
                                    "second_degree_std",
                                    "first_degree_diff",
                                    "second_degree_diff"])
    print "Genes to do:" + str(len(g.nodes))
    sys.stdout.write("Trial number:")
    for index, gene in enumerate(genes):
        data = infer_gene(method, genes, gene, g)
        results = results.append(pd.DataFrame(data, index=range(0, len(data))))
        sys.stdout.write(str(index) + ", ")
    results.to_csv('results.csv')
    return results
