"""
This file also needs the TCGA benchmark project available at https://github.com/mandanasmi/TCGA_Benchmark

File that saves a list of genes that correspond to genes covered by all graphs (but Regnet) in a given dataset
"""
import meta_dataloader.TCGA
import numpy as np
from genegraphconv.data.gene_graphs import StringDBGraph, HetIOGraph, FunCoupGraph, HumanNetV2Graph, GeneManiaGraph, \
    RegNetGraph

####################################################################################################################
# Evaluate simple classification pipeline on a specific task
####################################################################################################################

task = meta_dataloader.TCGA.TCGATask(('PAM50Call_RNAseq', 'BRCA'))  # ('_EVENT', 'LUNG'))
datastore = "/Users/paul/Desktop/user1/PycharmProjects/gene-graph-conv/genegraphconv/data"

####################################################################################################################
# List of graphs
####################################################################################################################

graph_initializer_list = [StringDBGraph, HetIOGraph, FunCoupGraph, HumanNetV2Graph, GeneManiaGraph, RegNetGraph]
graph_names_list = ["stringdb", "hetio", "funcoup", "humannet", "genemania", "regnet"]
all_sets_of_genes = []
covered_genes = set(task.gene_ids)
print(len(covered_genes))

for graph_index in range(5):
    # Loop over all graphs but regnet
    graph = graph_initializer_list[graph_index](datastore=datastore)

    # Computing interestection
    covered_genes = covered_genes.intersection(graph.nx_graph.nodes)
    print(len(covered_genes))

np.save("/Users/paul/PycharmProjects/TCGA_Benchmark/data/covered_genes",
        list(covered_genes))
