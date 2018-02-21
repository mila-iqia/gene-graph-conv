import os
import sys
from Bio.KEGG import REST
import cPickle as pkl
from tqdm import tqdm

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from utils import get_human_regulation_TRRUST, get_human_pathways, get_human_regulation_RegNetwork

def get_cancer_genes_KEGG(pathway_dict):
    '''
    Extracts the names of genes involved in cancer
    '''
    cancer_genes = []
    pathway_file =  pathway_dict['path:hsa05200']
    
    current_section = None
    for line in pathway_file.rstrip().split("\n"):
        section = line[:12].strip()  # section names are within 12 columns
        if not section == "":
            current_section = section

        if current_section == "GENE":
            gene_identifiers, gene_description = line[12:].split("; ")
            gene_id, gene_symbol = gene_identifiers.split()

            if not gene_symbol in cancer_genes:
                cancer_genes.append(gene_symbol)

    return cancer_genes

if __name__=='__main__':
    
    pathway_dict = get_human_pathways()   
    cancer_genes = get_cancer_genes_KEGG(pathway_dict)

    dataTRRUST, types = get_human_regulation_TRRUST(remove_unknown=False)
    # genes1, genes2 = zip(*data.keys())

    dataRegN = get_human_regulation_RegNetwork()
    # genes1, genes2 = zip(*data.keys())

    G = nx.DiGraph()
    # reg2color = {list(types)[0]:1 , list(types)[1]:2}
    # genes_absent = 0
    for gene1, gene2 in dataTRRUST.keys() + dataRegN:
        key = (gene1,gene2)
        G.add_node(gene1, color = 0)
        G.add_node(gene2, color = 0)
        G.add_edge(gene1, gene2, color=1)

    Gp = G.copy()
    for gene in G.nodes():
        if not gene in cancer_genes:
            Gp.remove_node(gene)
    G = Gp.copy()
    
    node_list = list(G.nodes())
    edge_list = list(G.edges())
    print len(node_list), len(edge_list)

    f = plt.figure()
    edge_dict = dict(nx.get_edge_attributes(G, 'color'))
    node_dict = dict(nx.get_node_attributes(G, 'color'))
    pos=nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=list(node_dict.keys()), node_color=list(node_dict.values()), node_size=10)
    nx.draw_networkx_edges(G, pos, edgelist=list(edge_dict.keys()), edge_color=list(edge_dict.values()), arrowsize=0.1, arrows=True)   
    # plt.show()

    print node_list
    M = nx.to_numpy_array(G)
    f = plt.figure()
    plt.imshow(M)
    # plt.show()

    with open('processed_data/data.pkl', 'w') as fout:
        pkl.dump( (node_list, M), fout)

        

    
    