import os
import sys
from Bio.KEGG import REST
import cPickle as pkl
from tqdm import tqdm

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from utils import get_human_regulation_TRRUST, get_human_pathways

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

    data, types = get_human_regulation_TRRUST(remove_unknown=True)
    genes1, genes2 = zip(*data.keys())


    G = nx.DiGraph()
    reg2color = {list(types)[0]:1 , list(types)[1]:2}
    genes_absent = 0
    for gene1, gene2 in data.keys():
        key = (gene1,gene2)
        G.add_node(gene1, color = 0)
        G.add_node(gene2, color = 0)
        G.add_edge(gene1, genes2, color=reg2color[data[key]])

    for gene in cancer_genes:
        if not gene in G.nodes():
            G.add_node(gene, color = 1)
        else:
            G.nodes()[gene]['color'] = 1
    

    found_neighbours = True
    while found_neighbours:
        found_neighbours = False
        for gene in cancer_genes:
            if gene in G.nodes():
                # neighbours = list(G.neighbors(gene))
                # if len(neighbours) == 0:
                #     continue
                # neighbours = neighbours[0]
                for neighbour in G[gene]:
                    print neighbour
                    n = G.nodes()[neighbour]
                    print n
                    if n['color']==0:
                        n['color'] = 0.5
                        found_neighbours = True

    node_list = list(G.nodes())
    edge_list = list(G.edges())
    print len(node_list), len(edge_list)

    for node in node_list:
        if not node in G.nodes(): continue
        n = G.nodes()[node]
        if len(n.keys())==0:
            # G.remove_node(node)
            continue
            # raise Exception('Something is wrong')
        if n['color'] == 0:
            print node
            G.remove_node(node)

    node_list = list(G.nodes())
    edge_list = list(G.edges())
    print len(node_list), len(edge_list)

    f = plt.figure()
    edge_dict = dict(nx.get_edge_attributes(G, 'color'))
    # print edge_dict
    node_dict = dict(nx.get_node_attributes(G, 'color'))
    pos=nx.spectral_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=list(node_dict.keys()), node_color=list(node_dict.values()), node_size=30)
    nx.draw_networkx_edges(G, pos, edgelist=list(edge_dict.keys()), edge_color=list(edge_dict.values()), arrowsize=1.0, arrows=True)
    
    plt.show()

        

    
    