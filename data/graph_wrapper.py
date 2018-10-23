""" This file contains the wrapper around our gene interaction graph, which is essentially a big adjacency matrix"""

import numpy as np
import pandas as pd
import h5py
import networkx as nx
import academictorrents as at


class GeneInteractionGraph(object):
    """ This class manages the data pertaining to the relationships between genes.
        It has an nx_graph, and some helper functions.
    """
    def __init__(self, path):
        h5_file = h5py.File(at.get(path))
        self.node_names = np.array(h5_file['gene_names'])
        self.df = pd.DataFrame(np.array(np.array(h5_file['graph_data']).astype('float32')))
        self.df.columns = self.node_names
        self.df.index = self.node_names
        self.nx_graph = nx.from_pandas_adjacency(self.df)

    @classmethod
    def get_at_hash(cls, graph_name):
        # This maps between the natural name of a graph and its Academic Torrents hash
        at_hash = ""
        if graph_name == "regnet":
            at_hash = "3c8ac6e7ab6fbf962cedb77192177c58b7518b23"
        elif graph_name == "genemania":
            at_hash = "ae2691e4f4f068d32f83797b224eb854b27bd3ee"
        return at_hash

    def first_degree(self, gene):
        neighbors = set([gene])
        # If the node is not in the graph, we will just return that node
        try:
            neighbors = neighbors.union(set(self.nx_graph.neighbors(gene)))
        except Exception as e:
            #print(e)
            pass
        neighborhood = np.asarray(nx.to_numpy_matrix(self.nx_graph.subgraph(neighbors)))
        return neighbors, neighborhood

    def bfs_sample_neighbors(self, gene, num_neighbors, include_self=True):
        results = set([])
        if include_self:
            results = set([gene])
        bfs = nx.bfs_edges(self.nx_graph, gene)
        for _, sink in bfs:
            if len(results) == num_neighbors:
                break
            results.add(sink)
        return results
