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
    def __init__(self):
        self.load_data()

    def load_data(self):
        raise NotImplementedError

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

class RegNetGraph(GeneInteractionGraph):
    def __init__(self, at_hash="3c8ac6e7ab6fbf962cedb77192177c58b7518b23", datastore=""):
        self.at_hash = at_hash
        self.datastore = datastore
        super(RegNetGraph, self).__init__()

    def load_data(self):
        h5_file = h5py.File(at.get(self.at_hash, datastore=self.datastore))
        self.node_names = np.array(h5_file['gene_names'])
        self.df = pd.DataFrame(np.array(np.array(h5_file['graph_data']).astype('float32')))
        self.df.columns = self.node_names
        self.df.index = self.node_names
        self.nx_graph = nx.from_pandas_adjacency(self.df)


class GeneManiaGraph(GeneInteractionGraph):
    def __init__(self, at_hash="2b7e470d87a017be3826ef6ec55893f3a61f5af7", datastore=""):
        self.at_hash = at_hash
        self.datastore = datastore
        super(GeneManiaGraph, self).__init__()

    def load_data(self):
        # You could replace the value of self.hash with a path to a local copy of your graph and AT can handle that.
        h5_file = h5py.File(at.get(self.at_hash, datastore=self.datastore))
        self.node_names = np.array(h5_file['gene_names'])
        self.df = pd.DataFrame(np.array(np.array(h5_file['graph_data']).astype('float32')))
        self.df.columns = self.node_names
        self.df.index = self.node_names
        self.nx_graph = nx.from_pandas_adjacency(self.df)


class EcoliEcocycGraph(GeneInteractionGraph):
    def __init__(self, path):  # data/ecocyc-21.5-pathways.col
        self.path = path
        super(EcoliEcocycGraph, self).__init__()

    def load_data(self):
        d = pd.read_csv(self.path, sep="\t", skiprows=40,header=None)
        d = d.set_index(0)
        del d[1]
        d = d.loc[:,:110] # filter gene ids

        # collect global names for nodes so all adj are aligned
        node_names = d.as_matrix().reshape(-1).astype(str)
        node_names = np.unique(node_names[node_names!= "nan"]) # nan removal

        #stores all subgraphs and their pathway names
        adjs = []
        adjs_name = []
        # for each pathway create a graph, add the edges and create a matrix
        for i, name in enumerate(d.index):
            G=nx.Graph()
            G.add_nodes_from(node_names)
            pathway_genes = np.unique(d.iloc[i].dropna().astype(str).as_matrix())
            for e1, e2 in itertools.product(pathway_genes, pathway_genes):
                G.add_edge(e1, e2)
            adj = nx.to_numpy_matrix(G)
            adjs.append(adj)
            adjs_name.append(name)

        #collapse all graphs to one graph
        adj = np.sum(adjs,axis=0)
        adj = np.clip(adj,0,1)

        self.adj = adj
        self.adjs = adjs
        self.adjs_name = adjs_name
