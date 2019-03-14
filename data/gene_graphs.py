""" This file contains the wrapper around our gene interaction graph, which is essentially a big adjacency matrix"""

import csv
import numpy as np
import pandas as pd
import h5py
import networkx as nx
import academictorrents as at
from data.utils import symbol_map, ncbi_to_hugo_map
import os


class GeneInteractionGraph(object):
    """ This class manages the data pertaining to the relationships between genes.
        It has an nx_graph, and some helper functions.
    """
    def __init__(self, relabel_genes=True):
        self.load_data()
        self.nx_graph = nx.relabel.relabel_nodes(self.nx_graph, symbol_map(self.nx_graph.nodes))

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
        neighbors = nx.OrderedGraph()
        if include_self:
            neighbors.add_node(gene)
        bfs = nx.bfs_edges(self.nx_graph, gene)
        for u, v in bfs:
            if neighbors.number_of_nodes() == num_neighbors:
                break
            neighbors.add_node(v)

        for node in neighbors.nodes():
            for u, v, d in self.nx_graph.edges(node, data="weight"):
                if neighbors.has_node(u) and neighbors.has_node(v):
                    neighbors.add_weighted_edges_from([(u, v, d)])
        return neighbors

    def adj(self):
        return nx.to_numpy_matrix(self.nx_graph)

class RegNetGraph(GeneInteractionGraph):
    def __init__(self, at_hash="e109e087a8fc8aec45bae3a74a193922ce27fc58", datastore=""):
        self.at_hash = at_hash
        self.datastore = datastore
        super(RegNetGraph, self).__init__()

    def load_data(self):
        self.nx_graph = nx.OrderedGraph(nx.readwrite.gpickle.read_gpickle(at.get(self.at_hash, datastore=self.datastore)))


class GeneManiaGraph(GeneInteractionGraph):
    def __init__(self, at_hash="5adbacb0b7ea663ac4a7758d39250a1bd28c5b40", datastore=""):
        self.at_hash = at_hash
        self.datastore = datastore
        super(GeneManiaGraph, self).__init__()

    def load_data(self):
        self.nx_graph = nx.OrderedGraph(nx.readwrite.gpickle.read_gpickle(at.get(self.at_hash, datastore=self.datastore)))


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


class EvolvedGraph(GeneInteractionGraph):
    def __init__(self, adjacency_path):
        """
        Given an adjacency matrix, builds the correponding GeneInteractionGraph
        :param adjacency_path: Path to numpy array (N_nodes, N_nodes)
        """
        self.adjacency_path = adjacency_path
        super(EvolvedGraph, self).__init__()

    def load_data(self):
        self.nx_graph = nx.OrderedGraph(nx.from_numpy_matrix(np.load(self.adjacency_path)))


class HumanNetV1Graph(GeneInteractionGraph):
    """
    More info on HumanNet V1 : http://www.functionalnet.org/humannet/about.html
    """
    def __init__(self):
        self.benchmark = "../data/graphs/HumanNet.v1.benchmark.txt"
        super(HumanNetV1Graph, self).__init__()

    def load_data(self):
        edgelist = pd.read_csv(self.benchmark, header=None, sep="\t").values.tolist()
        self.nx_graph = nx.OrderedGraph(edgelist)
        # Map nodes from ncbi to hugo names
        self.nx_graph = nx.relabel.relabel_nodes(self.nx_graph, ncbi_to_hugo_map(self.nx_graph.nodes))
        # Remove nodes which are not covered by the map
        for node in list(self.nx_graph.nodes):
            if isinstance(node, int):
                self.nx_graph.remove_node(node)


class HumanNetV2Graph(GeneInteractionGraph):
    """
    More info on HumanNet V1 : http://www.functionalnet.org/humannet/about.html
    """
    def __init__(self):
        self.benchmark = "../data/graphs/HumanNet-XN.tsv"
        super(HumanNetV2Graph, self).__init__()

    def load_data(self):
        edgelist = pd.read_csv(self.benchmark, header=None, sep="\t", skiprows=1).values[:, :2].tolist()
        self.nx_graph = nx.OrderedGraph(edgelist)
        # Map nodes from ncbi to hugo names
        self.nx_graph = nx.relabel.relabel_nodes(self.nx_graph, ncbi_to_hugo_map(self.nx_graph.nodes))
        # Remove nodes which are not covered by the map
        for node in list(self.nx_graph.nodes):
            if isinstance(node, float):
                self.nx_graph.remove_node(node)


class FunCoupGraph(GeneInteractionGraph):
    """
    Class for loading and processing FunCoup into a NetworkX object
    Please download the data file - 'FC4.0_H.sapiens_full.gz' from
    http://funcoup.sbc.su.se/downloads/ and place it in the 
    graphs folder before instantiating this class
    """
    def __init__(self, filename='funcoup.pkl'):
        self.filename = filename
        super(FunCoupGraph, self).__init__()

    def load_data(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.location = os.path.join(dir_path, 'graphs/')
        pkl_file = os.path.join(self.location, self.filename)
        if not os.path.isfile(pkl_file):
            self._preprocess_and_pickle(save_name=pkl_file)
        self.nx_graph = nx.OrderedGraph(nx.read_gpickle(pkl_file))
        
            
    def _preprocess_and_pickle(self, save_name):
        names_map_file =  os.path.join(self.location, 'ensembl_to_hugo.tsv')
        data_file = os.path.join(self.location,'FC4.0_H.sapiens_full.gz')
        
        names = pd.read_csv(names_map_file, sep='\t')
        names.columns = ['symbol', 'ensembl']
        names = names.dropna(subset=['ensembl']).drop_duplicates('ensembl')
        names = names.set_index('ensembl').squeeze()

        data = pd.read_csv(data_file, sep='\t', usecols=['#0:PFC', '1:FBS_max',
                                                         '2:Gene1','3:Gene2'])
        data['2:Gene1'] = data['2:Gene1'].map(names)
        data['3:Gene2'] = data['3:Gene2'].map(names)
        data = data.dropna(subset=['2:Gene1', '3:Gene2'])
        
        graph = nx.from_pandas_edgelist(data, source='2:Gene1', target='3:Gene2', 
                                       # edge_attr=['#0:PFC', '1:FBS_max'], # Uncomment to include edge attributes
                                        create_using=nx.OrderedGraph)
        nx.write_gpickle(graph, save_name)


class HetIOGraph(GeneInteractionGraph):
    """
    Class for the HetIO graph. More information about HetIO can be found on - 
    github.com/hetio/hetionet
    het.io
    """
    def __init__(self, graph_type='interaction'):
        name_to_edge = {'interaction':'GiG', 'regulation':'Gr>G', 'covariation':'GcG',
                        'all': 'GiG|Gr>G|GcG'}
        assert graph_type in name_to_edge.keys()
        self.graph_type = graph_type
        self.edge = name_to_edge[graph_type]
        self.filename = 'hetio_{}_graph.pkl'.format(graph_type)
        super(HetIOGraph, self).__init__()

    def load_data(self):
        self.location = os.path.join(__file__, './graphs/')
        pkl_file = os.path.join(self.location, self.filename)
        if not os.path.isfile(pkl_file):
            self._process_and_pickle(save_name=pkl_file)
        self.nx_graph = nx.OrderedGraph(nx.read_gpickle(pkl_file))
        
            
    def _process_and_pickle(self, save_name):
        names_map_file =  os.path.join(self.location, 'hetionet-v1.0-nodes.tsv')
        data_file = os.path.join(self.location, 'hetionet-v1.0-edges.sif.gz')
        if not(os.path.isfile(names_map_file) or os.path.isfile(data_file)):
            print(""" Please download the files from https://github.com/hetio/hetionet/tree/master/hetnet/tsv:
            
            -- hetionet-v1.0-nodes.tsv
            -- hetionet-v1.0-edges.sif.gz""")
            import sys;sys.exit()

              
        node_ids = pd.read_csv(names_map_file, sep='\t')
        node_ids = node_ids.loc[node_ids.kind == 'Gene'].drop(columns='kind')
        node_ids = node_ids.set_index('id').squeeze()
        
        edges = pd.read_csv(data_file, sep='\t', compression='gzip')
        edges = edges.loc[edges.metaedge.str.contains(self.edge)].drop(columns='metaedge')
        
        # Convert the HetIO Entrez IDs into gene symbols
        edges = edges.stack().map(node_ids).unstack()
        edges = edges.dropna().drop_duplicates() 

        graph = nx.from_pandas_edgelist(edges, create_using=nx.OrderedGraph)
        nx.write_gpickle(graph, save_name)
        