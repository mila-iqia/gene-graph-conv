import numpy as np
import h5py
import percolate
import random
import logging
import networkx as nx
import gene_datasets
import pandas as pd
import itertools

class Graph(object):
    def __init__(self):
        pass

    def intersection_with(self, dataset):
        intersection = np.intersect1d(self.node_names, dataset.node_names)
        dataset.df = dataset.df[intersection]
        dataset.data = dataset.df.as_matrix()
        self.df = self.df[intersection].filter(items=intersection, axis='index')
        self.adj = self.df.as_matrix()

    def load_random_adjacency(self, nb_nodes, approx_nb_edges, scale_free=True):
        nodes = np.arange(nb_nodes)

        # roughly nb_edges edges (sorry, it's not exact, but heh)
        if scale_free:
            # Read: https://en.wikipedia.org/wiki/Scale-free_network
            # There is a bunch of bells and swittle, but after a few handwavy tests, the defaults parameters seems okay.
            edges = np.array(nx.scale_free_graph(nb_nodes).edges())
        else:
            edges = np.array([(i, ((((i + np.random.randint(nb_nodes - 1)) % nb_nodes) + 1) % nb_nodes))
                             for i in [np.random.randint(nb_nodes) for i in range(approx_nb_edges)]])

        # Adding self loop.
        edges = np.concatenate((edges, np.array([(i, i) for i in nodes])))

        # adjacent matrix
        A = np.zeros((nb_nodes, nb_nodes))
        A[edges[:, 0], edges[:, 1]] = 1.
        A[edges[:, 1], edges[:, 0]] = 1.
        self.adj = A
        self.df = pd.DataFrame(np.array(self.adj))
        self.node_names = list(range(nb_nodes))

    def load_graph(self, path):
        f = h5py.File(path, 'r')
        self.adj = np.array(f['graph_data']).astype('float32')
        self.node_names = np.array(f['gene_names'])
        self.df = pd.DataFrame(np.array(self.adj))
        self.df.columns = self.node_names
        self.df.index = self.node_names

    def generate_percolate(self, opt):
        self.nb_class = 2
        self.size_x = opt.size_perc
        self.size_y = opt.size_perc
        self.num_samples = opt.nb_examples
        self.extra_cn = opt.extra_cn  # uninformative connected layers of nodes
        self.disconnected = opt.disconnected  # number of nodes to disconnect
        size_x = self.size_x
        size_y = self.size_y
        prob = 0.562
        num_samples = self.num_samples
        extra_cn = self.extra_cn
        disconnected = self.disconnected

        if self.extra_cn != 0:
            if self.size_x != self.size_y:
                print "Not designed to add extra nodes with non-square graphs"

        np.random.seed(0)
        random.seed(0)

        expression_data = []
        labels_data = []
        for i in range(num_samples):
            if i % 10 == 0:
                logging.info("."),
            perc = False
            if i % 2 == 0:  # generate positive example
                perc = False
                while perc is False:
                    G, T, perc, dens, nio = percolate.sq2d_lattice_percolation_simple(size_x, size_y, prob=prob,
                                                                                      extra_cn=extra_cn, disconnected=disconnected)
                attrs = nx.get_node_attributes(G, 'value')
                features = np.zeros((len(attrs),), dtype='float32')
                for j, node in enumerate(nio):
                    features[j] = attrs[node]
                expression_data.append(features)
                labels_data.append(1)

            else:  # generate negative example
                perc = True
                while perc is True:
                    G, T, perc, dens, nio = percolate.sq2d_lattice_percolation_simple(size_x, size_y, prob=prob,
                                                                                      extra_cn=extra_cn, disconnected=disconnected)
                attrs = nx.get_node_attributes(G, 'value')
                features = np.zeros((len(attrs),), dtype='float32')
                for j, node in enumerate(nio):
                    features[j] = attrs[node]
                expression_data.append(features)
                labels_data.append(0)
        adj = nx.adjacency_matrix(G, nodelist=nio).todense()
        expression_data = np.asarray(expression_data)
        labels_data = np.asarray(labels_data)

        self.nio = nio
        self.adj = adj
        self.data = expression_data
        self.labels = labels_data
        self.node_names = range(0, self.data.shape[1])
        self.nb_class = 2


def get_path(graph):
    if graph == "kegg":
        return "/data/lisa/data/genomics/graph/kegg.hdf5"
    elif graph == "trust":
        return "/data/lisa/data/genomics/graph/trust.hdf5"
    elif graph == "pathway":
        return "/data/lisa/data/genomics/graph/pathway_commons.hdf5"
    elif graph == "pancan":
        return "/data/lisa/data/genomics/graph/pancan-tissue-graph.hdf5"


class EcoliEcocycGraph():

    def __init__(self, opt=None):

        d = pd.read_csv("data/ecocyc-21.5-pathways.col", sep="\t", skiprows=40,header=None)
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
