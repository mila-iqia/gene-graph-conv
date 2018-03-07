import logging
import numpy as np
from torch.utils.data import Dataset
import random
import percolate
import networkx as nx


class Dataset(Dataset):
    def __init__(self, name, graph, opt):

        self.name = name
        self.graph = graph
        self.adj = graph.adj
        self.nb_nodes = graph.nb_nodes
        self.node_names = graph.node_names

        self.seed = opt.seed
        self.nb_class = 2 if opt.nb_class is None else opt.nb_class
        self.nb_examples = 1000 if opt.nb_examples is None else opt.nb_examples

        self.load_data()
        self.adj = (self.adj > 0.).astype(float)  # Don't care about the weights, for now.
        if opt.center:
            self.data = self.data - self.data.mean(axis=0)  # Ugly, to redo.

    def load_data(self):
        raise NotImplementedError()

    def labels_name(self, l):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __len__(self):
        return self.data.shape[0]

    def get_adj(self):
        return self.adj


class RandomDataset(Dataset):

    """
    A random dataset for debugging purposes
    """

    def __init__(self, graph, opt):
        super(RandomDataset, self).__init__(name='RandomDataset', graph=graph, opt=opt)

    def load_data(self):
        np.random.seed(self.seed)

        # Generating the data
        self.data = np.random.randn(self.nb_examples, self.nb_nodes, 1)
        self.labels = (np.sum(self.data, axis=1) > 0.)[:, 0].astype(np.long)  # try to predict if the sum. is > than 0.

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = {'sample': sample, 'labels': self.labels[idx]}
        return sample

    def labels_name(self, l):
        labels = {0: 'neg', 'neg': 0, 'pos': 1, 1: 'pos'}
        return labels[l]


class PercolateDataset(Dataset):

    """
    A random dataset where the goal if to find if we can percolate from one side of the graph to the other.
    """

    def __init__(self, graph, opt):
        self.nb_class = 2
        self.size_x = opt.size_perc
        self.size_y = opt.size_perc
        self.num_samples = 100 if opt.nb_examples is None else opt.nb_examples
        self.extra_cn = opt.extra_cn  # uninformative connected layers of nodes
        self.disconnected = opt.disconnected  # number of nodes to disconnect

        super(PercolateDataset, self).__init__(name='PercolateDataset', graph=graph, opt=opt)

    def load_data(self):
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

        self.nb_class = 2
        self.nb_nodes = self.data.shape[1]

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = np.expand_dims(sample, -1)  # Addin a dim for the channels

        sample = {'sample': sample, 'labels': self.labels[idx]}
        return sample

    def labels_name(self, l):
        labels = {0: 'neg', 'neg': 0, 'pos': 1, 1: 'pos'}
        return labels[l]
