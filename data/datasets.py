import numpy as np
from torch.utils.data import Dataset
from graph import Graph


class Dataset(Dataset):
    def __init__(self, name):

        self.load_data()

    def load_data(self):
        raise NotImplementedError()

    def labels_name(self, l):
        raise NotImplementedError()

    def set_graph(self, opt):
        self.graph = Graph(opt, self)
        self.adj = self.graph.adj
        self.node_names = self.graph.node_names
        try:
            self.labels = self.graph.labels
            self.data = self.graph.data
        except Exception as e:
            print e

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

    def __init__(self, opt):
        super(RandomDataset, self).__init__(name='RandomDataset', opt=opt)

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

    def __init__(self, opt):
        super(PercolateDataset, self).__init__(name='PercolateDataset', opt=opt)

    def load_data(self):
        return

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = np.expand_dims(sample, -1)  # Addin a dim for the channels\
        sample = {'sample': sample, 'labels': self.labels[idx]}
        return sample

    def labels_name(self, l):
        labels = {0: 'neg', 'neg': 0, 'pos': 1, 1: 'pos'}
        return labels[l]
