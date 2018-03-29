import numpy as np
from torch.utils.data import Dataset
from graph import Graph


class Dataset(Dataset):
    def __init__(self, name, opt, transform=None):

        self.name = name
        self.seed = opt.seed
        self.nb_class = 2 if opt.nb_class is None else opt.nb_class
        self.nb_examples = 1000 if opt.nb_examples is None else opt.nb_examples
        self.nb_nodes = opt.nb_nodes
        self.load_data()
        self.set_graph(opt)
        self.transform = transform

        if opt.graph is not None and opt.neighborhood is not 'all':
            self.set_graph(opt)
            import pdb; pdb.set_trace()
            self.adj = (self.adj > 0.).astype(float)
            self.nb_nodes = self.adj.shape[0]
        elif opt.graph is not None:
            self.set_graph(opt)
            self.adj = (self.adj > 0.).astype(float)
            self.nb_nodes = self.adj.shape[0]
        if opt.center:
            self.data = self.data - self.data.mean(axis=0)  # Ugly, to redo.

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
        sample = [sample, self.labels[idx]]

        if self.transform is not None:
            sample = self.transform(sample)

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
        sample = [sample, self.labels[idx]]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def labels_name(self, l):
        labels = {0: 'neg', 'neg': 0, 'pos': 1, 1: 'pos'}
        return labels[l]
