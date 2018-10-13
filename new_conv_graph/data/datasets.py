import numpy as np
from torch.utils.data import Dataset
import graph
from graph import Graph


class Dataset(Dataset):
    def __init__(self, name, seed, nb_class, nb_examples, nb_nodes, nb_master_nodes=0):

        self.name = name
        self.seed = seed
        self.nb_class = nb_class
        self.nb_examples = nb_examples
        self.nb_nodes = nb_nodes
        self.load_data()
        self.df = self.df - self.df.mean(0)
        self.nb_master_nodes = nb_master_nodes

        for master in range(nb_master_nodes):
            self.df.insert(0, 'master_{}'.format(master), 1.)
        self.data = self.df.as_matrix()

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

    def __init__(self, seed, nb_class=None, nb_examples=None, nb_nodes=None):
        nb_class = nb_class if nb_class is not None else 2
        nb_examples = nb_examples if nb_examples is not None else 100
        nb_nodes = nb_nodes if nb_nodes is not None else 100
        super(RandomDataset, self).__init__(name='RandomDataset', seed=seed, nb_class=nb_class, nb_examples=nb_examples, nb_nodes=nb_nodes)

    def load_data(self):
        np.random.seed(self.seed)

        # Generating the data
        self.data = np.random.randn(self.nb_examples, self.nb_nodes, 1)
        self.labels = (np.sum(self.data, axis=1) > 0.)[:, 0].astype(np.long)  # try to predict if the sum. is > than 0.

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = [sample, self.labels[idx]]
        return sample

    def labels_name(self, l):
        labels = {0: 'neg', 'neg': 0, 'pos': 1, 1: 'pos'}
        return labels[l]
