import logging
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import h5py
import networkx
import pandas as pd
import collections
import random


class Graph(object):
    def __init__(self, opt):
        if opt.scale_free:
            self.load_random_adjacency(nb_nodes=100, approx_nb_edges=100, scale_free=opt.scale_free)
        else:
            self.load_graph(opt.graph_path)
        self.nb_nodes = self.adj.shape[0]

    def load_random_adjacency(self, nb_nodes, approx_nb_edges, scale_free=True):
        nodes = np.arange(nb_nodes)

        # roughly nb_edges edges (sorry, it's not exact, but heh)
        if scale_free:
            # Read: https://en.wikipedia.org/wiki/Scale-free_network
            # There is a bunch of bells and swittle, but after a few handwavy tests, the defaults parameters seems okay.
            edges = np.array(networkx.scale_free_graph(nb_nodes).edges())
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
        self.node_names = list(range(nb_nodes))

    def load_graph(self, path):
        f = h5py.File(path, 'r')
        self.labels = f['labels_data']
        self.adj = np.array(f['graph_data']).astype('float32')
        self.sample_names = f['sample_names']
        self.node_names = np.array(f['gene_names'])

    @classmethod
    def add_noise(self, dataset, num_added_nodes=10):
        """
        Will add random features and add these nodes as not connected

        Usage:
        pdataset = datasets.PercolateDataset()
        dataset = Graph.add_noise(dataset=pdataset, num_added_nodes=100)
        """

        num_samples = dataset.data.shape[0]
        num_features = dataset.data.shape[1]

        newdata = np.random.random((num_samples, num_features+num_added_nodes))
        newdata = (newdata*2)-1  # normalize; maybe adapt to data?
        newdata[:num_samples, :num_features] = dataset.data  # set to 0 to see it in an image
        dataset.data = newdata

        oldadj = dataset.get_adj()

        newadj = np.zeros((num_features+num_added_nodes, num_features+num_added_nodes))
        newadj[:num_features, :num_features] = oldadj  # set to 0 to see it in an image
        dataset.adj = newadj
        dataset.nb_nodes = dataset.adj.shape[0]
        return dataset

    @classmethod
    def subsample_graph(adj, percentile=100):
        # if we want to sub-sample the edges, based on the edges value
        if percentile < 100:
            # small trick to ignore the 0.
            nan_adj = np.ma.masked_where(adj == 0., adj)
            nan_adj = np.ma.filled(nan_adj, np.nan)

            threshold = np.nanpercentile(nan_adj, 100 - percentile)
            logging.info("We will remove all the adges that has a value smaller than {}".format(threshold))

            to_keep = adj >= threshold  # throw away all the edges that are bigger than what we have.
            return adj * to_keep


class GraphDataset(Dataset):
    def __init__(self, name, graph, opt):

        self.name = name
        self.graph = graph
        self.adj = graph.adj
        self.nb_nodes = graph.nb_nodes
        import pdb; pdb.set_trace()
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


class GraphGeneDataset(GraphDataset):
    """General Dataset to load different graph gene dataset."""

    def __init__(self, data_dir=None, data_file=None, sub_class=None, name=None, graph=None, opt=None):
        """
        Args:
            data_file (string): Path to the h5df file.
        """

        self.sub_class = sub_class
        self.data_dir = data_dir
        self.data_file = data_file

        self.node_names = graph.node_names
        self.sample_names = graph.sample_names

        super(GraphGeneDataset, self).__init__(name=name, graph=graph, opt=opt)

    def load_data(self):
        data_file = os.path.join(self.data_dir, self.data_file)
        self.file = h5py.File(data_file, 'r')
        self.data = np.array(self.file['expression_data'])
        self.nb_nodes = self.data.shape[1]
        self.labels = self.file['labels_data']
        self.sample_names = self.file['sample_names']
        self.nb_class = self.nb_class if self.nb_class is not None else len(self.labels[0])
        self.label_name = self.labels.attrs

        if self.labels.shape != self.labels[:].reshape(-1).shape:
            print "Converting one-hot labels to integers"
            self.labels = np.argmax(self.labels[:], axis=1)

        # Take a number of subclasses
        if self.sub_class is not None:
            self.data = self.data[[i in self.sub_class for i in self.labels]]
            self.labels = self.labels[[i in self.sub_class for i in self.labels]]
            self.nb_class = len(self.sub_class)
            for i, c in enumerate(np.sort(self.sub_class)):
                self.labels[self.labels == c] = i

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = np.expand_dims(sample, axis=-1)
        label = self.labels[idx]
        sample = {'sample': sample, 'labels': label}
        return sample

    def labels_name(self, l):
        if type(self.label_name[str(l)]) == np.ndarray:
            return self.label_name[str(l)][0]
        return self.label_name[str(l)]


class TCGATissue(GraphGeneDataset):

    """TCGA Dataset. We predict tissue."""

    def __init__(self, data_dir='/data/lisa/data/genomics/TCGA/', data_file='TCGA_tissue_ppi.hdf5', **kwargs):
        super(TCGATissue, self).__init__(data_dir=data_dir, data_file=data_file, name='TCGATissue', **kwargs)


class TCGAForLabel(GraphGeneDataset):

    """TCGA Dataset."""

    def __init__(self,
                 data_dir='/data/lisa/data/genomics/TCGA/',
                 data_file='TCGA_tissue_ppi.hdf5',
                 clinical_file="PANCAN_clinicalMatrix.gz",
                 clinical_label="gender",
                 **kwargs):
        super(TCGAForLabel, self).__init__(data_dir=data_dir, data_file=data_file, name='TCGAForLabel', **kwargs)

        dataset = self

        clinical_raw = pd.read_csv(data_dir + clinical_file, compression='gzip', header=0, sep='\t', quotechar='"')
        clinical_raw = clinical_raw.set_index("sampleID")

        logging.info("Possible labels to select from ", clinical_file, " are ", list(clinical_raw.columns))
        logging.info("Selected label is ", clinical_label)

        clinical = clinical_raw[[clinical_label]]
        clinical = clinical.dropna()

        data_raw = pd.DataFrame(dataset.data, index=dataset.sample_names)
        data_raw.index.name = "sampleID"
        samples = pd.DataFrame(np.asarray(dataset.sample_names), index=dataset.sample_names)
        samples.index.name = "sampleID"

        data_joined = data_raw.loc[clinical.index]
        data_joined = data_joined.dropna()
        clinical_joined = clinical.loc[data_joined.index]

        logging.info("clinical_raw", clinical_raw.shape, ", clinical", clinical.shape, ", clinical_joined", clinical_joined.shape)
        logging.info("data_raw", data_raw.shape, "data_joined", data_joined.shape)
        logging.info("Counter for " + clinical_label + ": ")
        logging.info(collections.Counter(clinical_joined[clinical_label].as_matrix()))

        self.labels = pd.get_dummies(clinical_joined).as_matrix().astype(np.float)
        self.data = data_joined.as_matrix()


class BRCACoexpr(GraphGeneDataset):

    """Breast cancer, with coexpression graph. """

    def __init__(self, data_dir='/data/lisa/data/genomics/TCGA/', data_file='BRCA_coexpr.hdf5', nb_class=2, **kwargs):

        # For this dataset, when we chose 2 classes, it's the 'Infiltrating Ductal Carcinoma'
        # and the 'Infiltrating Lobular Carcinoma'

        sub_class = None
        if nb_class == 2:
            nb_class = None
            sub_class = [0, 7]

        super(BRCACoexpr, self).__init__(data_dir=data_dir, data_file=data_file,
                                         nb_class=nb_class, sub_class=sub_class, name='BRCACoexpr',
                                         **kwargs)


class RandomGraphDataset(GraphDataset):

    """
    A random dataset with a random graph for debugging porpucess
    """

    def __init__(self, graph, opt):
        super(RandomGraphDataset, self).__init__(name='RandomGraphDataset', graph=graph, opt=opt)

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


class PercolateDataset(GraphDataset):

    """
    A random dataset where the goal if to find if we can percolate from one side of the graph to the other.
    """

    def __init__(self, graph, opt):
        self.nb_class = 2
        self.size_x = opt.size_perc
        self.size_y = opt.size_perc
        self.num_samples = opt.nb_examples
        self.extra_cn = opt.extra_cn  # uninformative connected layers of nodes
        self.disconnected = opt.disconnected  # number of nodes to disconnect

        super(PercolateDataset, self).__init__(name='PercolateDataset', graph=graph, opt=opt)

    def load_data(self):

        import percolate
        import networkx as nx
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


def split_dataset(dataset, batch_size=100, random=False, train_ratio=0.8, seed=1993, nb_samples=None, nb_per_class=None):
    logger = logging.getLogger()
    all_idx = range(len(dataset))

    if random:
        np.random.seed(seed)
        np.random.shuffle(all_idx)

    if nb_samples is not None:
        all_idx = all_idx[:nb_samples]
        nb_example = len(all_idx)
        logger.info("Going to subsample to {} examples".format(nb_example))

    # If we want to keep a specific number of examples per class in the training set.
    # Since the
    if nb_per_class is not None:

        idx_train = []
        idx_rest = []

        idx_per_class = {i: [] for i in range(dataset.nb_class)}
        nb_finish = np.zeros(dataset.nb_class)

        last = None
        for no, i in enumerate(all_idx):

            last = no
            label = dataset[i]['labels']
            label = np.argmax(label)

            if len(idx_per_class[label]) < nb_per_class:
                idx_per_class[label].append(i)
                idx_train.append(i)
            else:
                nb_finish[label] = 1
                idx_rest.append(i)

            if nb_finish.sum() >= dataset.nb_class:
                break

        idx_rest += all_idx[last+1:]

        idx_valid = idx_rest[:len(idx_rest)/2]
        idx_test = idx_rest[len(idx_rest)/2:]

        logger.info("Keeping {} examples in training set total.".format(len(idx_train)))

    else:
        nb_example = len(all_idx)
        nb_train = int(nb_example * train_ratio)
        nb_rest = (nb_example - nb_train) / 2
        nb_valid = nb_train + int(nb_rest)
        nb_test = nb_train + 2 * int(nb_rest)

        idx_train = all_idx[:nb_train]
        idx_valid = all_idx[nb_train:nb_valid]
        idx_test = all_idx[nb_valid:nb_test]

    train_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(idx_train))
    test_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(idx_test))
    valid_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(idx_valid))
    logger.info("Our sets are of length: train={}, valid={}, tests={}".format(len(idx_train), len(idx_valid), len(idx_test)))
    return train_set, valid_set, test_set


class GBMDataset(GraphGeneDataset):

    " Glioblastoma Multiforme dataset with coexpression graph"

    def __init__(self, data_dir="/data/lisa/data/genomics/TCGA/", data_file="gbm.hdf5", graph=None, opt=None):
        super(GBMDataset, self).__init__(data_dir=data_dir, data_file=data_file, name='GBMDataset', graph=graph, opt=opt)


class NSLRSyntheticDataset(GraphGeneDataset):

    " SynMin dataset with coexpression graph"

    def __init__(self, data_dir="/data/lisa/data/genomics/TCGA/", data_file="syn_nslr.hdf5", nb_class=2, **kwargs):
        super(NSLRSyntheticDataset, self).__init__(data_dir=data_dir, data_file=data_file, nb_class=nb_class, name='NSLRSyntheticDataset', **kwargs)


# TODO: Factorize all the graph manipulation?
def get_dataset(opt):

    """
    Get a dataset based on the options.
    :param opt:
    :return:
    """
    graph = Graph(opt)

    if opt.dataset == 'random':
        logging.info("Getting a random graph")
        dataset = RandomGraphDataset(graph, opt)

    elif opt.dataset == 'tcga-tissue':
        logging.info("Getting TCGA tissue type")
        dataset = TCGATissue(graph, opt)

    elif opt.dataset == 'tcga-brca':
        logging.info("Getting TCGA BRCA type")
        dataset = BRCACoexpr(graph, opt)

    elif opt.dataset == 'percolate':
        dataset = PercolateDataset(graph, opt)

    elif opt.dataset == 'tcga-gbm':
        logging.info("Getting TCGA GBM Dataset")
        dataset = GBMDataset(graph=graph, opt=opt)

    elif opt.dataset == 'nslr-syn':
        logging.info("Getting NSLR Synthetic Dataset")
        dataset = NSLRSyntheticDataset(graph, opt)

    elif opt.dataset == 'percolate-plus':
        logging.info("Getting percolate-plus Dataset")
        pdata = PercolateDataset(graph, opt)
        dataset = Graph.add_noise(dataset=pdata, num_added_nodes=opt.extra_ucn)

    else:
        raise ValueError
    return dataset
