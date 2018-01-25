import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import h5py
import networkx
import pandas as pd
import collections
from torchvision import transforms, utils


class GraphDataset(Dataset):
    def __init__(self, name,
                 transform=None, use_random_adj=False, percentile=100):

        """
        Args:
            graph_file (string): Path to the h5df file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            transform_adj (callable, optional): Optional transform to be applied on the adjency matrix.
                Will be applied once at the beginning
        """

        self.name = name
        self.transform = transform
        self.use_random_adj = use_random_adj
        self.percentile = percentile

        # Load the data
        self.load_data()
        assert self.adj is not None

        # If we want a random adjancy matrix:
        if use_random_adj:
            print "Overwriting the adjacency matrix to have a random (scale-free) one..."
            dim = range(self.adj.shape[0])
            np.random.shuffle(dim)
            self.adj = self.adj[dim][:, dim]

        # if we want to sub-sample the edges, based on the edges value
        if percentile < 100:
            # small trick to ignore the 0.
            nan_adj = np.ma.masked_where(self.adj == 0., self.adj)
            nan_adj = np.ma.filled(nan_adj, np.nan)

            threshold = np.nanpercentile(nan_adj, 100 - percentile)
            print "We will remove all the adges that has a value smaller than {}".format(threshold)

            to_keep = self.adj >= threshold  # throw away all the edges that are bigger than what we have.
            self.adj = self.adj * to_keep

        self.adj = (self.adj > 0.).astype(float)  # Don't care about the weights, for now.
        self.transform = transform

        # Center
        self.data = self.data - self.data.mean(axis=0) # Ugly, to redo.

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

    def __init__(self, graph_dir=None, graph_file=None, nb_class=None, sub_class=None, **kwargs):
        """
        Args:
            graph_file (string): Path to the h5df file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            transform_adj (callable, optional): Optional transform to be applied on the adjency matrix.
                Will be applied once at the beginning
        """

        self.nb_class = nb_class
        self.sub_class = sub_class
        self.graph_dir = graph_dir
        self.graph_file = graph_file

        super(GraphGeneDataset, self).__init__(**kwargs)

        self.label_name = self.labels.attrs

    def load_data(self):


        graph_file = os.path.join(self.graph_dir, self.graph_file)

        self.file = h5py.File(graph_file, 'r')
        self.data = np.array(self.file['expression_data'])
        self.nb_nodes = self.data.shape[1]
        self.labels = self.file['labels_data']
        self.adj = np.array(self.file['graph_data']).astype('float32')
        self.sample_names = self.file['sample_names']
        self.node_names = np.array(self.file['gene_names'])
        self.nb_class = self.nb_class if self.nb_class is not None else len(self.labels[0])

        # Take a number of subclasses
        if self.sub_class is not None:
            self.data = self.data[sum([(self.labels[:, i] == 1.) for i in self.sub_class]) >= 1]
            self.labels = self.labels[sum([(self.labels[:, i] == 1.) for i in self.sub_class]) >= 1, :]
            self.labels = self.labels[:, self.sub_class]

            # Update the labels, since we only take a subset.
            self.label_name = {str(i): self.label_name[str(k)] for i, k in enumerate(self.sub_class)}
            self.label_name.update({self.label_name[str(k)]: str(k) for k in self.label_name.keys()})
            self.nb_class = len(self.sub_class)

    def __getitem__(self, idx):

        sample = self.data[idx]
        sample = np.expand_dims(sample, axis=-1)
        label = self.labels[idx]
        sample = {'sample': sample, 'labels': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def labels_name(self, l):
        return self.label_name[str(l)]


class TCGATissue(GraphGeneDataset):

    """TCGA Dataset. We predict tissue."""

    def __init__(self, graph_dir='/data/lisa/data/genomics/TCGA/', graph_file='TCGA_tissue_ppi.hdf5', **kwargs):
        super(TCGATissue, self).__init__(graph_dir=graph_dir, graph_file=graph_file, name='TCGATissue', **kwargs)

class TCGAForLabel(GraphGeneDataset):

    """TCGA Dataset."""

    def __init__(self,
                 graph_dir='/data/lisa/data/genomics/TCGA/',
                 graph_file='TCGA_tissue_ppi.hdf5',
                 clinical_file = "PANCAN_clinicalMatrix.gz",
                 clinical_label = "gender",
                 **kwargs):
        super(TCGAForLabel, self).__init__(graph_dir=graph_dir, graph_file=graph_file, name='TCGAForLabel', **kwargs)

        dataset = self

        clinical_raw = pd.read_csv(graph_dir + clinical_file, compression='gzip', header=0, sep='\t', quotechar='"')
        clinical_raw = clinical_raw.set_index("sampleID");


        print "Possible labels to select from ", clinical_file, " are ", list(clinical_raw.columns)
        print "Selected label is ", clinical_label


        clinical = clinical_raw[[clinical_label]]
        clinical = clinical.dropna()

        data_raw = pd.DataFrame(dataset.data, index=dataset.sample_names)
        data_raw.index.name ="sampleID"
        samples = pd.DataFrame(np.asarray(dataset.sample_names), index=dataset.sample_names)
        samples.index.name ="sampleID"

        data_joined = data_raw.loc[clinical.index]
        data_joined = data_joined.dropna()
        clinical_joined = clinical.loc[data_joined.index]


        print "clinical_raw", clinical_raw.shape, ", clinical", clinical.shape, ", clinical_joined", clinical_joined.shape
        print "data_raw", data_raw.shape, "data_joined", data_joined.shape
        print "Counter for " + clinical_label + ": "
        print collections.Counter(clinical_joined[clinical_label].as_matrix())

        self.labels = pd.get_dummies(clinical_joined).as_matrix().astype(np.float)
        self.data = data_joined.as_matrix()



class BRCACoexpr(GraphGeneDataset):

    """Breast cancer, with coexpression graph. """

    def __init__(self, graph_dir='/data/lisa/data/genomics/TCGA/', graph_file='BRCA_coexpr.hdf5', nb_class=None, **kwargs):

        # For this dataset, when we chose 2 classes, it's the 'Infiltrating Ductal Carcinoma'
        # and the 'Infiltrating Lobular Carcinoma'

        sub_class = None
        if nb_class == 2:
            nb_class = None
            sub_class = [0, 7]

        super(BRCACoexpr, self).__init__(graph_dir=graph_dir, graph_file=graph_file,
                                         nb_class=nb_class, sub_class=sub_class, name='BRCACoexpr',
                                         **kwargs)


class RandomGraphDataset(GraphDataset):

    """
    A random dataset with a random graph for debugging porpucess
    """

    def __init__(self, nb_nodes=1000, nb_edges=2000, nb_examples=1000, seed=1993, **kwargs):

        self.nb_nodes = nb_nodes
        self.nb_edges = nb_edges
        self.nb_examples = nb_examples
        self.seed = seed

        super(RandomGraphDataset, self).__init__(name='RandomGraphDataset', **kwargs)

    def load_data(self):

        np.random.seed(self.seed)
        nb_nodes = self.nb_nodes
        nb_examples = self.nb_examples

        # Creating the graph
        # Degree matrix
        self.adj = random_adjacency_matrix(nb_nodes, self.nb_edges, self.use_random_adj)
        self.nb_edges = (self.adj.sum() - nb_nodes) / 2
        self.nb_class = 2
        self.node_names = list(range(nb_nodes))

        # Generating the data
        self.data = np.random.randn(nb_examples, nb_nodes, 1)

        # It's technically a binary classification problem, but to make it more general I treat it as a classification problem
        # JPC Fixed to long vector to work with the updated code
        self.labels = np.zeros((nb_examples))
        self.labels = (np.sum(self.data, axis=1) > 0.)[:, 0].astype(np.long)  # try to predict if the sum. is > than 0.

    def __getitem__(self, idx):

        sample = self.data[idx]
        sample = {'sample': sample, 'labels': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def labels_name(self, l):
        labels = {0: 'neg', 'neg':0, 'pos':1, 1:'pos'}
        return labels[l]


# class RandomGraphDatasetHard(Dataset):
#     """
#     A random dataset with a random graph for debugging
#     """
#     def __init__(self, nb_nodes=10, nb_edges=5, nb_examples=1000,
#                  transform=None, transform_adj_func=None, scale_free=True, seed=1993):
#
#         np.random.seed(seed)
#         self.nb_nodes = nb_nodes
#         self.nb_examples = nb_examples
#
#         # Creating the graph
#         # Degree matrix
#         self.adj = random_adjacency_matrix(nb_nodes, nb_edges, scale_free)
#
#         ## ensure we have relationships between nodes for labels
#
#         print self.adj
#
#
#         self.nb_edges = (self.adj.sum() - nb_nodes) / 2
#         self.nb_class = 2
#
#         # Generating the data
#         self.data = np.random.randn(nb_examples, nb_nodes, 1)
#
#         # It's technically a binary classification problem, but to make it more general I treat it as a classification problem
#         self.labels = np.zeros((nb_examples, 2))
#         self.labels[:, 0] = self.gt_fn(self.data).reshape(-1)
#         self.labels[:, 1] = 1 - self.labels[:, 0]
#
#         # Transform and stuff
#         self.transform = transform
#         self.transform_adj = None
#
#         if transform_adj_func is not None:
#             assert False # Francis was lazy, to refactorize.
#             self.transform_adj = transform_adj_func(self.adj)
#
#
#     def gt_fn(self,x):
#         return (np.max(x[:,[0,1]], axis=1) > ((x[:,2]) + x[:,3]/2.0))*1.0
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def __getitem__(self, idx):
#
#         sample = self.data[idx]
#         sample = {'sample': sample, 'labels': self.labels[idx]}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
#
#     def get_adj(self, transform=True):
#
#         if self.transform_adj is None:
#             return self.adj
#         else:
#             return self.transform_adj
#
#     def labels_name(self, l):
#
#         labels = {0: 'neg', 'neg':0, 'pos':1, 1:'pos'}
#
#         return labels[l]

class PercolateDataset(GraphDataset):

    """
    A random dataset where the goal if to find if we can percolate from one side of the graph to the other.
    """

    def __init__(self, graph_dir='./Dataset/SyntheticData/', graph_file='test.hdf5', **kwargs):

        self.graph_dir = graph_dir
        self.graph_file = graph_file

        super(PercolateDataset, self).__init__(name='PercolateDataset', **kwargs)


    def load_data(self):
        graph_file = os.path.join(self.graph_dir, self.graph_file)

        self.file = h5py.File(graph_file, 'r')
        self.data = np.array(self.file['expression_data'])
        self.nb_nodes = self.data.shape[1]
        self.labels = self.file['labels_data']
        self.adj = np.array(self.file['graph_data']).astype('float32')
        self.sample_names = np.array(range(self.data.shape[0]))
        self.node_names = np.array(range(self.data.shape[1]))

        self.nb_class = 2


    def __getitem__(self, idx):

        sample = self.data[idx]
        sample = np.expand_dims(sample, -1) # Addin a dim for the channels

        sample = {'sample': sample, 'labels': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def labels_name(self, l):
        labels = {0: 'neg', 'neg':0, 'pos':1, 1:'pos'}
        return labels[l]

def split_dataset(dataset, batch_size=100, random=False, train_ratio=0.8, seed=1993, nb_samples=None, nb_per_class=None):

    all_idx = range(len(dataset))

    if random:
        np.random.seed(seed)
        np.random.shuffle(all_idx)

    if nb_samples is not None:
        all_idx = all_idx[:nb_samples]
        nb_example = len(all_idx)
        print "Going to subsample to {} examples".format(nb_example)

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


        print "Keeping {} examples in training set total.".format(len(idx_train))
    else:

        nb_example = len(all_idx)
        nb_train = int(nb_example * train_ratio)
        nb_rest = (nb_example - nb_train) / 2

        idx_train = all_idx[:nb_train]
        idx_valid = all_idx[nb_train:nb_train + nb_rest]
        idx_test = all_idx[nb_train:nb_train + nb_rest]

    train_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(idx_train))
    test_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(idx_test))
    valid_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(idx_valid))

    print "Our sets are of length: train={}, valid={}, tests={}".format(len(idx_train), len(idx_valid), len(idx_test))
    return train_set, valid_set, test_set


class GBMDataset(GraphGeneDataset):
    " Glioblastoma Multiforme dataset with coexpression graph"
    def __init__(self, graph_dir="/data/lisa/data/genomics/TCGA/", graph_file="gbm.hdf5", clinical_file="pathway_commons_adj.csv.gz", **kwargs):
        super(GBMDataset, self).__init__(graph_dir=graph_dir, graph_file=graph_file, name='GBMDataset', **kwargs)
        dataset = self


def random_adjacency_matrix(nb_nodes, approx_nb_edges, scale_free=True):

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
    return A

class PruneGraph(object):

    def __init__(self, deep=None, step=1, nb=None, please_ignore=True):

        self.deep = deep
        self.step = step
        self.nb = nb
        self.please_ignore = please_ignore

    def __call__(self, adj):

        """
        Iterativaly prune a graph until no node are left.
        :param adj: The adj matrix
        :param deep: If we want to cap the pruning
        :param step: If we want to prune fore rapidly
        :param nb: The number of intermediate graph to send.
        :param please_ignore: We are not doing pruning, this option is to make things more consistant.
        :return:
        """

        deep = self.deep
        step = self.step
        nb = self.nb
        please_ignore = self.please_ignore

        # We don't do pruning.
        if please_ignore or nb == 1:
            return [adj] * nb
        else:
            print "Pruning the graph."

        adjs = []

        degree = adj.sum(axis=0)
        nb_edges = sorted(set(degree))

        if deep == None:
            deep = nb

        for i in range(0, step * deep):
            adj_c = adj.copy()

            try:
                to_erase = np.where(degree <= nb_edges[step * i])[0]
                adj_c[to_erase] = 0
                adj_c[:, to_erase] = 0

                if adj_c.sum() != 0:
                    adjs.append(adj_c)
                else:
                    adjs.append(adjs[-1])

            except IndexError: # It's all zero, with just add the last in memory
                adjs.append(adjs[-1])


        if nb is not None:
            adjs = [x for i, x in enumerate(adjs) if i % len(adjs) / (nb-1) == 0]

        assert len(adjs) == (nb - 1)


        return [adj.copy()] + adjs

# TODO: Factorize all the graph manipulation?
def get_dataset(opt):

    """
    Get a dataset based on the options.
    :param opt:
    :return:
    """

    dataset_name = opt.dataset
    nb_examples = opt.nb_examples
    scale_free = opt.scale_free
    nb_class = opt.nb_class
    model = opt.model
    num_layer = opt.num_layer
    seed = opt.seed
    percentile = opt.percentile

    if dataset_name == 'random':

        print "Getting a random graph"
        nb_samples = 10000 if nb_examples is None else nb_examples

        dataset = RandomGraphDataset(nb_nodes=100, nb_edges=100, nb_examples=nb_samples,
                                     use_random_adj=scale_free, seed=seed)
        nb_class = 2 # Right now we only have 2 class

    elif dataset_name == 'tcga-tissue':

        print "Getting TCGA tissue type"
        # To have a feel of TCGA, take a look at 'view_graph_TCGA.ipynb'
        dataset = TCGATissue(nb_class=nb_class, use_random_adj=scale_free, percentile=percentile)

        if nb_class is None: # means we keep all the class (29 I think)
            nb_class = len(dict(dataset.labels.attrs))/2

    elif dataset_name == 'tcga-brca':

        print "Getting TCGA BRCA type"
        # To have a feel of TCGA, take a look at 'view_graph_TCGA.ipynb'
        dataset = BRCACoexpr(nb_class=nb_class, use_random_adj=scale_free, percentile=percentile)

        if nb_class is None: # means we keep all the class (29 I think)
            nb_class = len(dict(dataset.labels.attrs))/2

    elif dataset_name == 'percolate':
        dataset = PercolateDataset(use_random_adj=scale_free)
        nb_class = 2

    elif dataset_name == 'tcga-gbm':
        print "Getting TCGA GBM Dataset"
        nb_class = 2
        dataset = GBMDataset(nb_class=nb_class)

    else:
        raise ValueError

    return dataset, nb_class # TODO: factorize
