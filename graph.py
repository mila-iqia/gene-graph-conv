import logging
import numpy as np
import h5py
import networkx


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
        self.adj = np.array(f['graph_data']).astype('float32')
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
