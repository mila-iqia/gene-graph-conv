import os
import time
import logging
import getpass
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from scipy import sparse
import sklearn
import sklearn.cluster
import joblib
from joblib import Memory
import numpy as np
from sklearn.cluster import KMeans


class PoolGraph(object):

    """
    Given x values, a adjacency graph, and a list of value to keep, return the coresponding x.
    """

    def __init__(self, adj, to_keep, please_ignore=False, type='max', cuda=False, **kwargs):

        self.type = type
        self.please_ignore = please_ignore
        self.adj = adj
        self.to_keep = to_keep
        self.cuda = cuda
        self.nb_nodes = self.adj.shape[0]

        logging.info("We are keeping {} elements.".format(to_keep.sum()))
        if to_keep.sum() == adj.shape[0]:
            logging.info("We are keeping all the nodes. ignoring the agregation step.")
            self.please_ignore = True

    def __call__(self, x):
        # x if of the shape (ex, node, channel)
        if self.please_ignore:
            return x

        x = x.permute(0, 2, 1).contiguous()  # put in ex, channel, node
        original_x_shape = x.size()

        if self.cluster_type == 'max':
            temp = []
            for i in range(self.adj.shape[0]):
                if any(self.adj[i].nonzero()):
                    temp.append(x[:, self.adj[i].nonzero()[1]].max(dim=1)[0])
                else:
                    temp.append(x[:, i])
            max_value = torch.stack(temp)

        retn = max_value.view(original_x_shape).permute(0, 2, 1).contiguous()  # put back in ex, node, channel
        return retn


class AggregationGraph(object):

    """
    Master Aggregator. Will return the aggregator function and the adj for each layer of the network.
    """

    def __init__(self, adj, nb_layer, cuda=False, cluster_type=None, **kwargs):

        self.nb_layer = nb_layer
        self.cuda = cuda
        self.cluster_type = cluster_type
        self.adj = adj

        # Build the hierarchy of clusters.
        # Cluster multi-scale everything

        all_to_keep = []  # At each agregation, which node to keep.
        all_aggregate_adjs = []  # At each agregation, which node are connected to whom.

        coo_data=adj.tocoo()
        indices=torch.LongTensor([coo_data.row,coo_data.col])
        mask = torch.sparse.LongTensor(torch.LongTensor(indices), torch.ones(adj.data.size), adj.shape)
        current_adj = adj.copy()

        # For each layer, build the adjs and the nodes to keep.
        for layer_id in range(self.nb_layer):

            # add self loop
            adj_hash = joblib.hash(current_adj) + str(adj.shape)
            processed_path = ".cache/ApprNormalizeLaplacian_{}.npz".format(adj_hash)
            if os.path.isfile(processed_path):
                current_adj = sparse.load_npz(processed_path)
            else:
                D = current_adj.sum(axis=0)
                D_inv = sparse.diags(np.array(1. / np.sqrt(D))[0], 0)
                current_adj = D_inv.dot(adj_sparse).dot(D_inv)
                if not os.path.exists(processed_path):
                    os.makedirs(processed_path)
                sparse.save_npz(processed_path, current_adj)

            # Do the clustering
            ids = range(current_adj.shape[0])
            n_clusters = int(current_adj.shape[0] / (2 ** (layer_id + 1)))
            if self.cluster_type == "hierarchy":
                adj_hash = joblib.hash(current_adj.indices.tostring()) + joblib.hash(sparse.csr_matrix(current_adj).data.tostring()) + str(n_clusters)
                processed_path = ".cache/" + '{}.npy'.format(adj_hash)
                if os.path.isfile(processed_path):
                    ids = np.load(processed_path)
                else:
                    ids = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                                                         memory='.cache', connectivity=(current_adj > 0.).astype(int),
                                                                         compute_full_tree='auto', linkage='ward').fit_predict(adj)
                    np.save(processed_path, np.array(ids))
            elif self.cluster_type == "random":
                adj_hash = joblib.hash(adj.data.tostring()) + joblib.hash(current_adj.indices.tostring()) + joblib.hash(sparse.csr_matrix(current_adj).data.tostring()) + joblib.hash(sparse.csr_matrix(current_adj).indices.tostring()) + str(n_clusters)
                processed_path = ".cache/random" + '{}.npy'.format(adj_hash)
                if os.path.isfile(processed_path):
                    ids = np.load(processed_path)
                else:
                    start = time.time()
                    ids = []
                    for gene in gene_graph.nx_graph.nodes:
                        if len(ids) == n_clusters:
                            break
                        neighbors = list(gene_graph.nx_graph[gene])
                        if neighbors:
                           ids.append(np.random.choice(neighbors))
                        else:
                            ids.append(gene)
                np.save(processed_path, np.array(ids))
            elif self.cluster_type == "kmeans":
                adj_hash = joblib.hash(adj.data.tostring()) + joblib.hash(current_adj.indices.tostring()) + joblib.hash(sparse.csr_matrix(current_adj).data.tostring()) + joblib.hash(sparse.csr_matrix(current_adj).indices.tostring()) + str(n_clusters)
                processed_path = ".cache/kmeans" + '{}.npy'.format(adj_hash)
                if os.path.isfile(processed_path):
                    ids = np.load(processed_path)
                else:
                    ids = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto').fit(adj).labels_
                np.save(processed_path, np.array(ids))

            n_clusters = len(set(ids))
            new_adj = np.zeros((n_clusters, adj.shape[0]))
            for i, cluster in enumerate(ids):
                new_adj[cluster] += adj[i]
            new_adj = sparse.csr_matrix(new_adj)
            all_aggregate_adjs.append(new_adj)
            current_adj = new_adj

        self.to_keeps = all_to_keep
        self.aggregate_adjs = all_aggregate_adjs

        # Build the aggregate function
        self.aggregates = []
        for adj, to_keep in zip(self.aggregate_adjs, self.to_keeps):
            aggregate_adj = PoolGraph(adj=adj, to_keep=to_keep, cuda=cuda, please_ignore=False)
            self.aggregates.append(aggregate_adj)


    def get_aggregate(self, layer_id):
        return self.aggregates[layer_id]


class GraphLayer(nn.Module):
    def __init__(self, adj, in_dim=1, channels=1, cuda=False, id_layer=None, aggregate_adj=None):
        super(GraphLayer, self).__init__()
        self.my_layers = []
        self.cuda = cuda
        self.nb_nodes = adj.shape[0]
        self.in_dim = in_dim
        self.channels = channels
        self.id_layer = id_layer
        self.adj = adj
        self.aggregate_adj = aggregate_adj
        # May want to add self loop here
        if self.aggregate_adj is not None:
            self.aggregate_adj = self.aggregate_adj(id_layer)
        self.init_params()

    def init_params(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    From: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class GCNLayer(GraphLayer):

    def init_params(self):
        self.edges = torch.LongTensor(np.array(self.adj.nonzero()))
        logging.info("Constructing the sparse matrix...")
        data = self.adj.data if type(self.adj) == sparse.csr.csr_matrix else self.adj.flatten()[np.where(self.adj.flatten())]
        sparse_adj = torch.sparse.FloatTensor(self.edges, torch.FloatTensor(data), torch.Size([self.nb_nodes, self.nb_nodes]))  # .to_dense()
        self.register_buffer('sparse_adj', sparse_adj)
        self.linear = nn.Conv1d(in_channels=self.in_dim, out_channels=int(self.channels/2), kernel_size=1, bias=True)  # something to be done with the stride?
        self.eye_linear = nn.Conv1d(in_channels=self.in_dim, out_channels=int(self.channels/2), kernel_size=1, bias=True)

    def _adj_mul(self, x, D):
        nb_examples, nb_channels, nb_nodes = x.size()
        x = x.view(-1, nb_nodes)

        # Needs this hack to work: https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
        #x = D.mm(x.t()).t()
        x = SparseMM(D)(x.t()).t()

        x = x.contiguous().view(nb_examples, nb_channels, nb_nodes)
        return x

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()  # from ex, node, ch, -> ex, ch, node
        adj = Variable(self.sparse_adj, requires_grad=False)

        eye_x = self.eye_linear(x)

        x = self._adj_mul(x, adj)  # + old_x# local average

        x = torch.cat([self.linear(x), eye_x], dim=1)  # + old_x# conv

        x = x.permute(0, 2, 1).contiguous()  # from ex, ch, node -> ex, node, ch

        # We can do max pooling and stuff, if we want.
        if self.aggregate_adj:
            x = self.aggregate_adj(x)

        return x

def get_transform(adj, cuda, add_self=True, norm_adj=True, num_layer=1, pooling="ignore"):

    """
    Return a list of transform that can be applied to the adjacency matrix.
    :param opt: the options
    :return: The list of transform.
    """
    aggregator = AggregationGraph(adj, num_layer, cuda=cuda, cluster_type=pooling)
    get_aggregate = lambda layer_id: aggregator.get_aggregate(layer_id)
    return get_aggregate
