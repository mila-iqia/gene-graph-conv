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

    def __init__(self, adj, type='max', cuda=False, **kwargs):

        self.type = type
        self.adj = adj
        self.cuda = cuda
        self.nb_nodes = self.adj.shape[0]


    def __call__(self, x):
        x = x.permute(0, 2, 1).contiguous()  # put in ex, channel, node
        original_x_shape = x.size()
        x = x.view(-1, x.shape[-1])

        if self.type == 'max':
            temp = []
            for i in range(self.adj.shape[0]):
                if len(self.adj[i].nonzero()[1]) != 0:
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
        adj = sparse.csr_matrix(adj)

        # Build the hierarchy of clusters.
        # Cluster multi-scale everything

        aggregates = []  # At each agregation, which node are connected to whom.

        coo_data=adj.tocoo()
        indices=torch.LongTensor([coo_data.row,coo_data.col])
        mask = torch.sparse.LongTensor(torch.LongTensor(indices), torch.ones(adj.data.size), adj.shape)

        # For each layer, build the adjs and the nodes to keep.
        for layer_id in range(self.nb_layer):

            adj_hash = joblib.hash(adj) + str(adj.shape)
            processed_path = ".cache/ApprNormalizeLaplacian_{}.npz".format(adj_hash)
            if os.path.isfile(processed_path):
                adj = sparse.load_npz(processed_path)
            else:
                D = adj.sum(axis=0)
                D_inv = sparse.diags(np.array(np.divide(1., np.sqrt(D)))[0], 0)
                adj = D_inv.dot(adj).dot(D_inv)
                sparse.save_npz(processed_path, adj)

            # Do the clustering
            ids = range(adj.shape[0])
            n_clusters = int(adj.shape[0] / (2 ** (layer_id + 1)))
            if self.cluster_type == "hierarchy":
                adj_hash = joblib.hash(adj.indices.tostring()) + joblib.hash(sparse.csr_matrix(adj).data.tostring()) + str(n_clusters)
                processed_path = ".cache/" + '{}.npy'.format(adj_hash)
                if os.path.isfile(processed_path):
                    ids = np.load(processed_path)
                else:
                    ids = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                                                         memory='.cache', connectivity=adj,
                                                                         compute_full_tree='auto', linkage='ward').fit_predict(adj.toarray())
                    np.save(processed_path, np.array(ids))
            elif self.cluster_type == "random":
                adj_hash = joblib.hash(adj.data.tostring()) + joblib.hash(adj.indices.tostring()) + str(n_clusters)
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
                adj_hash = joblib.hash(adj.data.tostring()) + joblib.hash(adj.indices.tostring()) + str(n_clusters)
                processed_path = ".cache/kmeans" + '{}.npy'.format(adj_hash)
                if os.path.isfile(processed_path):
                    ids = np.load(processed_path)
                else:
                    ids = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto').fit(adj).labels_
                np.save(processed_path, np.array(ids))

            n_clusters = len(set(ids))
            new_adj = np.zeros((adj.shape[0], adj.shape[0]))
            for i, cluster in enumerate(ids):
                new_adj[cluster] += adj[i]
            new_adj = sparse.csr_matrix(new_adj)
            aggregates.append(new_adj)
            adj = new_adj
        self.aggregates = [PoolGraph(adj=adj, cuda=cuda) for adj in aggregates]


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
        sparse_adj = torch.sparse.FloatTensor(self.edges, torch.FloatTensor(self.adj.data), torch.Size([self.nb_nodes, self.nb_nodes]))  # .to_dense()
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
