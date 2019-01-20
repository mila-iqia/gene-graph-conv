import os
import time
import torch
import glob
import networkx as nx
from torch import nn
from torch.autograd import Variable
from scipy import sparse
from collections import defaultdict
import sklearn
import sklearn.cluster
import joblib
import numpy as np
from sklearn.cluster import KMeans
from torch_scatter import scatter_max, scatter_add



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


class GCNLayer(nn.Module):
    def __init__(self, adj, in_dim=1, channels=1, cuda=False, id_layer=None, centroids=None):
        super(GCNLayer, self).__init__()
        self.my_layers = []
        self.cuda = cuda
        self.nb_nodes = adj.shape[0]
        self.in_dim = in_dim
        self.channels = channels
        self.id_layer = id_layer
        self.adj = adj
        self.centroids = torch.tensor(centroids)

        edges = torch.LongTensor(np.array(self.adj.nonzero()))
        sparse_adj = torch.sparse.FloatTensor(edges, torch.FloatTensor(self.adj.data), torch.Size([self.nb_nodes, self.nb_nodes]))
        self.register_buffer('sparse_adj', sparse_adj)
        self.linear = nn.Conv1d(in_channels=self.in_dim, out_channels=int(self.channels/2), kernel_size=1, bias=True)
        self.eye_linear = nn.Conv1d(in_channels=self.in_dim, out_channels=int(self.channels/2), kernel_size=1, bias=True)
        
        if self.cuda:
            self.sparse_adj = self.sparse_adj.cuda()
            self.centroids = self.centroids.cuda()

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
        shape = x.size()
        
        x = max_pool(x, self.centroids)
        x = x.permute(0, 2, 1).contiguous()
        return x
    
def max_pool(x, centroids):
    shape = x.shape
    x = x.view(x.shape[0] * x.shape[1], -1)
    x = scatter_max(x, centroids)[0]
    x = x.view(shape[0], shape[1], -1)  # put back in ex, node, channel
    return x

def norm_laplacian(adj):
    D = np.array(adj.astype(bool).sum(axis=0))[0].astype("float32")
    D_inv = np.divide(1., np.sqrt(D), out=np.zeros_like(D), where=D!=0.)
    D_inv_diag = sparse.diags(D_inv)
    adj = D_inv_diag.dot(adj).dot(D_inv_diag)
    return adj

def resize(path, n_clusters, adj, clusters):
    if len(clusters) != n_clusters:
        # Find the right cluster file
        for filename in glob.glob(".cache/*" + str(n_clusters) + "*" + ".npy") :
            other_clusters = np.load(filename)
            if len(other_clusters) == n_clusters:
                print(filename)
            import pdb ;pdb.set_trace()
        pass

def hierarchical_clustering(adj, n_clusters):
    adj_hash = joblib.hash(adj.indices.tostring()) + joblib.hash(sparse.csr_matrix(adj).data.tostring()) + str(n_clusters)
    path = ".cache/" + '{}.npy'.format(adj_hash)
    if os.path.isfile(path):
        clusters = np.load(path)
        resize(path, n_clusters, adj, clusters)
    else:
        clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                                           memory='.cache', connectivity=adj,
                                                           compute_full_tree='auto', linkage='ward').fit_predict(adj.toarray())
    np.save(path, np.array(clusters))
    return clusters

def random_clustering(adj, n_clusters):
    adj_hash = joblib.hash(adj.data.tostring()) + joblib.hash(adj.indices.tostring()) + str(n_clusters)
    path = ".cache/random" + '{}.npy'.format(adj_hash)
    if os.path.isfile(path):
        clusters = np.load(path)
    else:
        clusters = []
        for gene in gene_graph.nx_graph.nodes:
            if len(clusters) == n_clusters:
                break
            neighbors = list(gene_graph.nx_graph[gene])
            if neighbors:
                clusters.append(np.random.choice(neighbors))
            else:
                clusters.append(gene)
        np.save(path, np.array(clusters))
    return clusters

def kmeans_clustering(adj, n_clusters):
    adj_hash = joblib.hash(adj.data.tostring()) + joblib.hash(adj.indices.tostring()) + str(n_clusters)
    path = ".cache/kmeans" + '{}.npy'.format(adj_hash)
    resize(path, n_clusters, adj)

    if os.path.isfile(path):
        clusters = np.load(path)
    else:
        clusters = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto').fit(adj).labels_
        np.save(path, np.array(clusters))
    return clusters

def setup_aggregates(adj, nb_layer, cluster_type="hierarchy"):
    aggregates = [adj]
    centroids = []

    # For each layer, build the adjs and the nodes to keep.
    for _ in range(nb_layer):
        adj = norm_laplacian(adj)
        # Do the clustering

        n_clusters = int(adj.shape[0] / 2)
        if cluster_type == "hierarchy":
            clusters = hierarchical_clustering(adj, n_clusters)
        elif cluster_type == "random":
            clusters = random_clustering(adj, n_clusters)
        elif cluster_type == "kmeans":
            clusters = kmeans_clustering(adj, n_clusters)
        else:
            clusters = range(adj.shape[0])

        adj = scatter_add(torch.FloatTensor(adj.toarray()), torch.LongTensor(clusters)).numpy()[:n_clusters]
        adj = sparse.csr_matrix(adj)
        aggregates.append(adj)
        centroids.append(clusters)
    return aggregates, centroids

