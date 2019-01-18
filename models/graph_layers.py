import os
import time
import torch
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
    def __init__(self, adj, in_dim=1, channels=1, cuda=False, id_layer=None, mask=None):
        super(GCNLayer, self).__init__()
        self.my_layers = []
        self.cuda = cuda
        self.nb_nodes = adj.shape[0]
        self.in_dim = in_dim
        self.channels = channels
        self.id_layer = id_layer
        self.adj = adj
        self.mask = mask

        self.edges = torch.LongTensor(np.array(self.adj.nonzero()))
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
        shape = x.size()

        # Sparse Max Pooling
        # x = x.view(-1, x.shape[-1])
        # temp = []
        # for i in range(int(adj.shape[0] / 2)):
        #     neighbors = self.centroids[i]
        #     if len(self.adj[neighbors].nonzero()[1]) != 0:
        #         temp.append(x[:, self.adj[neighbors].nonzero()[1]].max(dim=1)[0])
        #     else:
        #         temp.append(x[:, neighbors])
        #
        # max_value = torch.stack(temp)
        # x = max_value.view(shape[0], shape[1], -1).permute(0, 2, 1).contiguous()  # put back in ex, node, channel
        # return x


        # Dense Max Pooling
        start = time.time()
        if self.cuda:
            adj = self.sparse_adj.cuda()
        adj = adj.to_dense()
        temp = []
        x = x.view(-1, x.size(-1))

        for i in range(len(x)):
            temp.append((x[i] * adj).max(dim=1)[0])
        max_value = torch.stack(temp)
        # max_value = (x.view(-1, x.size(-1), 1) * adj[0]).max(dim=1)[0] # old way that will cause memory errors

        print(time.time() - start)
        x = max_value[:, :int(adj.shape[0] / 2)] # Masking
        x = x.view(shape[0], shape[1], -1).permute(0, 2, 1).contiguous()  # put back in ex, node, channel
        return x


def setup_aggregates(adj, nb_layer, cluster_type=None):
    aggregates = [adj]

    # For each layer, build the adjs and the nodes to keep.
    for _ in range(nb_layer):
        adj_hash = joblib.hash(adj) + str(adj.shape)
        processed_path = ".cache/ApprNormalizeLaplacian_{}.npz".format(adj_hash)
        if os.path.isfile(processed_path):
            adj = sparse.load_npz(processed_path)
        else:
            D = np.array(adj.astype(bool).sum(axis=0))[0]
            D_inv = sparse.diags(np.divide(1., np.sqrt(D)), 0)
            adj = D_inv.dot(adj).dot(D_inv)
            sparse.save_npz(processed_path, adj)

        # Do the clustering
        clusters = range(adj.shape[0])
        n_clusters = int(adj.shape[0] / 2)
        if cluster_type == "hierarchy":
            adj_hash = joblib.hash(adj.indices.tostring()) + joblib.hash(sparse.csr_matrix(adj).data.tostring()) + str(n_clusters)
            processed_path = ".cache/" + '{}.npy'.format(adj_hash)
            if os.path.isfile(processed_path):
                clusters = np.load(processed_path)
            else:
                clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                                                     memory='.cache', connectivity=adj,
                                                                     compute_full_tree='auto', linkage='ward').fit_predict(adj.toarray())
                np.save(processed_path, np.array(clusters))
        elif cluster_type == "random":
            adj_hash = joblib.hash(adj.data.tostring()) + joblib.hash(adj.indices.tostring()) + str(n_clusters)
            processed_path = ".cache/random" + '{}.npy'.format(adj_hash)
            if os.path.isfile(processed_path):
                clusters = np.load(processed_path)
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
                np.save(processed_path, np.array(clusters))
        elif cluster_type == "kmeans":
            adj_hash = joblib.hash(adj.data.tostring()) + joblib.hash(adj.indices.tostring()) + str(n_clusters)
            processed_path = ".cache/kmeans" + '{}.npy'.format(adj_hash)
            if os.path.isfile(processed_path):
                clusters = np.load(processed_path)
            else:
                clusters = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto').fit(adj).labels_
            np.save(processed_path, np.array(clusters))

        n_clusters = len(set(clusters))
        cleaned = defaultdict(list)

        for i, cluster in enumerate(clusters):
            cleaned[i] = np.argwhere(clusters == cluster).flatten()

        coo = adj.tocoo()
        for i, col in enumerate(coo.__dict__["col"]):
            coo.__dict__["col"][i] = clusters[col]
        for i, row in enumerate(coo.__dict__["row"]):
            coo.__dict__["row"][i] = clusters[row]

        # indices = torch.LongTensor([coo.row, coo.col])
        # mask = torch.sparse.LongTensor(torch.LongTensor(indices), torch.ones(coo.data.size), coo.shape).coalesce()
        # masks.append(mask)

        adj = coo.tocsr()[:n_clusters, :n_clusters]
        aggregates.append(adj)
    return aggregates
