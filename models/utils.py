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
from torch_scatter import scatter_max, scatter_add, scatter_mul

def max_pool(x, centroids, adj):
    ex, channels, nodes = x.shape
    x = x.view(-1, nodes, 1)

    temp = []
    for i in range(len(x)):
        temp.append((x[i] * adj).max(dim=0)[0])
    res = torch.stack(temp)
    res = scatter_max(src=res, index=centroids, dim=1, fill_value=-1000)[0]
    res = res.view(ex, channels, -1)
    res = res.permute(0, 2, 1).contiguous()
    return res

# We use this to calculate the noramlized laplacian for our graph convolution signal propagation
def norm_laplacian(adj):
    adj.setdiag(np.ones(adj.shape[0]))
    D = np.array(adj.astype(bool).sum(axis=0))[0].astype("float32")
    D_inv = np.divide(1., np.sqrt(D), out=np.zeros_like(D), where=D!=0.)
    D_inv_diag = sparse.diags(D_inv)
    adj = D_inv_diag.dot(adj).dot(D_inv_diag)
    return adj

# We have several methods for clustering the graph. We use them to define the shape of the model and pooling
def hierarchical_clustering(adj, n_clusters):
    adj_hash = joblib.hash(adj.indices.tostring()) + joblib.hash(sparse.csr_matrix(adj).data.tostring()) + str(n_clusters)
    path = ".cache/" + '{}.npy'.format(adj_hash)
    if os.path.isfile(path):
        clusters = np.load(path)
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
        import pdb; pdb.set_trace()
        for gene in range(adj.shape[0]):
            if len(clusters) == n_clusters:
                break
            neighbors = list(adj[gene].nonzero()[1])
            if neighbors:
                clusters.append(np.random.choice(neighbors))
            else:
                clusters.append(gene)
        np.save(path, np.array(clusters))
    return clusters

def kmeans_clustering(adj, n_clusters):
    adj_hash = joblib.hash(adj.data.tostring()) + joblib.hash(adj.indices.tostring()) + str(n_clusters)
    path = ".cache/kmeans" + '{}.npy'.format(adj_hash)

    if os.path.isfile(path):
        clusters = np.load(path)
    else:
        clusters = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto').fit(adj).labels_
        np.save(path, np.array(clusters))
    return clusters


# This function takes in the full adjacency matrix and a number of layers, then returns a bunch of clustered adjacencies
def setup_aggregates(adj, nb_layer, aggregation="hierarchy"):
    aggregates = [adj]
    centroids = []

    for _ in range(nb_layer):
        adj = norm_laplacian(adj)

        if not aggregation:
            aggregates.append(adj)
            centroids.append(np.array(range(adj.shape[0])))
            continue

        n_clusters = int(adj.shape[0] / 2)
        if aggregation == "hierarchy":
            clusters = hierarchical_clustering(adj, n_clusters)
        elif aggregation == "random":
            clusters = random_clustering(adj, n_clusters)
        elif aggregation == "kmeans":
            clusters = kmeans_clustering(adj, n_clusters)

        # When we cluster the adjacency matrix to reduce the graph dimensionality, we do a scatter add to preserve the edge weights.
        # We may want to replace this pytorch-scatter call with a call to the relatively undocumented pytorch _scatter_add function,
        # or change this to a mask (or slice?)
        adj = sparse.csr_matrix((scatter_add(torch.FloatTensor(adj.toarray()), torch.LongTensor(clusters)) > 0.).numpy())
        aggregates.append(adj)
        centroids.append(clusters)
    return aggregates, centroids

def save_computations(self, input, output):
    setattr(self, "input", input)
    setattr(self, "output", output)

def get_every_n(a, n=2):
    for i in range(a.shape[0] // 2):
        yield a[2*i:2*(i+1)]
