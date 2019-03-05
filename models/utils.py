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
from scipy import sparse


cache_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/.cache/" # this needs to change if we move utils.py

def max_pool(x, centroids, adj):
    from torch_scatter import scatter_max
    ex, channels, nodes = x.shape
    x = x.view(-1, nodes, 1)

    temp = []
    for i in range(len(x)):
        temp.append((x[i] * adj).max(dim=0)[0])
    res = torch.stack(temp)
    res = scatter_max(src=res, index=centroids, dim=1, fill_value=-1000)[0]
    res = res.view(ex, channels, -1)
#    res = res.permute(0, 2, 1).contiguous()
    return res

# We use this to calculate the noramlized laplacian for our graph convolution signal propagation
def norm_laplacian(adj):
    D = np.array(adj.astype(bool).sum(axis=0))[0].astype("float32")
    D_inv = np.divide(1., np.sqrt(D), out=np.zeros_like(D), where=D!=0.)
    D_inv_diag = sparse.diags(D_inv)
    adj = D_inv_diag.dot(adj).dot(D_inv_diag)
    return adj

# We have several methods for clustering the graph. We use them to define the shape of the model and pooling
def hierarchical_clustering(adj, n_clusters, verbose=True):
    adj_hash = joblib.hash(adj.indices.tostring()) + joblib.hash(sparse.csr_matrix(adj).data.tostring()) + str(n_clusters)
    path = cache_dir + "hierarchical" + '{}.npy'.format(adj_hash)
    if os.path.isfile(path):
        if verbose:
            print("Found cache for " + path);
        clusters = np.load(path)
    else:
        if verbose:
            print("No cache for " + path);
        clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                                           memory=cache_dir, connectivity=adj,
                                                           compute_full_tree='auto', linkage='ward').fit_predict(adj.toarray())
    np.save(path, np.array(clusters))
    return clusters

def random_clustering(adj, n_clusters):
    adj_hash = joblib.hash(adj.data.tostring()) + joblib.hash(adj.indices.tostring()) + str(n_clusters)
    path = cache_dir + "random" + '{}.npy'.format(adj_hash)
    if os.path.isfile(path):
        clusters = np.load(path)
    else:
        clusters = []
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
    path = cache_dir + "kmeans" + '{}.npy'.format(adj_hash)

    if os.path.isfile(path):
        clusters = np.load(path)
    else:
        clusters = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto').fit(adj).labels_
        np.save(path, np.array(clusters))
    return clusters


# This function takes in the full adjacency matrix and a number of layers, then returns a bunch of clustered adjacencies
def setup_aggregates(adj, nb_layer, x, aggregation="hierarchy", agg_reduce=2, verbose=True):
    adj.resize((x.shape[1], x.shape[1]))
    adj = (adj > 0.).astype(int)
    adj.setdiag(np.ones(adj.shape[0]))
    adjs = [norm_laplacian(adj)]
    centroids = []
    for _ in range(nb_layer):
        n_clusters = int(adj.shape[0] / agg_reduce) if int(adj.shape[0] / agg_reduce) > 0 else adj.shape[0]
        if verbose:
            print("Reducing graph by a factor of " + str(agg_reduce) + " to " + str(n_clusters) + " nodes")
        if aggregation == "hierarchy":
            clusters = hierarchical_clustering(adj, n_clusters, verbose)
        elif aggregation == "random":
            clusters = random_clustering(adj, n_clusters)
        elif aggregation == "kmeans":
            clusters = kmeans_clustering(adj, n_clusters)
        else:
            clusters = np.array(range(adj.shape[0]))
        _, to_keep = np.unique(clusters, return_index=True)

        adj = torch.zeros(adj.shape).index_add_(1,  torch.LongTensor(clusters), torch.FloatTensor(adj.todense()))
        adj = torch.index_select(adj, 1, torch.LongTensor(to_keep))[:len(to_keep)]
        adj = sparse.csr_matrix(adj > 0)

        adjs.append(norm_laplacian(adj))
        centroids.append(to_keep)
    return adjs, centroids

def save_computations(self, input, output):
    setattr(self, "input", input)
    setattr(self, "output", output)

def get_every_n(a, n=2):
    for i in range(0, a.shape[0], n):
        yield a[i:i+n]
