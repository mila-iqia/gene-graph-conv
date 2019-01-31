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

# try:
#     from torch_scatter import scatter_max, scatter_add
#     def max_pool(x, centroids):
#         shape = x.shape
#         x = x.view(x.shape[0] * x.shape[1], -1)
#         x = scatter_max(x, centroids)[0]
#         x = x.view(shape[0], shape[1], -1)  # put back in ex, node, channel
#         return x
# except ImportError:
#     def max_pool(x, centroids):
#         x = x.permute(0, 2, 1).contiguous()  # put in ex, channel, node
#         original_x_shape = x.size()
#         x = x.view(-1, x.shape[-1])
#         #adj = adj.to_dense()
#         temp = []
#         for i in range(adj.shape[0]):
#             neighbors = np.argwhere(centroids==i)
#             if len(neighbors) != 0:
#                 temp.append(x[neighbors].max(dim=1)[0])
#             else:
#                 temp.append(x[i])
#         max_value = torch.stack(temp)
#         max_value.view(original_x_shape).permute(0, 2, 1).contiguous()
#         return max_value

# want inputs of shape ex, node, channels
def max_pool_torch_scatter(x, centroids):
    ex, channels, nodes = x.shape
    x = x.view(ex * channels, -1)
    x = scatter_max(x, centroids)[0]
    x = x.view(ex, channels, -1)
    x = x.permute(0, 2, 1).contiguous()
    return x

def max_pool_dense(x, centroids, adj):
    ex, channels, nodes  = x.shape
    x = x.view(-1, nodes, 1)
    res = (x * adj).max(dim=1)[0]
    res = res.view(ex, channels, nodes)
    res = res.permute(0, 2 ,1).contiguous()
    res = res.narrow(1, 0, len(set(centroids.cpu().numpy()))).contiguous()
    return res

def max_pool_dense_iter(x, centroids, adj):
    ex, channels, nodes = x.shape
    x = x.view(-1, nodes, 1)

    temp = []
    for i in range(len(x)):
        temp.append((x[i] * adj).max(dim=0)[0])
    res = torch.stack(temp)

    res = res.view(ex, channels, nodes)
    res = res.narrow(2, 0, len(set(centroids.cpu().numpy())))
    res = res.permute(0, 2, 1).contiguous()  # put back in ex, node, channel
    return res

# def sparse_max_pool(x, centroids, adj):
#     ex, channels, nodes = x.shape
#     x = x.view(-1, nodes)
#
#     temp = []
#     for i in range(adj.shape[0]):
#         neighbors_to_pool = x[:, adj[i].nonzero().flatten()]
#         maxed = neighbors_to_pool.max(dim=1)[0]
#         temp.append(maxed)
#     res = torch.stack(temp)
#     res = res.narrow(0, 0, len(set(centroids.cpu().numpy())))
#
#     res = res.view(ex, -1, channels)
#     #res = res.permute(0, 2, 1).contiguous()  # put back in ex, node, channel
#     return res

# def sparse_max_pool(x, centroids, adj):
#     ex, channels, nodes = x.shape
#     x = x.view(-1, nodes)
#
#     temp = []
#     for i in range(adj.shape[0]):
#         neighbors_to_pool = x[:, adj[i].nonzero().flatten()]
#         maxed = neighbors_to_pool.max(dim=1)[0]
#         temp.append(maxed)
#     res = torch.stack(temp)
#     res = res.narrow(0, 0, len(set(centroids.cpu().numpy())))
#     res = res.view(ex, channels, nodes)
#     res = res.permute(0, 2, 1).contiguous()  # put back in ex, node, channel

    # res = res.narrow(0, 0, len(set(centroids.cpu().numpy())))
    # res = res.view(ex, -1, channels).contiguous()
    #res = res.permute(0, 2, 1).contiguous()
    return res



from torch_scatter import scatter_max, scatter_add

# try:
#     from torch_scatter import scatter_max, scatter_add
#     def max_pool(x, centroids):
#         shape = x.shape
#         x = x.view(x.shape[0] * x.shape[1], -1)
#         x = scatter_max(x, centroids)[0]
#         x = x.view(shape[0], shape[1], -1)  # put back in ex, node, channel
#         return x
# except ImportError:
def max_pool(x, centroids, adj):
    ex, channels, nodes = x.shape
    x = x.view(-1, nodes, 1)

    temp = []
    for i in range(len(x)):
        temp.append((x[i] * adj).max(dim=0)[0])
    res = torch.stack(temp)

    res = res.view(ex, channels, nodes)
    res = res.narrow(2, 0, len(set(centroids.cpu().numpy())))
    res = res.permute(0, 2, 1).contiguous()  # put back in ex, node, channel
    return res

    # def max_pool(x, centroids):
    #     x = x.permute(0, 2, 1).contiguous()  # put in ex, channel, node
    #     original_x_shape = x.size()
    #     x = x.view(-1, x.shape[-1])
    #     #adj = adj.to_dense()
    #     temp = []
    #     for i in range(adj.shape[0]):
    #         neighbors = np.argwhere(centroids==i)
    #         if len(neighbors) != 0:
    #             temp.append(x[neighbors].max(dim=1)[0])
    #         else:
    #             temp.append(x[i])
    #     max_value = torch.stack(temp)
    #     max_value.view(original_x_shape).permute(0, 2, 1).contiguous()
    #     return max_value

# We use this to calculate the noramlized laplacian for our graph convolution signal propagation
def norm_laplacian(adj):
    D = np.array(adj.astype(bool).sum(axis=0))[0].astype("float32")
    D_inv = np.divide(1., np.sqrt(D), out=np.zeros_like(D), where=D!=0.)
    D_inv_diag = sparse.diags(D_inv)
    try:
        adj = D_inv_diag.dot(adj).dot(D_inv_diag)
    except Exception as e:
        print(e)
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
def setup_aggregates(adj, nb_layer, cluster_type="hierarchy"):
    aggregates = [adj]
    centroids = []

    for _ in range(nb_layer):
        adj = norm_laplacian(adj)

        if not cluster_type:
            aggregates.append(adj)
            centroids.append(np.array(range(adj.shape[0])))
            continue

        # Determine what kind of clustering should we perform on our adjacency matrix?
        n_clusters = int(adj.shape[0] / 2)
        if cluster_type == "hierarchy":
            clusters = hierarchical_clustering(adj, n_clusters)
        elif cluster_type == "random":
            clusters = random_clustering(adj, n_clusters)
        elif cluster_type == "kmeans":
            clusters = kmeans_clustering(adj, n_clusters)

        # When we cluster the adjacency matrix to reduce the graph dimensionality, we do a scatter add to preserve the edge weights.
        # We may want to replace this pytorch-scatter call with a call to the relatively undocumented pytorch _scatter_add function,
        # or change this to a mask (or slice?)
        adj = scatter_add(torch.FloatTensor(adj.toarray()), torch.LongTensor(clusters)).numpy()[:n_clusters]
        adj = sparse.csr_matrix(adj)
        aggregates.append(adj)
        centroids.append(clusters)
    return aggregates, centroids
