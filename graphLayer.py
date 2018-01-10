import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os

class SelfConnection(object):

    """
    Add (or not) the self connection to the network.
    """

    def __init__(self, add_self_connection, please_ignore, **kwargs):
        self.add_self_connection = add_self_connection
        self.please_ignore = please_ignore

    def __call__(self, adj):

        if self.add_self_connection:
            np.fill_diagonal(adj, 1.)
        else:
            np.fill_diagonal(adj, 0.)

        return adj

class ApprNormalizeLaplacian(object):
    """
    Approximate a normalized Laplacian based on https://arxiv.org/pdf/1609.02907.pdf

    Args:
        processed_path (string): Where to save the processed normalized adjency matrix.
        overwrite (bool): If we want to overwrite the saved processed data.

    """

    def __init__(self, processed_dir='/data/milatmp1/dutilfra/transcriptome/graph/',
                 processed_path=None, unique_id=None, overwrite=False, **kwargs):

        self.processed_dir = processed_dir
        self.processed_path = processed_path
        self.overwrite = overwrite
        self.unique_id = unique_id

    def __call__(self, adj):

        #if self.please_ignore:
        #    return adj

        adj = np.array(adj)
        processed_path = self.processed_path
        if processed_path:
            processed_path = processed_path + '_{}.npy'.format(self.unique_id)

        if processed_path and os.path.exists(processed_path) and not self.overwrite:
            print "returning a saved transformation."
            return np.load(self.processed_path)

        print "Doing the approximation..."
        # Fill the diagonal
        np.fill_diagonal(adj, 1.)

        D = adj.sum(axis=1)
        D_inv = np.diag(1. / np.sqrt(D))
        norm_transform = D_inv.dot(adj).dot(D_inv)

        print "Done!"

        # saving the processed approximation
        if self.processed_path:
            print "Saving the approximation in {}".format(self.processed_path)
            np.save(self.processed_path, norm_transform)
            print "Done!"

        return norm_transform


# TODO right now it's kind of a... average/sum graph? a better restructuration need to be done.
class PoolGraph(object):

    def __init__(self, stride=1, kernel_size=1, please_ignore=True, **kwargs):

        self.stride = stride
        self.kernel_size = kernel_size
        self.please_ignore = please_ignore

    def __call__(self, adj):

        """
        Iterativaly pool a graph. Should be semilar to what usually happen in a image CNN.
        :param adj: The adj matrix
        :param stride: The stride of the pooling. Akin to CNN.
        :param kernel_size: The size of the neibourhood. Same thing as in CNN.
        :param please_ignore: We are not doing pruning, this option is to make things more consistant.
        :return:
        """

        stride = self.stride
        kernel_size = self.kernel_size
        please_ignore = self.please_ignore

        # We don't do pruning.
        if please_ignore:
            return adj
        else:
            print "Pruning the graph."

        # TODO: do it by order of degree, so that we have some garantee
        degrees = adj.sum(axis=0)
        degrees = np.argsort(degrees)[::-1]

        current_adj = adj
        removed_node = []

        # We link all the neighbour of the neighbour (times kernel_size) to our node.
        for i in range(kernel_size):
            current_adj = current_adj.dot(current_adj.T)

        frozen_adj = current_adj.copy()

        # Delete all the unlucky nodes.
        for i, no_node in enumerate(degrees):

            if i % (stride + 1) or no_node in removed_node:
                frozen_adj[no_node] = 0
                removed_node.append(no_node)

        new_adj = (frozen_adj > 0).astype(float)
        return new_adj

class IdentityAgregate(object):

    def __init__(self):
        pass

    def __call__(self, x, adj):
        return x

# TODO: Have the pooling here.
# Add the transform thing here?
class CGNLayer(nn.Module):

    def __init__(self, nb_nodes, adj, on_cuda=True, transform_adj=None, agregate_adj=None):
        super(CGNLayer, self).__init__()

        self.my_layers = []
        self.on_cuda = on_cuda
        self.nb_nodes = nb_nodes
        self.transform_adj = transform_adj # How to transform the adj matrix.
        self.agregate_adj = agregate_adj

        # We can technically do that online, but it's a bit messy and slow, if we need to
        # doa sparse matrix all the time.
        if self.transform_adj:
            print "Transforming the adj matrix"
            adj = transform_adj(adj)


        self.adj = adj
        print adj.sum()
        self.edges = torch.LongTensor(np.array(np.where(self.adj))) # The list of edges
        flat_adj = self.adj.flatten()[np.where(self.adj.flatten())] # get the value
        flat_adj = torch.FloatTensor(flat_adj)

        # Constructing a sparse matrix
        print "Constructing the sparse matrix..."
        self.sparse_adj = torch.sparse.FloatTensor(self.edges, flat_adj, torch.Size([nb_nodes ,nb_nodes]))#.to_dense()
        self.register_buffer('sparse_adj', self.sparse_adj)

    def _adj_mul(self, x, D):

        nb_examples, nb_channels, nb_nodes = x.size()
        x = x.view(-1, nb_nodes)

        # Needs this hack to work: https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
        x = D.mm(x.t()).t()
        x = x.contiguous().view(nb_examples, nb_channels, nb_nodes)
        return x

    def forward(self, x):

        adj = Variable(self.sparse_adj, requires_grad=False)

        if self.on_cuda:
            adj = adj.cuda()

        x = self._adj_mul(x, adj) # local average

        # We can do max pooling and stuff, if we want.
        if self.agregate_adj:
            x = self.agregate_adj(x, adj)

        return x

def get_transform(opt):

    """
    Return a list of transform that can be applied to the adjacency matrix.
    :param opt: the options
    :return: The list of transform.
    """

    transform = []

    # TODO add some kind of different pruning, like max, average, etc... that will be determine here.
    # Right now the intax is a bit intense, but in the future it will be more parametrizable.
    if opt.prune_graph: # graph pruning, etc.
        print "Pruning the graph..."
        transform += [lambda **kargs: PoolGraph(**kargs)]

    if opt.add_self:
        print "Adding self connection to the graph..."
    transform += [lambda **kargs: SelfConnection(opt.add_self, **kargs)] # Add a self connection.

    if opt.norm_adj:
        print "Normalizing the graph..."
        transform += [lambda **kargs: ApprNormalizeLaplacian(**kargs)] # Normalize the graph

    return transform