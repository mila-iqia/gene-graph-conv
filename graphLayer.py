import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from torchvision import transforms

class AgregateGraph(object):

    """
    Given x values, a adjacency graph, and a list of value to keep, return the coresponding x.
    """

    def __init__(self, adj, to_keep, please_ignore=False, type='max',  **kwargs):

        self.type = type
        self.please_ignore = please_ignore
        self.adj = adj
        self.to_keep = to_keep

        print "We are keeping {} elements.".format(to_keep.sum())

    def __call__(self, x):

        x_shape = x.size()
        if self.please_ignore:
            return x

        adj = Variable(self.adj, requires_grad=False)
        to_keep = Variable(torch.FloatTensor(self.to_keep.astype(float)), requires_grad=False)

        # For now let's only do the MaxPooling agregate one.
        if self.type == 'max':
            max_value = (x.view(-1, x.size(1), 1) * adj).max(dim=-1)[0]
        elif self.type == 'mean':
            max_value = (x.view(-1, x.size(1), 1) * adj).mean(dim=-1)[0]
        elif self.type == 'strip':
            max_value = x.view(-1, x.size(1), 1)
        else:
            raise ValueError()


        retn = max_value * to_keep # Zero out The one that we don't care about.
        return retn.view(x_shape)



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

# TODO: Should have the linerar conv here.
class CGNLayer(nn.Module):

    def __init__(self, adj, in_dim=1, channels=1, on_cuda=False, transform_adj=None, agregate_adj=None):
        super(CGNLayer, self).__init__()

        self.my_layers = []
        self.on_cuda = on_cuda
        self.nb_nodes = adj.shape[0]
        self.transform_adj = transform_adj # How to transform the adj matrix.
        self.agregate_adj = agregate_adj

        # We can technically do that online, but it's a bit messy and slow, if we need to
        # doa sparse matrix all the time.
        if self.transform_adj:
            print "Transforming the adj matrix"
            adj = transform_adj(adj)


        self.adj = adj
        self.to_keep = adj.sum(axis=0) > 0.

        self.edges = torch.LongTensor(np.array(np.where(self.adj))) # The list of edges
        flat_adj = self.adj.flatten()[np.where(self.adj.flatten())] # get the value
        flat_adj = torch.FloatTensor(flat_adj)

        # Constructing a sparse matrix
        print "Constructing the sparse matrix..."
        self.sparse_adj = torch.sparse.FloatTensor(self.edges, flat_adj, torch.Size([self.nb_nodes ,self.nb_nodes]))#.to_dense()
        self.register_buffer('sparse_adj', self.sparse_adj)
        self.linear = nn.Conv1d(in_dim, channels, 1, bias=True)

        if self.agregate_adj:
            self.agregate_adj = transforms.Compose([tr(adj=self.sparse_adj.to_dense(), to_keep=self.to_keep) for tr in agregate_adj])


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

        x = x.permute(0, 2, 1).contiguous()  # from ex, node, ch, -> ex, ch, node

        x = self._adj_mul(x, adj) # local average

        # We can do max pooling and stuff, if we want.

        if self.agregate_adj:
            x = x.permute(0, 2, 1).contiguous()  # from ex, ch, node -> ex, node, ch
            x = self.agregate_adj(x)
            x = x.permute(0, 2, 1).contiguous()  # from ex, node, ch, -> ex, ch, node

        x = self.linear(x) # conv
        x = x.permute(0, 2, 1).contiguous()  # from ex, ch, node -> ex, node, ch

        return x


class LCGLayer(nn.Module):
    def __init__(self, adj, in_dim=1, channels=1, on_cuda=False, arg_max=-1, transform_adj=None, agregate_adj=None):
        super(LCGLayer, self).__init__()

        self.on_cuda = on_cuda
        self.nb_nodes = adj.shape[0]
        self.in_dim = in_dim
        self.nb_channels = channels  # we only support 1 layer for now.
        self.transform_adj = transform_adj
        self.agregate_adj = agregate_adj

        # We can technically do that online, but it's a bit messy and slow, if we need to
        # doa sparse matrix all the time.
        if self.transform_adj:
            print "Transforming the adj matrix"
            adj = transform_adj(adj)


        self.adj = adj
        self.to_keep = adj.sum(axis=0) > 0.

        print "Constructing the network..."
        self.max_edges = sorted((adj > 0.).sum(0))[arg_max]

        print "Each node will have {} edges.".format(self.max_edges)

        # Get the list of all the edges. All the first index is 0, we fix that later
        edges_np = [np.asarray(np.where(adj[i:i + 1] > 0.)).T for i in range(len(adj))]

        # pad the edges, so they all nodes have the same number of edges. help to automate everything.
        edges_np = [np.concatenate([x, [[0, self.nb_nodes]] * (self.max_edges - len(x))]) if len(x) < self.max_edges
                    else x[:self.max_edges] if len(x) > self.max_edges  # Some Nodes have too many connection!
        else x
                    for i, x in enumerate(edges_np)]

        # fix the index that was all 0.
        for i in range(len(edges_np)):
            edges_np[i][:, 0] = i

        edges_np = np.array(edges_np).reshape(-1, 2)
        edges_np = edges_np[:, 1:2]

        self.edges = torch.LongTensor(edges_np)
        self.super_edges = torch.cat([self.edges] * channels)

        # We have one set of parameters per input dim. might be slow, but for now we will do with that.
        self.my_weights = [nn.Parameter(torch.rand(self.edges.shape[0], channels), requires_grad=True) for _ in range(in_dim)]
        self.my_weights = nn.ParameterList(self.my_weights)

        if self.agregate_adj:
            self.agregate_adj = transforms.Compose([tr(adj=torch.FloatTensor(self.adj), to_keep=self.to_keep) for tr in agregate_adj])

        print "Done!"

    def GraphConv(self, x, edges, batch_size, weights):

        edges = edges.contiguous().view(-1)
        useless_node = Variable(torch.zeros(x.size(0), 1, x.size(2)))

        if self.on_cuda:
            edges = edges.cuda()
            weights = weights.cuda()
            useless_node = useless_node.cuda()

        x = torch.cat([x, useless_node], 1)  # add a random filler node
        tocompute = torch.index_select(x, 1, Variable(edges)).view(batch_size, -1, weights.size(-1))

        conv = tocompute * weights
        conv = conv.view(-1, self.nb_nodes, self.max_edges, weights.size(-1)).sum(2)
        return conv

    def forward(self, x):

        nb_examples, nb_nodes, nb_channels = x.size()
        edges = Variable(self.super_edges, requires_grad=False)

        if self.on_cuda:
            edges = edges.cuda()

        #import ipdb; ipdb.set_trace()

        # DO all the input channel and sum them.
        x = sum([self.GraphConv(x[:, :, i].unsqueeze(-1), edges.data, nb_examples, self.my_weights[i]) for i in range(self.in_dim)])

        # We can do max pooling and stuff, if we want.
        if self.agregate_adj:
            x = self.agregate_adj(x)

        return x


# spectral graph conv
class SGCLayer(nn.Module):
    def __init__(self, adj, in_dim=1, channels=1, on_cuda=False, transform_adj=None, agregate_adj=None):
        super(SGCLayer, self).__init__()

        # We can technically do that online, but it's a bit messy and slow, if we need to
        # doa sparse matrix all the time.
        self.transform_adj = transform_adj
        if self.transform_adj:
            print "Transforming the adj matrix"
            adj = transform_adj(adj)


        self.adj = adj
        self.to_keep = adj.sum(axis=0) > 0.

        self.my_layers = []
        self.on_cuda = on_cuda
        self.nb_nodes = adj.shape[0]
        self.agregate_adj = agregate_adj

        self.channels = 1  # channels
        assert channels == 1 # Other number of channels not suported.

        # dims = [input_dim] + channels

        print "Constructing the eigenvectors..."

        D = np.diag(adj.sum(axis=1))
        self.L = D - adj
        self.L = torch.FloatTensor(self.L)

        self.g, self.V = torch.eig(self.L, eigenvectors=True)

        #self.V = self.V.half()
        #self.g = self.g.half()

        print "self.nb_nodes", self.nb_nodes
        self.F = nn.Parameter(torch.rand(self.nb_nodes, self.nb_nodes), requires_grad=True)
        #self.my_bias = nn.Parameter(torch.zeros(self.nb_nodes, channels), requires_grad=True) # To add.

        if self.agregate_adj:
            self.agregate_adj = transforms.Compose([tr(adj=torch.FloatTensor(self.adj), to_keep=self.to_keep) for tr in agregate_adj])

        print "Done!"

    def forward(self, x):

        Vx = torch.matmul(torch.transpose(Variable(self.V), 0, 1), x)
        FVx = torch.matmul(self.F, Vx)
        VFVx = torch.matmul(Variable(self.V), FVx)
        x = VFVx

        # We can do max pooling and stuff, if we want.
        if self.agregate_adj:
            x = self.agregate_adj(x)

        return x

def get_transform(opt):

    """
    Return a list of transform that can be applied to the adjacency matrix.
    :param opt: the options
    :return: The list of transform.
    """

    const_transform = []
    transform = []

    # TODO add some kind of different pruning, like max, average, etc... that will be determine here.
    # Right now the intax is a bit intense, but in the future it will be more parametrizable.
    if opt.prune_graph: # graph pruning, etc.
        print "Pruning the graph..."
        const_transform += [lambda **kargs: PoolGraph(**kargs)]
        transform += [lambda **kargs: AgregateGraph(**kargs)]

    if opt.add_self:
        print "Adding self connection to the graph..."
        const_transform += [lambda **kargs: SelfConnection(opt.add_self, **kargs)] # Add a self connection.

    if opt.norm_adj:
        print "Normalizing the graph..."
        const_transform += [lambda **kargs: ApprNormalizeLaplacian(**kargs)] # Normalize the graph

    return const_transform, transform