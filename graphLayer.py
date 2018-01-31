import torch
import logging
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from torchvision import transforms
import networkx

class AgregateGraph(object):

    """
    Given x values, a adjacency graph, and a list of value to keep, return the coresponding x.
    """

    def __init__(self, adj, to_keep, please_ignore=False, type='max', on_cuda=False, **kwargs):

        self.type = type
        self.please_ignore = please_ignore
        self.adj = adj
        self.to_keep = to_keep
        self.on_cuda = on_cuda

        print to_keep

        print "We are keeping {} elements.".format(to_keep.sum())

    def __call__(self, x):

        # x if of the shape (ex, node, channel)
        if self.please_ignore:
            return x

        adj = Variable(self.adj, requires_grad=False)
        to_keep = Variable(torch.FloatTensor(self.to_keep.astype(float)), requires_grad=False)
        if self.on_cuda:
            adj = adj.cuda()
            to_keep = to_keep.cuda()

        x = x.permute(0, 2, 1).contiguous() # put in ex, channel, node
        x_shape = x.size()
        #import ipdb; ipdb.set_trace()

        # For now let's only do the MaxPooling agregate one.
        if self.type == 'max':
            max_value = (x.view(-1, x.size(-1), 1) * adj).max(dim=1)[0]
        elif self.type == 'mean':
            max_value = (x.view(-1, x.size(-1), 1) * adj).mean(dim=1)
        elif self.type == 'strip':
            max_value = x.view(-1, x.size(-1), 1)
        else:
            raise ValueError()

        retn = max_value * to_keep # Zero out The one that we don't care about.
        retn = retn.view(x_shape).permute(0, 2, 1).contiguous() # put back in ex, node, channel
        return retn

def selectNodes(opt, layer_id, adj, seed=1993):

    nb_nodes = adj.shape[0]
    np.random.seed(seed)

    if opt == 'random': # keep a node on
        rand_order = np.arange(nb_nodes)
        np.random.shuffle(rand_order)

        to_keep = np.array([0 if i % (2**(layer_id+1)) == 0 else 1 for i in rand_order])

    if opt == 'grid': # It's a grid. Gonna simulate  stride of 2.

        tmp = int(np.sqrt(nb_nodes))
        order = np.arange(nb_nodes).reshape((tmp, tmp))

        for i in range(order.shape[0]):
            if (i % 2**(layer_id+1)) != 0:
                order[i] = -1

        for i in range(order.shape[1]):
            if (i % 2**(layer_id+1)) != 0:
                order[:, i] = -1

        to_keep = np.array([0 if i in order.flatten() else 1 for i in range(len(order.flatten()))])

    to_keep = (to_keep == 0).astype(float)
    return to_keep



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


    # TODO: add unittests
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
            logging.info("returning a saved transformation.")
            return np.load(self.processed_path)

        logging.info("Doing the approximation...")

        # Fill the diagonal
        np.fill_diagonal(adj, 1.) # TODO: Hummm, thik it's a 0.

        D = adj.sum(axis=1)
        D_inv = np.diag(1. / np.sqrt(D))
        norm_transform = D_inv.dot(adj).dot(D_inv)

        logging.info("Done!")

        # saving the processed approximation
        if self.processed_path:
            logging.info("Saving the approximation in {}".format(self.processed_path))
            np.save(self.processed_path, norm_transform)
            logging.info("Done!")

        return norm_transform


class AugmentGraphConnectivity(object):

    def __init__(self, kernel_size=1, please_ignore=False, **kwargs):

        self.kernel_size = kernel_size
        self.please_ignore = please_ignore

    def __call__(self, adj):

        """
        Augment the connectivity of the nodes in the graph.
        :param adj: The adj matrix
        :param stride: The stride of the pooling. Akin to CNN.
        :param kernel_size: The size of the neibourhood. Same thing as in CNN.
        :param please_ignore: We are not doing pruning, this option is to make things more consistant.
        :return:
        """

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
        # We link all the neighbour of the neighbour (times kernel_size) to our node.
        for i in range(kernel_size):
            current_adj = current_adj.dot(current_adj.T)

        frozen_adj = current_adj.copy()

        new_adj = (frozen_adj > 0).astype(float)

        return new_adj

class GraphLayer(nn.Module):
    def __init__(self, adj, in_dim=1, channels=1, on_cuda=False, id_layer=None, transform_adj=None, agregate_adj=None, striding_method=None):
        super(GraphLayer, self).__init__()
        self.my_layers = []
        self.on_cuda = on_cuda
        self.nb_nodes = adj.shape[0]
        self.transform_adj = transform_adj  # How to transform the adj matrix.
        self.agregate_adj = agregate_adj
        self.in_dim = in_dim
        self.channels = channels
        self.id_layer = id_layer
        self.striding_method = striding_method


        # We can technically do that online, but it's a bit messy and slow, if we need to
        # doa sparse matrix all the time.

        if self.transform_adj:
            logging.info("Transforming the adj matrix")
            adj = transform_adj(adj)
        self.adj = adj

        self.to_keep = np.ones((self.nb_nodes,))

        if self.striding_method is not None:
            self.to_keep = selectNodes(self.striding_method, id_layer, adj)

        if self.agregate_adj:
            self.agregate_adj = transforms.Compose(
                [tr(adj=torch.FloatTensor(self.adj), to_keep=self.to_keep) for tr in agregate_adj])

        self.init_params()

    def init_params(self):
        raise NotImplementedError()


    def forward(self, x):
        raise NotImplementedError()

class CGNLayer(GraphLayer):

    def __init__(self, adj, in_dim=1, channels=1, on_cuda=False, id_layer=None, transform_adj=None, agregate_adj=None, striding_method=None):
        super(CGNLayer, self).__init__(adj, in_dim, channels, on_cuda, id_layer, transform_adj, agregate_adj, striding_method=striding_method)

    def init_params(self):
        self.edges = torch.LongTensor(np.array(np.where(self.adj))) # The list of edges
        flat_adj = self.adj.flatten()[np.where(self.adj.flatten())] # get the value
        flat_adj = torch.FloatTensor(flat_adj)

        # Constructing a sparse matrix
        logging.info("Constructing the sparse matrix...")
        sparse_adj = torch.sparse.FloatTensor(self.edges, flat_adj, torch.Size([self.nb_nodes ,self.nb_nodes]))#.to_dense()
        self.register_buffer('sparse_adj', sparse_adj)
        self.linear = nn.Conv1d(self.in_dim, self.channels, 1, bias=True)

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


class LCGLayer(GraphLayer):
    def __init__(self, adj, in_dim=1, channels=1, on_cuda=False, id_layer=None, transform_adj=None, agregate_adj=None, striding_method=None):
        super(LCGLayer, self).__init__(adj, in_dim, channels, on_cuda, id_layer, transform_adj, agregate_adj, striding_method=striding_method)


    def init_params(self):
        logging.info("Constructing the network...")
        self.max_edges = sorted((self.adj > 0.).sum(0))[-1]

        logging.info("Each node will have {} edges.".format(self.max_edges))

        # Get the list of all the edges. All the first index is 0, we fix that later
        edges_np = [np.asarray(np.where(self.adj[i:i + 1] > 0.)).T for i in range(len(self.adj))]

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
        self.super_edges = torch.cat([self.edges] * self.channels)

        # We have one set of parameters per input dim. might be slow, but for now we will do with that.
        self.my_weights = [nn.Parameter(torch.rand(self.edges.shape[0], self.channels), requires_grad=True) for _ in # TODO: to glorot
                           range(self.in_dim)]
        self.my_weights = nn.ParameterList(self.my_weights)

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
class SGCLayer(GraphLayer):
    def __init__(self, adj, in_dim=1, channels=1, on_cuda=False, id_layer=None, transform_adj=None, agregate_adj=None, striding_method=None):
        super(SGCLayer, self).__init__(adj, in_dim, channels, on_cuda, id_layer, transform_adj, agregate_adj, striding_method=striding_method)

    def init_params(self):
        if self.channels != 1: logging.info("Setting Channels to 1 on SGCLayer, only number of channels supported")
        self.channels = 1  # Other number of channels not suported.

        # dims = [input_dim] + channels

        logging.info("Constructing the eigenvectors...")

        D = np.diag(self.adj.sum(axis=1))
        self.L = D - self.adj
        self.L = torch.FloatTensor(self.L)

        self.g, self.V = torch.eig(self.L, eigenvectors=True)

        # self.V = self.V.half()
        # self.g = self.g.half()

        self.F = nn.Parameter(torch.rand(self.nb_nodes, self.nb_nodes), requires_grad=True)

    def forward(self, x):
        V = self.V
        if self.on_cuda:
            V = self.V.cuda()

        Vx = torch.matmul(torch.transpose(Variable(V), 0, 1), x)
        FVx = torch.matmul(self.F, Vx)
        VFVx = torch.matmul(Variable(V), FVx)
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
    if opt.pool_graph is not None: # graph pruning, etc.
        print "Adding a pooling mecanism in the graph..."
        const_transform += [lambda **kargs: AugmentGraphConnectivity(**kargs)] # Add edges,
        transform += [lambda **kargs: AgregateGraph(on_cuda=opt.cuda, **kargs)] # remove nodes.

    if opt.add_self:
        logging.info("Adding self connection to the graph...")
        const_transform += [lambda **kargs: SelfConnection(opt.add_self, **kargs)] # Add a self connection.

    if opt.norm_adj:
        logging.info("Normalizing the graph...")
        const_transform += [lambda **kargs: ApprNormalizeLaplacian(**kargs)] # Normalize the graph

    return const_transform, transform
