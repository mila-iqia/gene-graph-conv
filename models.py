import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import time


# Create a module for the CGN:
class CGN(nn.Module):

    def __init__(self, nb_nodes, input_dim, channels, adj, out_dim,
                 on_cuda=True, to_dense=False, add_residual=False,
                 ):
        super(CGN, self).__init__()

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda
        self.add_residual = add_residual

        self.adj = adj # The normalization transformation (need to be precomputed)
        self.edges = torch.LongTensor(np.array(np.where(self.adj))) # The list of edges
        flat_adj = self.adj.flatten()[np.where(self.adj.flatten())] # get the value
        flat_adj = torch.FloatTensor(flat_adj)

        # Constructing a sparse matrix
        print "Constructing the sparse matrix..."
        self.sparse_D_norm = torch.sparse.FloatTensor(self.edges, flat_adj, torch.Size([nb_nodes ,nb_nodes]))#.to_dense()
        if to_dense:
            print "Converting the matrix to a dense one, might take some time..."
            self.sparse_D_norm = self.sparse_D_norm.to_dense()
            print "Done!"


        self.register_buffer('sparse_D_norm', self.sparse_D_norm)


        dims = [input_dim] + channels

        print "Constructing the network..."
        layers = []
        for c_in, c_out in zip(dims[:-1], dims[1:]):
            layer = nn.Conv1d(c_in, c_out, 1, bias=True)
            layers.append(layer)
        self.my_layers = nn.ModuleList(layers)


        logistic_layer = []

        if not channels: # Only have one layer
            logistic_in_dim = [nb_nodes * input_dim]
        elif not add_residual:
            logistic_in_dim = [nb_nodes * channels[-1]]
        else:
            print "Adding skip connections..."
            logistic_in_dim = [d * nb_nodes for d in dims]

        for d in logistic_in_dim:
            layer = nn.Linear(d, out_dim)
            logistic_layer.append(layer)

        self.my_logistic_layers = nn.ModuleList(logistic_layer)
        print "Done!"

        # The gradient of the inputs/intermediate variable.
        self.saved_grad = {}

    def _adj_mul(self, x, D):

        nb_examples, nb_channels, nb_nodes = x.size()
        x = x.view(-1, nb_nodes)

        if self.on_cuda:
            x = x.cuda()

        # Needs this hack to work: https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
        x = D.mm(x.t()).t()

        x = x.contiguous().view(nb_examples, nb_channels, nb_nodes)
        return x

    def forward(self, x):

        D_norm = Variable(self.sparse_D_norm, requires_grad=False)

        if self.on_cuda:
            D_norm = D_norm.cuda()

        out = None
        nb_examples, nb_nodes, nb_channels = x.size()

        x = x.permute(0, 2, 1).contiguous()# from ex, node, ch, -> ex, ch, node

        # Do graph convolution for all
        for num, layer in enumerate(self.my_layers):

            #x.register_hook((lambda y: print y))

            if self.add_residual: # skip connection
                if out is None:
                    out = self.my_logistic_layers[num](x.view(nb_examples, -1))
                else:
                    out += self.my_logistic_layers[num](x.view(nb_examples, -1))


            x = self._adj_mul(x, D_norm) # local average
            x = F.tanh(layer(x))  # or relu, sigmoid...



        if out is None:
            out = self.my_logistic_layers[-1](x.view(nb_examples, -1))
        else:
            out += self.my_logistic_layers[-1](x.view(nb_examples, -1))

        out = F.softmax(out)

        return out

# Create a module for MLP
class MLP(nn.Module):
    def __init__(self,input_dim, channels, out_dim=None, on_cuda=True):
        super(MLP, self).__init__()

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda

        dims = [input_dim] + channels

        print "Constructing the network..."
        layers = []
        for c_in, c_out in zip(dims[:-1], dims[1:]):
            layer = nn.Linear(c_in, c_out)
            layers.append(layer)
        self.my_layers = nn.ModuleList(layers)

        if channels:
            self.last_layer = nn.Linear(channels[-1], out_dim)
        else:
            self.last_layer = nn.Linear(input_dim, out_dim)

        print "Done!"

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        representations = []

        x = x.permute(0, 2, 1).contiguous()  # from ex, node, ch, -> ex, ch, node
        for layer in self.my_layers:
            x = F.tanh(layer(x.view(nb_examples, -1)))  # or relu, sigmoid...
            representations.append(x)

        x = self.last_layer(x.view(nb_examples, -1))
        x = F.softmax(x)


        return x

class LCG(nn.Module):
    def __init__(self,input_dim, A, channels=16, out_dim=2, on_cuda=False, num_layers = 1, arg_max = -200):
        super(LCG, self).__init__()

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda
        self.nb_nodes = A.shape[0]
        self.num_layers = num_layers

        self.channels = channels
        #dims = [input_dim] + channels

        #import ipdb; ipdb.set_trace()

        print "Constructing the network..."   
        self.max_edges = sorted((A > 0.).sum(0))[arg_max]
        #self.max_edges = max((A > 0.).sum(0))

        print "Each node will have {} edges.".format(self.max_edges)


        edges_np = [np.asarray(np.where(A[i:i+1] > 0.)).T for i in range(len(A))]
        edges_np = [np.concatenate([x, [[0, i]] * (self.max_edges - len(x))]) if len(x) < self.max_edges
                    else x[:self.max_edges] if len(x) > self.max_edges # Some Nodes has to many connection!
                    else x
                    for i, x in enumerate(edges_np)]

        for i in range(len(edges_np)):
            edges_np[i][:, 0] = i

        edges_np = np.array(edges_np).reshape(-1, 2)
        edges_np = edges_np[:, 0:1]
        print edges_np.shape

        self.edges = torch.LongTensor(edges_np)
        self.super_edges = torch.cat([self.edges] * channels)

        self.my_weights = [nn.Parameter(torch.rand(self.edges.shape[0], channels), requires_grad=True)]
        self.my_bias = [nn.Parameter(torch.zeros(self.nb_nodes, channels), requires_grad=True)]

        self.last_layer = nn.Linear(input_dim * channels, out_dim)
        self.my_layers = nn.ModuleList([self.last_layer])

        #self.edge_selector = [np.where(edges_np[:,0] == i)[0] for i in range(input_dim)]
        self.register_buffer('edges', self.edges)

        print "Done!"

    def GraphConv(self, x, edges, batch_size, weights, bias):
        
        edges = edges.contiguous().view(-1)

        if self.on_cuda:
            edges = edges.cuda()
            weights = weights.cuda()
            bias = bias.cuda()

        #import ipdb; ipdb.set_trace()

        # We have some memory issues when using big
        # convs = []
        # nb = 100000
        # for i in range(edges.size(0)/nb + 1):
        #
        #     if i* nb >= edges.size(0):
        #         break
        #
        #
        #     sub_edges = edges[i*nb:min((i+1)*nb, edges.size(0))]
        #     sub_weight = weights[i*nb/weights.size(-1):min((i+1)*nb, edges.size(0))/weights.size(-1)]
        #
        #     tocompute = torch.index_select(x, 1, sub_edges).view(batch_size, -1, sub_weight.size(-1))#, 2)
        #
        #     print i*nb,(i+1)*nb, weights.size(0)
        #     conv = tocompute*sub_weight
        #     convs.append(conv)
        #
        # import ipdb;
        # ipdb.set_trace()
        # conv = torch.cat(convs, dim=1)

        tocompute = torch.index_select(x, 1, edges).view(batch_size, -1, weights.size(-1))
        conv = tocompute * weights
        conv = conv.view(-1, self.nb_nodes, self.max_edges, weights.size(-1)).sum(2)
        #conv = conv + bias
        return F.tanh(conv)

    def forward(self, x):

        nb_examples, nb_nodes, nb_channels = x.size()
        edges = Variable(self.super_edges, requires_grad=False)

        if self.on_cuda:
            edges = edges.cuda()

        x = self.GraphConv(x, edges.data, nb_examples, self.my_weights[0], self.my_bias[0])
        x = self.last_layer(x.view(nb_examples, -1))
        x = F.softmax(x)

        return x