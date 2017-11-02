import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import time


# Create a module for the CGN:
class CGN(nn.Module):

    def __init__(self, nb_nodes, input_dim, channels, D_norm, out_dim=None, on_cuda=True, to_dense=False
                 ):
        super(CGN, self).__init__()

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda

        self.D_norm = D_norm # The normalization transformation (need to be precomputed)
        self.edges = torch.LongTensor(np.array(np.where(self.D_norm))) # The list of edges
        flat_D_norm = self.D_norm.flatten()[np.where(self.D_norm.flatten())] # get the value
        flat_D_norm = torch.FloatTensor(flat_D_norm)

        # Constructing a sparse matrix
        print "Constructing the sparse matrix..."
        self.sparse_D_norm = torch.sparse.FloatTensor(self.edges, flat_D_norm, torch.Size([nb_nodes ,nb_nodes]))#.to_dense()
        if to_dense:
            print "Converting the matrix to a dense one, might take some time..."
            self.sparse_D_norm = self.sparse_D_norm.to_dense()
            print "Done!"


        self.sparse_D_norm = Variable(self.sparse_D_norm, requires_grad=False)


        dims = [input_dim] + channels

        print "Constructing the network..."
        layers = []
        for c_in, c_out in zip(dims[:-1], dims[1:]):
            layer = nn.Conv1d(c_in, c_out, 1, bias=False)
            layers.append(layer)
        self.my_layers = nn.ModuleList(layers)

        # If we have only one target per graph, we have a linear layer.
        if out_dim is not None:
            if channels:
                self.last_layer = nn.Linear(nb_nodes * channels[-1], out_dim)
            else:
                self.last_layer = nn.Linear(nb_nodes * input_dim, out_dim)

        print "Done!"

    def forward(self, x):

        if self.on_cuda:
            print "putting stuff on gpu..."
            self.sparse_D_norm.cuda()

        nb_examples, nb_nodes, nb_channels = x.size()

        def batch_mul(x, D):
            nb_examples, nb_channels, nb_nodes = x.size()
            x = x.view(-1, nb_nodes)

            # Needs this hack to work: https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
            x = D.mm(x.t()).t()

            x = x.contiguous().view(nb_examples, nb_channels, nb_nodes)
            return x

        x = x.permute(0, 2, 1).contiguous()# from ex, node, ch, -> ex, ch, node

        # Do graph convolution for all
        for layer in self.my_layers:

            # TOTRY: see the big ass-multiplication as a convolution on the example.

            x = batch_mul(x, self.sparse_D_norm)#.cuda()
            x = F.tanh(layer(x))  # or relu, sigmoid...

        if self.out_dim is not None:
            x = self.last_layer(x.view(nb_examples, -1))
            x = F.softmax(x)

        return x