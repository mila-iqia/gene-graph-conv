import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import time


# Create a module for the CGN:
class CGN(nn.Module):

    def __init__(self, nb_nodes, input_dim, channels, D_norm, out_dim,
                 on_cuda=True, to_dense=False, add_residual=False,
                 ):
        super(CGN, self).__init__()

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda
        self.add_residual = add_residual

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


        #self.sparse_D_norm = Variable(self.sparse_D_norm, requires_grad=False)
        self.register_buffer('sparse_D_norm', self.sparse_D_norm)


        dims = [input_dim] + channels

        print "Constructing the network..."
        layers = []
        for c_in, c_out in zip(dims[:-1], dims[1:]):
            layer = nn.Conv1d(c_in, c_out, 1, bias=False)
            layers.append(layer)
        self.my_layers = nn.ModuleList(layers)


        logistic_layer = []
        logistic_in_dim = []

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

        # If we have only one target per graph, we have a linear layer.
        #if channels:
        #    self.last_layer = nn.Linear(nb_nodes * channels[-1], out_dim)
        #else:
        #    self.last_layer = nn.Linear(nb_nodes * input_dim, out_dim)

        print "Done!"

    def _adj_mul(self, x, D):

        nb_examples, nb_channels, nb_nodes = x.size()
        x = x.view(-1, nb_nodes).cuda()

        # Needs this hack to work: https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
        x = D.mm(x.t()).t()

        x = x.contiguous().view(nb_examples, nb_channels, nb_nodes)
        return x

    def forward(self, x):

        D_norm = Variable(self.sparse_D_norm, requires_grad=False).cuda()
        out = None
        nb_examples, nb_nodes, nb_channels = x.size()

        x = x.permute(0, 2, 1).contiguous()# from ex, node, ch, -> ex, ch, node

        # Do graph convolution for all
        for num, layer in enumerate(self.my_layers):

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

class CGN2(nn.Module):
    def __init__(self,input_dim, edges, channels=1, out_dim=2, on_cuda=True):
        super(CGN2, self).__init__()

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda
        self.edges = edges
        self.channels = channels
        #dims = [input_dim] + channels
        
        print "Constructing the network..."   
        
        
        self.weights1 = nn.Parameter(torch.rand(edges.shape), requires_grad=True)
        print self.weights1.size()

        self.last_layer = nn.Linear(input_dim, out_dim)
        self.my_layers = nn.ModuleList([self.last_layer])

        print "Done!"

        
    #batch_size = 2
    #print x
    def GraphConv(self, x, edges, batch_size, weights):
        
        x = x.clone()
        x = x.view(batch_size, -1)
        
        tocompute = torch.index_select(x, 1, Variable(edges.view(-1))).view(batch_size, -1, 2)
        #print tocompute
        conv = tocompute*weights
        #print conv
        for i, edges_to_select in enumerate(edge_selector):
            #print "x", conv
            #print "e", edges_to_select
            selected_edges = torch.index_select(conv, 1, Variable(torch.LongTensor(edges_to_select)))
            #print "m", selected_edges
            selected_edges = selected_edges.view(-1,edges_to_select.shape[0]*2)
            #print "m", selected_edges
            pooled_edges = torch.max(selected_edges,1)[0]
            #print "mmo",pooled_edges
            x[:,i] = pooled_edges
            #print "xx",x[:,i]
        return x

        
        
    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        
        self.GraphConv(x,self.edges, nb_examples, self.weights1)

        x = self.last_layer(x.view(nb_examples, -1))
        x = F.softmax(x)

        return x