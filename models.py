import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

def agregate_nodes(x, opt='mean'):

    if opt == 'mean':
        return x.mean(dim=2)
    elif opt == 'max':
        return x.max(dim=2)[0]

    return x

class EmbeddingLayer(nn.Module):

    def __init__(self, nb_emb, emb_size=32):

        self.emb_size = emb_size
        super(EmbeddingLayer, self).__init__()

        # The embeddings
        self.emb_size = emb_size
        self.emb = nn.Parameter(torch.rand(nb_emb, emb_size))

    def forward(self, x):
        emb = x * self.emb
        return emb



class AttentionLayer(nn.Module):

    def __init__(self, in_dim):

        self.in_dim = in_dim
        super(AttentionLayer, self).__init__()

        # The view vector.
        self.attn = nn.Linear(self.in_dim, 1)
        self.temperature = 1.

    def forward(self, x):
        nb_examples, nb_channels, nb_nodes = x.size()
        x = x.permute(0, 2, 1).contiguous()  # from ex, ch, node -> ex, node, ch
        x = x.view(-1, nb_channels)

        # attn_weights = F.softmax(self.attn(x), dim=1)# Should be able to do that,
        # I have some problem with pytorch right now, so I'm doing i manually. Also between you and me, the pytorch example for attention sucks.
        attn_weights = torch.exp(self.attn(x)*self.temperature)
        attn_weights = attn_weights.view(nb_examples, nb_nodes, 1)
        attn_weights = attn_weights / attn_weights.sum(dim=1).unsqueeze(-1) # normalizing

        x = x.view(nb_examples, nb_nodes, nb_channels)
        attn_applied = x * attn_weights
        attn_applied = attn_applied.sum(dim=1)
        #print attn_weights[0].max()

        return attn_applied

class CGNLayer(nn.Module):

    def __init__(self, nb_nodes, adj, on_cuda=True):
        super(CGNLayer, self).__init__()

        self.my_layers = []
        self.on_cuda = on_cuda
        self.nb_nodes = nb_nodes

        self.adj = adj # The normalization transformation (need to be precomputed)
        self.edges = torch.LongTensor(np.array(np.where(self.adj))) # The list of edges
        flat_adj = self.adj.flatten()[np.where(self.adj.flatten())] # get the value
        flat_adj = torch.FloatTensor(flat_adj)

        # Constructing a sparse matrix
        print "Constructing the sparse matrix..."
        #print self.edges
        #print flat_adj

        self.sparse_D_norm = torch.sparse.FloatTensor(self.edges, flat_adj, torch.Size([nb_nodes ,nb_nodes]))#.to_dense()
        self.register_buffer('sparse_D_norm', self.sparse_D_norm)

    def _adj_mul(self, x, D):

        nb_examples, nb_channels, nb_nodes = x.size()
        x = x.view(-1, nb_nodes)

        # Needs this hack to work: https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
        x = D.mm(x.t()).t()
        x = x.contiguous().view(nb_examples, nb_channels, nb_nodes)
        return x

    def forward(self, x):

        D_norm = Variable(self.sparse_D_norm, requires_grad=False)

        if self.on_cuda:
            D_norm = D_norm.cuda()

        x = self._adj_mul(x, D_norm) # local average

        return x


# Create a module for the CGN:
class CGN(nn.Module):

    def __init__(self, nb_nodes, input_dim, channels, adj, out_dim,
                 on_cuda=True, add_residual=False, attention_layer=0, add_emb=None
                 ):
        super(CGN, self).__init__()

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda
        self.add_residual = add_residual
        self.nb_nodes = nb_nodes
        self.nb_channels = channels
        self.attention_layer = attention_layer
        self.add_emb = add_emb

        if type(adj) != list:
            adj = [adj] * len(channels)

        self.adj = adj # the list of adjacency matrix

        if add_emb:
            print "Adding node embeddings."
            self.emb = EmbeddingLayer(nb_nodes, add_emb)
            input_dim = self.emb.emb_size

        dims = [input_dim] + channels

        print "Constructing the network..."
        # The normal layer
        layers = []
        for c_in, c_out in zip(dims[:-1], dims[1:]):
            layer = nn.Conv1d(c_in, c_out, 1, bias=True)
            layers.append(layer)
        self.my_layers = nn.ModuleList(layers)

        # The convolutional layer
        convs = []
        for i in range(len(adj)):
            convs.append(CGNLayer(nb_nodes, adj[i], on_cuda))
        self.my_convs = nn.ModuleList(convs)

        # The logistic layer
        logistic_layer = []
        if not channels: # Only have one layer
            logistic_in_dim = [nb_nodes * input_dim]
        elif not add_residual: # Adding a final logistic regression.
            if attention_layer > 0:
                logistic_in_dim = [channels[-1] * attention_layer]  # Changed
            else:
                logistic_in_dim = [nb_nodes * channels[-1]] # Changed here
        else:
            print "Adding skip connections..."
            if attention_layer > 0:
                logistic_in_dim = [d * nb_nodes for d in dims]
            else:
                logistic_in_dim = [d * attention_layer for d in dims]

        for d in logistic_in_dim:
            layer = nn.Linear(d, out_dim)
            logistic_layer.append(layer)

        self.my_logistic_layers = nn.ModuleList(logistic_layer)
        print "Done!"

        if attention_layer > 0:
            print "Adding {} attentions layer.".format(attention_layer)
            self.att = nn.ModuleList([AttentionLayer(channels[-1])] * attention_layer)

    def forward(self, x):

        out = None
        nb_examples, nb_nodes, nb_channels = x.size()
        if self.add_emb:
            x = self.emb(x)

        x = x.permute(0, 2, 1).contiguous()# from ex, node, ch, -> ex, ch, node

        # Do graph convolution for all
        for num, [conv, layer] in enumerate(zip(self.my_convs, self.my_layers)):

            if self.add_residual: # skip connection
                if out is None:
                    out = self.my_logistic_layers[num](x.view(nb_examples, -1))
                else:
                    out += self.my_logistic_layers[num](x.view(nb_examples, -1))

            x = conv(x) # conv
            x = F.relu(layer(x))  # or relu, sigmoid...

        # agregate the node
        if self.attention_layer > 0:
            x = torch.stack([att(x) for att in self.att], dim=-1)

        if out is None:
            out = self.my_logistic_layers[-1](x.view(nb_examples, -1))
        else:
            out += self.my_logistic_layers[-1](x.view(nb_examples, -1))

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

        x = x.permute(0, 2, 1).contiguous()  # from ex, node, ch, -> ex, ch, node
        for layer in self.my_layers:
            x = F.relu(layer(x.view(nb_examples, -1)))  # or relu, sigmoid...

        x = self.last_layer(x.view(nb_examples, -1))

        return x

class LCG(nn.Module):
    def __init__(self,input_dim, A, channels=16, out_dim=2, on_cuda=False, num_layers = 1, arg_max = -1):
        super(LCG, self).__init__()

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda
        self.nb_nodes = A.shape[0]
        self.num_layers = num_layers

        self.nb_channels = [channels] # we only support 1 layer for now.

        print "Constructing the network..."   
        self.max_edges = sorted((A > 0.).sum(0))[arg_max]

        print "Each node will have {} edges.".format(self.max_edges)


        # Get the list of all the edges. All the first index is 0, we fix that later
        edges_np = [np.asarray(np.where(A[i:i+1] > 0.)).T for i in range(len(A))]


        # pad the edges, so they all nodes have the same number of edges. help to automate everything.
        edges_np = [np.concatenate([x, [[0, self.nb_nodes]] * (self.max_edges - len(x))]) if len(x) < self.max_edges
                    else x[:self.max_edges] if len(x) > self.max_edges # Some Nodes have to many connection!
                    else x
                    for i, x in enumerate(edges_np)]

        # fix the index that was all 0.
        for i in range(len(edges_np)):
            edges_np[i][:, 0] = i


        edges_np = np.array(edges_np).reshape(-1, 2)
        edges_np = edges_np[:, 1:2]

        self.edges = torch.LongTensor(edges_np)
        self.super_edges = torch.cat([self.edges] * channels)

        # we add a weight the fake node (that we introduced in the padding)
        my_weights = [nn.Parameter(torch.rand(self.edges.shape[0], channels), requires_grad=True) for _ in range(num_layers)]
        self.my_weights = nn.ParameterList(my_weights)

        last_layer = nn.Linear(input_dim * channels, out_dim)
        self.my_logistic_layers = nn.ModuleList([last_layer])

        self.register_buffer('edges', self.edges)

        print "Done!"

    def GraphConv(self, x, edges, batch_size, weights):
        
        edges = edges.contiguous().view(-1)
        useless_node = Variable(torch.zeros(x.size(0), 1, x.size(2)))

        if self.on_cuda:
            edges = edges.cuda()
            weights = weights.cuda()
            useless_node = useless_node.cuda()

        x = torch.cat([x, useless_node], 1) # add a random filler node
        tocompute = torch.index_select(x, 1, Variable(edges)).view(batch_size, -1, weights.size(-1))

        conv = tocompute * weights
        conv = conv.view(-1, self.nb_nodes, self.max_edges, weights.size(-1)).sum(2)
        return F.relu(conv)

    def forward(self, x):

        nb_examples, nb_nodes, nb_channels = x.size()
        edges = Variable(self.super_edges, requires_grad=False)

        if self.on_cuda:
            edges = edges.cuda()

        for i in range(self.num_layers):
            x = self.GraphConv(x, edges.data, nb_examples, self.my_weights[i])

        x = self.my_logistic_layers[-1](x.view(nb_examples, -1))
        #x = F.softmax(x)

        return x

def get_model(opt, dataset, nb_class):

    """
    Return a model based on the options.
    :param opt:
    :param dataset:
    :param nb_class:
    :return:
    """

    model = opt.model
    num_channel = opt.num_channel
    num_layer = opt.num_layer
    on_cuda = opt.cuda
    skip_connections = opt.skip_connections

    if model == 'cgn':
        # To have a feel of the model, please take a look at cgn.ipynb
        my_model = CGN(dataset.nb_nodes, 1, [num_channel] * num_layer, dataset.get_adj(), nb_class,
                       on_cuda=on_cuda, add_residual=skip_connections, attention_layer=opt.attention_layer, add_emb=opt.use_emb)

    elif model == 'mlp':
        my_model = MLP(dataset.nb_nodes, [num_channel] * num_layer, nb_class, on_cuda=on_cuda) # TODO: add a bunch of the options

    elif model == 'lcg':
        my_model = LCG(dataset.nb_nodes, dataset.get_adj(), out_dim=nb_class,
                              on_cuda=on_cuda, channels=num_channel, num_layers=num_layer)# TODO: add a bunch of the options
    else:
        raise ValueError

    return my_model

#spectral graph conv
class SGC(nn.Module):
    def __init__(self,input_dim, A, channels=16, out_dim=2, on_cuda=False, num_layers = 1, arg_max = -200):
        super(SGC, self).__init__()

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda
        self.nb_nodes = A.shape[0]
        self.num_layers = num_layers

        self.channels = channels
        #dims = [input_dim] + channels

        #import ipdb; ipdb.set_trace()

        print "Constructing the eigenvectors..."   
        
        D = np.diag(A.sum(axis=1))
        self.L = D-A
        self.L = torch.FloatTensor(self.L)
        if self.on_cuda:
            self.L = self.L.cuda()
        self.g, self.V = torch.eig(self.L, eigenvectors=True)
        
        
        self.F = [nn.Parameter(torch.rand(self.nb_nodes, self.nb_nodes), requires_grad=True)]
        self.my_bias = [nn.Parameter(torch.zeros(self.nb_nodes, channels), requires_grad=True)]

        last_layer = nn.Linear(input_dim * channels, out_dim)
        self.my_logistic_layers = nn.ModuleList([last_layer])


        print "Done!"

    def forward(self, x):

        nb_examples, nb_nodes, nb_channels = x.size()

        #First draft. There needs to be a nonlinearity here. 
        #The output shape is wrong because we should get a vector back.
        x = self.V*self.F*torch.transpose(self.V, 0,1)*x
            
        x = self.my_logistic_layers[-1](x.view(nb_examples, -1))
        x = F.softmax(x)

        return x
