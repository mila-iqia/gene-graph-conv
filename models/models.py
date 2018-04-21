import torch
import logging
import numpy as np
from itertools import repeat
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import graphLayer


class EmbeddingLayer(nn.Module):

    def __init__(self, nb_emb, emb_size=32):

        self.emb_size = emb_size
        super(EmbeddingLayer, self).__init__()

        # The embeddings
        self.emb_size = emb_size
        self.emb = nn.Parameter(torch.rand(nb_emb, emb_size))
        self.reset_parameters()

    def forward(self, x):
        emb = x * self.emb
        return emb

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.emb.size(1))        # also, reduce the gene dataset to just the first degree neighbors of the target gene
        #if self.labels.shape != self.labels[:].reshape(-1).shape:
        #    print "Converting one-hot labels to integers"
        #    self.labels = np.argmax(self.labels[:], axis=1)
        self.emb.data.uniform_(-stdv, stdv)


class AttentionLayer(nn.Module):

    def __init__(self, in_dim, nb_attention_head=1):

        self.in_dim = in_dim
        self.nb_attention_head = nb_attention_head
        super(AttentionLayer, self).__init__()

        # The view vector.
        self.attn = nn.Linear(self.in_dim, nb_attention_head)
        self.temperature = 1.

    def forward(self, x):

        nb_examples, nb_nodes, nb_channels = x.size()
        #x = x.permute(0, 2, 1).contiguous()  # from ex, ch, node -> ex, node, ch
        x = x.view(-1, nb_channels)

        # attn_weights = F.softmax(self.attn(x), dim=1)# Should be able to do that,
        # I have some problem with pytorch right now, so I'm doing i manually. Also between you and me, the pytorch example for attention sucks.
        attn_weights = torch.exp(self.attn(x)*self.temperature)
        attn_weights = attn_weights.view(nb_examples, nb_nodes, self.nb_attention_head)
        attn_weights = attn_weights / attn_weights.sum(dim=1).unsqueeze(1)  # normalizing

        x = x.view(nb_examples, nb_nodes, nb_channels)
        attn_applied = x.unsqueeze(-1) * attn_weights.unsqueeze(-2)
        attn_applied = attn_applied.sum(dim=1)
        attn_applied = attn_applied.view(nb_examples, -1)

        return attn_applied, attn_weights


class SoftPoolingLayer(nn.Module):

    def __init__(self, in_dim, nb_attention_head=10):

        self.in_dim = in_dim
        self.nb_attention_head = nb_attention_head
        super(SoftPoolingLayer, self).__init__()

        # The view vector.
        self.attn = nn.Linear(self.in_dim, self.nb_attention_head)
        self.temperature = 1.

    def forward(self, x):

        nb_examples, nb_nodes, nb_channels = x.size()
        #x = x.permute(0, 2, 1).contiguous()  # from ex, ch, node -> ex, node, ch
        x = x.view(-1, nb_channels)

        # attn_weights = F.softmax(self.attn(x), dim=1)# Should be able to do that,
        # I have some problem with pytorch right now, so I'm doing i manually. Also between you and me, the pytorch example for attention sucks.
        attn_weights = torch.exp(self.attn(x)*self.temperature)
        attn_weights = attn_weights.view(nb_examples, nb_nodes, self.nb_attention_head)
        attn_weights = attn_weights / attn_weights.sum(dim=1).unsqueeze(1)  # normalizing
        attn_weights = attn_weights.sum(dim=-1)

        #x = x.view(nb_examples, nb_nodes, nb_channels)
        #attn_applied = x * attn_weights.unsqueeze(-1)
        #attn_applied = attn_applied.view(nb_examples, -1)

        return attn_weights.unsqueeze(-1)


class ElementwiseGateLayer(nn.Module):
    def __init__(self, id_dim):
        self.in_dim = id_dim
        super(ElementwiseGateLayer, self).__init__()
        self.attn = nn.Linear(self.in_dim, 1, bias=True)

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(-1, nb_channels)
        gate_weights = torch.sigmoid(self.attn(x))
        gate_weights = gate_weights.view(nb_examples, nb_nodes, 1)
        return gate_weights


class StaticElementwiseGateLayer(nn.Module):
    def __init__(self, id_dim):
        self.in_dim = id_dim
        super(StaticElementwiseGateLayer, self).__init__()
        self.attn = nn.Parameter(torch.zeros(50), requires_grad=True) + 1.

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        gate_weights = torch.sigmoid(self.attn)
        gate_weights = gate_weights.view(nb_nodes, 1)
        return gate_weights


def save_computations(self, input, output):
    setattr(self, "input", input)
    setattr(self, "output", output)


class SparseLogisticRegression(nn.Module):
    def __init__(self, nb_nodes, input_dim, adj, out_dim, on_cuda=True):
        super(SparseLogisticRegression, self).__init__()

        self.nb_nodes = nb_nodes
        self.input_dim = input_dim
        out_dim = out_dim if out_dim is not None else 2

        np.fill_diagonal(adj, 0.)
        D = adj.sum(0) + 1e-5
        laplacian = np.eye(D.shape[0]) - np.diag((D**-0.5)).dot(adj).dot(np.diag((D**-0.5)))

        self.laplacian = torch.FloatTensor(laplacian)
        self.out_dim = out_dim
        self.on_cuda = on_cuda

        # The logistic layer.
        logistic_in_dim = nb_nodes * input_dim
        logistic_layer = nn.Linear(logistic_in_dim, out_dim)
        logistic_layer.register_forward_hook(save_computations)  # For monitoring


        self.my_logistic_layers = nn.ModuleList([logistic_layer])  # A lsit to be consistant with the other layer.

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(nb_examples, -1)
        x = self.my_logistic_layers[-1](x)
        return x

    def regularization(self, reg_lambda):
        laplacian = Variable(self.laplacian, requires_grad=False)
        if self.on_cuda:
            laplacian = laplacian.cuda()
        weight = self.my_logistic_layers[-1].weight
        reg = torch.abs(weight).mm(laplacian) * torch.abs(weight)
        return reg.sum() * reg_lambda


class LogisticRegression(nn.Module):
    def __init__(self, nb_nodes, input_dim, out_dim, on_cuda=True):
        super(LogisticRegression, self).__init__()

        self.nb_nodes = nb_nodes
        self.input_dim = input_dim
        out_dim = out_dim if out_dim is not None else 2

        self.out_dim = out_dim
        self.on_cuda = on_cuda

        # The logistic layer.
        logistic_in_dim = nb_nodes * input_dim
        logistic_layer = nn.Linear(logistic_in_dim, out_dim)
        logistic_layer.register_forward_hook(save_computations)  # For monitoring
        self.my_logistic_layers = nn.ModuleList([logistic_layer])  # A list to be consistant with the other layer.

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(nb_examples, -1)
        x = self.my_logistic_layers[-1](x)
        return x

    def regularization(self, reg_lambda):
        return 0.0


class GraphNetwork(nn.Module):
    def __init__(self, nb_nodes, input_dim, channels, adj, out_dim,
                 on_cuda=True, add_emb=None, transform_adj=None, aggregate_adj=None, graphLayerType=graphLayer.CGNLayer, use_gate=0.0001, dropout=False, attention_head=0,
                 training_mode=None):
        super(GraphNetwork, self).__init__()

        if transform_adj is None:
            transform_adj = []
        self.my_layers = []
        self.out_dim = out_dim if out_dim is not None else 2
        self.on_cuda = on_cuda
        self.nb_nodes = adj.shape[0]
        self.nb_channels = channels
        self.add_emb = add_emb
        self.graphLayerType = graphLayerType
        self.aggregate_adj = aggregate_adj
        self.dropout = dropout
        self.attention_head = attention_head
        self.training_mode = training_mode

        if add_emb:
            logging.info("Adding node embeddings.")
            self.emb = EmbeddingLayer(nb_nodes, add_emb)
            self.emb.register_forward_hook(save_computations)  # For monitoring
            input_dim = self.emb.emb_size

        # The graph convolutional layers
        convs = []
        dims = [input_dim] + channels
        self.dims = dims
        for i, [c_in, c_out] in enumerate(zip(dims[:-1], dims[1:])):
            # transformation to apply at each layer.
            layer = graphLayerType(adj, c_in, c_out, on_cuda, i, transform_adj=transform_adj, aggregate_adj=aggregate_adj)
            layer.register_forward_hook(save_computations)  # For monitoringv
            convs.append(layer)
            adj = convs[-1].adj

        self.my_convs = nn.ModuleList(convs)

        # The logistic layer
        logistic_layer = []
        if self.attention_head > 0:
            logistic_in_dim = [self.attention_head * dims[-1]]
        else:
            logistic_in_dim = [self.nb_nodes * dims[-1]]

        for d in logistic_in_dim:
            layer = nn.Linear(d, self.out_dim)
            layer.register_forward_hook(save_computations)  # For monitoring
            logistic_layer.append(layer)

        self.my_logistic_layers = nn.ModuleList(logistic_layer)

        # The gating
        self.use_gate = use_gate

        if use_gate > 0.:
            gates = []
            for c_in in dims[1:]:
                gate = ElementwiseGateLayer(c_in)#SoftPoolingLayer(c_in)
                gate.register_forward_hook(save_computations)  # For monitoring
                gates.append(gate)
            self.gates = nn.ModuleList(gates)
        else:
            self.gates = [None] * (len(dims) - 1)

        # Drop out
        self.my_dropouts = [None] * (len(dims) - 1)
        if dropout:
            print "Doing drop-out"
            self.my_dropouts = nn.ModuleList([torch.nn.Dropout(int(dropout)*min((id_layer+1) / 10., 0.4)) for id_layer in range(len(dims)-1)])

        # Attention
        if self.attention_head:
            self.attentionLayer = AttentionLayer(dims[-1], attention_head)
            self.attentionLayer.register_forward_hook(save_computations)  # For monitoringv

        if self.training_mode == 'semi':
            self.last_semi_layer = nn.Conv1d(dims[-1], 1, 1, bias=True)

        if self.training_mode == 'gene-inference':
            self.last_inference_layer = nn.Conv1d(dims[-1], 1, 1, bias=True)

        logging.info("Done!")
        # TODO: add all the funky bells and stuff that the old CGN has.

        # register grads
        self.grads = {}
        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad.data.cpu().numpy()
            return hook
        self.save_grad = save_grad


    def forward(self, x):
    # Depending on the training mode, we will call different functions.
        #import ipdb;
        #ipdb.set_trace()
        nb_examples, nb_nodes, nb_channels = x.size()
        if self.training_mode is None:
            return self.supervised(x)
        elif self.training_mode == 'semi':
            supervised = self.supervised(x[:, :, :nb_channels/2])
            semi = self.semi_supervised(x[:, :, nb_channels/2:])

            return [supervised, semi]

        elif self.training_mode == 'gene-inference':
            return self.gene_inference(x)


    def gene_inference(self, x):

        nb_examples, nb_nodes, nb_channels = x.size()

        if self.add_emb:
            x = self.emb(x)
            x.register_hook(self.save_grad('emb'))

        for i, [layer, dropout] in enumerate(zip(self.my_convs, self.my_dropouts)):

            x = layer(x)
            x = F.relu(x)  # + old_x
            x.register_hook(self.save_grad('layer_{}'.format(i)))


            if dropout is not None:
                id_to_keep = dropout(torch.FloatTensor(np.ones((x.size(0), x.size(1))))).unsqueeze(2)
                if self.on_cuda:
                    id_to_keep = id_to_keep.cuda()

                x = x * id_to_keep

        # Do attention pooling here
        #if self.attention_head:
        #    x, attn = self.attentionLayer(x)

        x = x.permute(0, 2, 1).contiguous()  # from ex, node, ch, -> ex, ch, node
        x = self.last_inference_layer(x)
        x = x.permute(0, 2, 1).contiguous()  # from ex, ch, node -> ex, node, ch

        return x

    def semi_supervised(self, x):

        nb_examples, nb_nodes, nb_channels = x.size()
        if self.add_emb:
            x = self.emb(x)

        for i, layer in enumerate(self.my_convs):
            x = layer(x)
            x = F.relu(x)

        x = x.permute(0, 2, 1).contiguous()  # from ex, node, ch, -> ex, ch, node
        x = self.last_semi_layer(x)
        x = x.permute(0, 2, 1).contiguous()  # from ex, ch, node -> ex, node, ch
        return x

    def supervised(self, x):

        nb_examples, nb_nodes, nb_channels = x.size()

        if self.add_emb:
            x = self.emb(x)
            x.register_hook(self.save_grad('emb'))

        for i, [layer, gate, dropout] in enumerate(zip(self.my_convs, self.gates, self.my_dropouts)):

            if self.use_gate > 0.:
                x = layer(x)
                g = gate(x)
                x = g * x
            else:
                x = layer(x)

            x = F.relu(x)  # + old_x
            x.register_hook(self.save_grad('layer_{}'.format(i)))


            if dropout is not None:
                id_to_keep = dropout(torch.FloatTensor(np.ones((x.size(0), x.size(1))))).unsqueeze(2)
                if self.on_cuda:
                    id_to_keep = id_to_keep.cuda()

                x = x * id_to_keep

        # Do attention pooling here
        if self.attention_head:
            x, attn = self.attentionLayer(x)

        x = self.my_logistic_layers[-1](x.view(nb_examples, -1))
        x.register_hook(self.save_grad('logistic'))
        return x

    def regularization(self, reg_lambda):
        return 0.0

    def get_representation(self):

        # TODO: There is a more systematic way to do that with self.named_children or something, but for that we have
        # to refactor the code first.

        def add_rep(layer, name, rep):
            rep[name] = {'input': layer.input[0].cpu().data.numpy(), 'output': layer.output.cpu().data.numpy()}

        representation = {}

        if self.add_emb:
            add_rep(self.emb, 'emb', representation)

        for i, [layer, gate] in enumerate(zip(self.my_convs, self.gates)):

            if self.use_gate > 0.:
                add_rep(layer, 'layer_{}'.format(i), representation)
                add_rep(gate, 'gate_{}'.format(i), representation)

            else:
                add_rep(layer, 'layer_{}'.format(i), representation)

        add_rep(self.my_logistic_layers[-1], 'logistic', representation)

        if self.attention_head:
            representation['attention'] = {'input': self.attentionLayer.input[0].cpu().data.numpy(),
                         'output': [self.attentionLayer.output[0].cpu().data.numpy(), self.attentionLayer.output[1].cpu().data.numpy()]}

        return representation

    # because of fucking sparse matrices.
    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except AttributeError as e:
                pass # because of fucking sparse matrices.


class CGN(GraphNetwork):
    def __init__(self, **kwargs):
        super(CGN, self).__init__(graphLayerType=graphLayer.CGNLayer, **kwargs)


class SGC(GraphNetwork):
    def __init__(self, **kwargs):
        super(SGC, self).__init__(graphLayerType=graphLayer.SGCLayer, **kwargs)


class LCG(GraphNetwork):
    def __init__(self, **kwargs):
        super(LCG, self).__init__(graphLayerType=graphLayer.LCGLayer, **kwargs)


class MLP(nn.Module):
    def __init__(self, input_dim, channels, out_dim=None, on_cuda=True, dropout=False, training_mode=None):
        super(MLP, self).__init__()
        out_dim = out_dim if out_dim is not None else 2
        input_dim = input_dim if input_dim is not None else 2

        if training_mode == 'gene-inference':
            out_dim = input_dim

        self.my_layers = []
        self.out_dim = out_dim
        self.on_cuda = on_cuda
        self.dropout = dropout
        self.training_mode = training_mode

        dims = [input_dim] + channels

        logging.info("Constructing the network...")
        layers = []
        for c_in, c_out in zip(dims[:-1], dims[1:]):
            layer = nn.Linear(c_in, c_out)
            layers.append(layer)
        self.my_layers = nn.ModuleList(layers)

        if channels:
            self.last_layer = nn.Linear(channels[-1], out_dim)
        else:
            self.last_layer = nn.Linear(input_dim, out_dim)

        self.my_dropout = None
        if dropout:
            print "Doing Drop-out"
            self.my_dropout = torch.nn.Dropout(0.5)

        logging.info("Done!")

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.permute(0, 2, 1).contiguous()  # from ex, node, ch, -> ex, ch, node
        for layer in self.my_layers:
            x = F.relu(layer(x.view(nb_examples, -1)))  # or relu, sigmoid...

            if self.dropout:
                x = self.my_dropout(x)

        x = self.last_layer(x.view(nb_examples, -1))

        return x

    def regularization(self, reg_lambda):
        return 0.0


# class Random(nn.Module):
#     def __init__(self, input_dim, channels, out_dim=None, on_cuda=True):
#         super(Random, self).__init__()
#
#         self.my_layers = []
#         self.out_dim = out_dim
#         self.on_cuda = on_cuda
#
#         dims = [input_dim] + channels
#
#         logging.info("Constructing the network...")
#         layers = []
#         for c_in, c_out in zip(dims[:-1], dims[1:]):
#             layer = nn.Linear(c_in, c_out)
#             layers.append(layer)
#         self.my_layers = nn.ModuleList(layers)
#
#         if channels:
#             self.last_layer = nn.Linear(channels[-1], out_dim)
#         else:
#             self.last_layer = nn.Linear(input_dim, out_dim)
#
#         logging.info("Done!")
#
#     def forward(self, x):
#         nb_examples, nb_nodes, nb_channels = x.size()
#         guesses = np.random.permutation(x[0].data.numpy())
#         x = Variable(torch.FloatTensor(guesses))
#         return x
#
#     def regularization(self, reg_lambda):
#         return 0.0
#

class CNN(nn.Module):
    def __init__(self, input_dim, channels, grid_shape, out_dim=None, on_cuda=True):
        super(CNN, self).__init__()

        self.input_dim = input_dim
        self.channels = channels
        self.out_dim = out_dim
        self.grid_shape = grid_shape
        kernel_size = 2
        stride = 1
        padding = 0

        layers = []
        dims = [input_dim] + channels

        current_size = grid_shape[0]

        for c_in, c_out in zip(dims[:-1], dims[1:]):
            layer = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.ReLU(),
            )

            layers.append(layer)
            new_size = np.ceil((current_size + 2*padding - (kernel_size - 1))/float(stride))
            print current_size, new_size
            current_size = int(new_size)

        self.my_layers = nn.ModuleList(layers)
        out = current_size * current_size * dims[-1]
        self.fc = nn.Linear(out, out_dim)

    def forward(self, x):
        x = x.view(-1, 1, self.grid_shape[0], self.grid_shape[1])

        # The conv.
        out = x
        for layer in self.my_layers:
            out = layer(out)

        # fully connected.
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def regularization(self, reg_lambda):
        return 0.0


def get_model(seed, nb_class, nb_examples, nb_nodes, model, on_cuda, num_channel, num_layer, use_emb, dropout, training_mode, use_gate, nb_attention_head, graph, dataset, model_state=None, opt=None):
    """
    Return a model based on the options.
    :param opt:
    :param dataset:
    :param nb_class:
    :return:
    """

    # TODO: add a bunch of the options
    if model == 'cgn':
        assert graph is not None
        adj_transform, aggregate_function = graphLayer.get_transform(opt, graph.adj)
        my_model = CGN(nb_nodes=dataset.nb_nodes, input_dim=1, channels=[num_channel] * num_layer, adj=graph.adj, out_dim=nb_class,
                       on_cuda=on_cuda, add_emb=use_emb, transform_adj=adj_transform, aggregate_adj=aggregate_function, use_gate=use_gate, dropout=dropout,
                       attention_head=nb_attention_head, training_mode=training_mode)

    elif model == 'lcg':
        assert graph is not None
        adj_transform, aggregate_function = graphLayer.get_transform(opt, graph.adj)
        my_model = LCG(nb_nodes=dataset.nb_nodes, input_dim=1, channels=[num_channel] * num_layer, adj=graph.adj, out_dim=nb_class,
                       on_cuda=on_cuda, add_emb=use_emb, transform_adj=adj_transform, aggregate_adj=aggregate_function, use_gate=use_gate, dropout=dropout,
                       attention_head=nb_attention_head, training_mode=training_mode)

    elif model == 'sgc':
        assert graph is not None
        adj_transform, aggregate_function = graphLayer.get_transform(opt, graph.adj)
        my_model = SGC(nb_nodes=dataset.nb_nodes, input_dim=1, channels=[num_channel] * num_layer, adj=graph.adj, out_dim=nb_class,
                       on_cuda=on_cuda, add_emb=use_emb, transform_adj=adj_transform, aggregate_adj=aggregate_function, use_gate=use_gate, dropout=dropout,
                       attention_head=nb_attention_head, training_mode=training_mode)

    elif model == 'slr':
        assert graph is not None
        my_model = SparseLogisticRegression(nb_nodes=nb_nodes, input_dim=1, adj=graph.adj, out_dim=nb_class, on_cuda=on_cuda)

    elif model == 'lr':
        my_model = LogisticRegression(nb_nodes=nb_nodes, input_dim=1, out_dim=nb_class, on_cuda=on_cuda)

    elif model == 'mlp':
        my_model = MLP(dataset.nb_nodes, [num_channel] * num_layer, nb_class, on_cuda=on_cuda, dropout=dropout, training_mode=training_mode)

    # elif model == 'random':
    #     my_model = Random(dataset.nb_nodes, [num_channel] * num_layer, dataset.nb_class, on_cuda=on_cuda)

    elif model == 'cnn':
        assert opt.dataset == 'percolate'
        assert graph is not None
        # TODO: to change the shape.
        grid_shape = int(np.sqrt(graph.adj.shape[0]))
        grid_shape = [grid_shape, grid_shape]
        my_model = CNN(input_dim=1, channels=[num_channel] * num_layer, grid_shape=grid_shape, out_dim=nb_class, on_cuda=on_cuda)

    else:
        raise ValueError

    if model_state is not None:
        init_state_dict = my_model.state_dict()
        init_state_dict.update(model_state)
        my_model.load_state_dict(init_state_dict)

    return my_model


def setup_l1_loss(my_model, l1_loss_lambda, l1_criterion, on_cuda):
    l1_loss = 0
    if hasattr(my_model, 'my_logistic_layers'):
        l1_loss += calculate_l1_loss(my_model.my_logistic_layers.parameters(), l1_loss_lambda, l1_criterion, on_cuda)
    if hasattr(my_model, 'my_layers') and len(my_model.my_layers) > 0 and type(my_model.my_layers[0]) == torch.nn.modules.linear.Linear:
        l1_loss += calculate_l1_loss(my_model.my_layers[0].parameters(), l1_loss_lambda, l1_criterion, on_cuda)
    if hasattr(my_model, 'last_layer') and type(my_model.last_layer) == torch.nn.modules.linear.Linear:
        l1_loss += calculate_l1_loss(my_model.last_layer.parameters(), l1_loss_lambda, l1_criterion, on_cuda)
    return l1_loss


def calculate_l1_loss(param_generator, l1_loss_lambda, l1_criterion, on_cuda):
    l1_loss = 0
    for param in param_generator:
        if on_cuda:
            l1_target = Variable(torch.FloatTensor(param.size()).zero_()).cuda()
        else:
            l1_target = Variable(torch.FloatTensor(param.size()).zero_())
        l1_loss += l1_criterion(param, l1_target)
    return l1_loss * l1_loss_lambda
