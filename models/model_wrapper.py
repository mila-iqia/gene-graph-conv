"""A SKLearn-style wrapper around our PyTorch models (like Graph Convolutional Network and SparseLogisticRegression) implemented in models.py"""

import logging
import sklearn
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.neural_network
import sklearn.tree
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from models.graph_layers import get_transform, GCNLayer, SGCLayer, LCGLayer

# For Monitoring
def save_computations(self, input, output):
    setattr(self, "input", input)
    setattr(self, "output", output)

class Method:
    def __init__(self):
        pass

class Model(nn.Module):

    def __init__(self, name=None, column_names=None, num_epochs=100, channels=16, num_layer=2, embedding=8, gating=0.0001, dropout=False, cuda=False, seed=0, adj=None, graph_name=None, pooling="ignore", prepool_extralayers=0):
        self.name = name
        self.column_names = column_names
        self.channels = channels
        self.num_layer = num_layer
        self.embedding = embedding
        self.gating = gating
        self.dropout = dropout
        self.on_cuda = cuda
        self.num_epochs = num_epochs
        self.seed = seed
        self.adj = adj
        self.graph_name = graph_name
        self.prepool_extralayers = prepool_extralayers
        self.pooling = pooling
        self.batch_size = 10
        self.start_patience = 10
        self.attention_head = 0
        self.train_valid_split = 0.8
        self.best_model = None
        super(Model, self).__init__()

    def fit(self, X, y, adj=None):
        self.adj = adj
        self.X = X
        self.setup_layers()

        x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, stratify=y, train_size=self.train_valid_split, test_size=1-self.train_valid_split, random_state=self.seed)

        # pylint: disable=E1101
        x_train = torch.FloatTensor(np.expand_dims(x_train, axis=2))
        x_valid = torch.FloatTensor(np.expand_dims(x_valid, axis=2))
        y_train = torch.FloatTensor(y_train.values)
        # pylint: enable=E1101

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        if self.on_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            self.cuda()

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)
        max_valid = 0
        patience = self.start_patience
        for epoch in range(0, self.num_epochs):
            for i in range(0, x_train.shape[0], self.batch_size):
                inputs, labels = x_train[i:i + self.batch_size], y_train[i:i + self.batch_size]

                inputs = Variable(inputs, requires_grad=False).float()
                if self.on_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                self.train()
                y_pred = self(inputs)

                targets = Variable(labels, requires_grad=False).long()
                loss = criterion(y_pred, targets)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.eval()

            auc = {'train': 0., 'valid': 0.}
            res = []
            for i in range(0, x_train.shape[0], self.batch_size):
                inputs = Variable(x_train[i:i + self.batch_size]).float()
                if self.on_cuda:
                    inputs = inputs.cuda()
                res.append(self(inputs)[:, 1].data.cpu().numpy())
            y_hat = np.concatenate(res).ravel()
            auc['train'] = sklearn.metrics.roc_auc_score(y_train.numpy(), y_hat)

            res = []
            for i in range(0, x_valid.shape[0], self.batch_size):
                inputs = Variable(x_valid[i:i + self.batch_size]).float()
                if self.on_cuda:
                    inputs = inputs.cuda()
                res.append(self(inputs)[:, 1].data.cpu().numpy())
            y_hat = np.concatenate(res).ravel()
            auc['valid'] = sklearn.metrics.roc_auc_score(y_valid, y_hat)

            patience = patience - 1
            if patience == 0:
                break
            if (max_valid <= auc['valid']) and epoch > 5:
                max_valid = auc['valid']
                patience = self.start_patience
                self.best_model = self.state_dict().copy()
        self.load_state_dict(self.best_model)

    def predict(self, inputs):
        """ Run the trained model on the inputs"""
        return self.forward(inputs)


class MLP(Model):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

    def setup_layers(self):
        out_dim = 2
        channels = [self.channels] * self.num_layer
        input_dim = len(self.X.keys())
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

        self.my_dropout = torch.nn.Dropout(0.5) if self.dropout else None
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

class SLR(Model):
    def __init__(self, **kwargs):
        super(SLR, self).__init__(**kwargs)

    def setup_layers(self):
        self.nb_nodes = len(self.X.keys())
        self.input_dim = 1
        self.out_dim = 2

        np.fill_diagonal(adj, 0.)
        D = adj.sum(0) + 1e-5
        laplacian = np.eye(D.shape[0]) - np.diag((D**-0.5)).dot(adj).dot(np.diag((D**-0.5)))
        self.laplacian = torch.FloatTensor(laplacian)

        # The logistic layer.
        logistic_in_dim = self.nb_nodes * self.input_dim
        logistic_layer = nn.Linear(logistic_in_dim, self.out_dim)
        logistic_layer.register_forward_hook(save_computations)
        self.my_logistic_layers = nn.ModuleList([logistic_layer])

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

class LR(Model):
    def __init__(self, **kwargs):
        super(LR, self).__init__(**kwargs)

    def setup_layers(self):
        self.nb_nodes = len(self.X.keys())
        self.input_dim = 1
        self.out_dim = 2

        # The logistic layer.
        logistic_in_dim = nb_nodes * input_dim
        logistic_layer = nn.Linear(logistic_in_dim, out_dim)
        logistic_layer.register_forward_hook(save_computations)
        self.my_logistic_layers = nn.ModuleList([logistic_layer])

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(nb_examples, -1)
        x = self.my_logistic_layers[-1](x)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, nb_emb, emb_size=32):
        self.emb_size = emb_size
        super(EmbeddingLayer, self).__init__()
        self.emb_size = emb_size
        self.emb = nn.Parameter(torch.rand(nb_emb, emb_size))
        self.reset_parameters()

    def forward(self, x):
        emb = x * self.emb
        return emb

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.emb.size(1))
        self.emb.data.uniform_(-stdv, stdv)


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, nb_attention_head=1):
        self.in_dim = in_dim
        self.nb_attention_head = nb_attention_head
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(self.in_dim, nb_attention_head)
        self.temperature = 1.

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(-1, nb_channels)

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
        self.attn = nn.Linear(self.in_dim, self.nb_attention_head)
        self.temperature = 1.

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(-1, nb_channels)

        attn_weights = torch.exp(self.attn(x)*self.temperature)
        attn_weights = attn_weights.view(nb_examples, nb_nodes, self.nb_attention_head)
        attn_weights = attn_weights / attn_weights.sum(dim=1).unsqueeze(1)  # normalizing
        attn_weights = attn_weights.sum(dim=-1)

        return attn_weights.unsqueeze(-1)


class ElementwiseGateLayer(nn.Module):
    def __init__(self, in_dim):
        self.in_dim = in_dim
        super(ElementwiseGateLayer, self).__init__()
        self.attn = nn.Linear(self.in_dim, 1, bias=True)

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(-1, nb_channels)
        gate_weights = torch.sigmoid(self.attn(x))
        gate_weights = gate_weights.view(nb_examples, nb_nodes, 1)
        return gate_weights


class StaticElementwiseGateLayer(nn.Module):
    def __init__(self, in_dim):
        self.in_dim = in_dim
        super(StaticElementwiseGateLayer, self).__init__()
        self.attn = nn.Parameter(torch.zeros(50), requires_grad=True) + 1.

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        gate_weights = torch.sigmoid(self.attn)
        gate_weights = gate_weights.view(nb_nodes, 1)
        return gate_weights

class GraphModel(Model):
    def __init__(self, **kwargs):
        super(GraphModel, self).__init__(**kwargs)

    def setup_layers(self):
        adj_transforms, aggregate_adj = get_transform(self.adj, self.on_cuda, num_layer=self.num_layer, pooling=self.pooling)

        self.my_layers = []
        self.out_dim = 2
        self.nb_nodes = self.adj.shape[0]
        self.channels = [self.channels] * self.num_layer
        self.aggregate_adj = aggregate_adj
        self.adj_transforms = adj_transforms
        self.master_nodes = 0
        self.input_dim = 1

        if self.embedding:
            self.add_embedding_layer()
            self.input_dim = self.emb.emb_size
        self.dims = [self.input_dim] + self.channels

        self.add_graph_convolutional_layers()
        self.add_logistic_layer()
        self.add_gating_layers()
        self.add_dropout_layers()

        if self.attention_head:
            self.attention_layer = AttentionLayer(self.channels[-1], attention_head)
            self.attention_layer.register_forward_hook(save_computations)

        self.grads = {}
        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad.data.cpu().numpy()
            return hook
        self.save_grad = save_grad

    def add_embedding_layer(self):
        self.emb = EmbeddingLayer(self.nb_nodes, self.embedding)
        self.emb.register_forward_hook(save_computations)

    def add_dropout_layers(self):
        self.dropout_layers = [None] * (len(self.dims) - 1)
        if self.dropout:
            self.dropout_layers = nn.ModuleList([torch.nn.Dropout(int(self.dropout)*min((id_layer+1) / 10., 0.4)) for id_layer in range(len(self.dims)-1)])

    def add_graph_convolutional_layers(self):
        convs = []
        for i, [c_in, c_out] in enumerate(zip(self.dims[:-1], self.dims[1:])):
            # transformation to apply at each layer.
            if self.aggregate_adj is not None:
                for extra_layer in range(self.prepool_extralayers):
                    layer = self.graph_layer_type(self.adj, c_in, c_in, self.on_cuda, i, adj_transforms=None, aggregate_adj=None)
                    convs.append(layer)

            layer = self.graph_layer_type(self.adj, c_in, c_out, self.on_cuda, i, adj_transforms=self.adj_transforms, aggregate_adj=self.aggregate_adj)
            layer.register_forward_hook(save_computations)
            convs.append(layer)
        self.conv_layers = nn.ModuleList(convs)

    def add_gating_layers(self):
        if self.gating > 0.:
            gating_layers = []
            for c_in in self.channels:
                gate = ElementwiseGateLayer(c_in)
                gate.register_forward_hook(save_computations)
                gating_layers.append(gate)
            self.gating_layers = nn.ModuleList(gating_layers)
        else:
            self.gating_layers = [None] * (len(self.dims) - 1)

    def add_logistic_layer(self):
        logistic_layer = []
        if self.attention_head > 0:
            logistic_in_dim = [self.attention_head * self.dims[-1]]
        else:
            logistic_in_dim = [self.nb_nodes * self.dims[-1]]

        for d in logistic_in_dim:
            layer = nn.Linear(d, self.out_dim)
            layer.register_forward_hook(save_computations)
            logistic_layer.append(layer)

            self.my_logistic_layers = nn.ModuleList(logistic_layer)

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()

        if self.embedding:
            x = self.emb(x)
            x.register_hook(self.save_grad('emb'))

        for i, [layer, gate, dropout] in enumerate(zip(self.conv_layers, self.gating_layers, self.dropout_layers)):

            if self.gating > 0.:
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
            x = self.attention_layer(x)[0]

        x = self.my_logistic_layers[-1](x.view(nb_examples, -1))
        x.register_hook(self.save_grad('logistic'))
        return x

    def get_representation(self):
        def add_rep(layer, name, rep):
            rep[name] = {'input': layer.input[0].cpu().data.numpy(), 'output': layer.output.cpu().data.numpy()}

        representation = {}

        if self.add_emb:
            add_rep(self.emb, 'emb', representation)

        for i, [layer, gate] in enumerate(zip(self.conv_layers, self.gating_layers)):

            if self.gating > 0.:
                add_rep(layer, 'layer_{}'.format(i), representation)
                add_rep(gate, 'gate_{}'.format(i), representation)

            else:
                add_rep(layer, 'layer_{}'.format(i), representation)

        add_rep(self.my_logistic_layers[-1], 'logistic', representation)

        if self.attention_head:
            representation['attention'] = {'input': self.attention_layer.input[0].cpu().data.numpy(),
                         'output': [self.attention_layer.output[0].cpu().data.numpy(), self.attention_layer.output[1].cpu().data.numpy()]}

        return representation

    # because of the sparse matrices.
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
            except (AttributeError, RuntimeError):
                pass # because of the sparse matrices.


class GCN(GraphModel):
    def __init__(self, **kwargs):
        super(GCN, self).__init__(**kwargs)

    def setup_layers(self):
        self.graph_layer_type = GCNLayer
        super(GCN, self).setup_layers()
        return model
