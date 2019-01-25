"""A SKLearn-style wrapper around our PyTorch models (like Graph Convolutional Network and SparseLogisticRegression) implemented in models.py"""

import logging
import time
import itertools
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
from scipy import sparse
from models.utils import setup_aggregates, max_pool_torch_scatter, max_pool_dense_iter, max_pool_dense, sparse_max_pool, norm_laplacian, hierarchical_clustering, random_clustering, kmeans_clustering

# For Monitoring
def save_computations(self, input, output):
    setattr(self, "input", input)
    setattr(self, "output", output)

class Model(nn.Module):

    def __init__(self, name=None, column_names=None, num_epochs=100, channels=16, num_layer=2, embedding=8, gating=0.0001, dropout=False, cuda=False, seed=0, adj=None, graph_name=None, pooling=None, prepool_extralayers=0):
        self.name = name
        self.column_names = column_names
        self.num_layer = num_layer
        self.channels = [channels] * self.num_layer
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
        self.adj = sparse.csr_matrix(adj)
        self.X = X
        start = time.time()
        self.setup_layers()
        print("setup layers took: " + str(time.time() - start))
        # Cleanup these vars, todo refactor them from setup_layers()
        self.adj = None
        self.X = None
        try:
            x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, stratify=y, train_size=self.train_valid_split, test_size=1-self.train_valid_split, random_state=self.seed)
        except ValueError as e:
            print(e)
            self.best_model = self.state_dict().copy()
            return

        # pylint: disable=E1101
        x_train = torch.FloatTensor(np.expand_dims(x_train, axis=2))
        x_valid = torch.FloatTensor(np.expand_dims(x_valid, axis=2))
        y_train = torch.FloatTensor(y_train.values)
        # pylint: enable=E1101

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)
        max_valid = 0
        patience = self.start_patience
        self.best_model = self.state_dict().copy()
        all_time = time.time()
        for epoch in range(0, self.num_epochs):

            start = time.time()
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
            start = time.time()

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
            if (max_valid < auc['valid']) and epoch > 5:
                max_valid = auc['valid']
                patience = self.start_patience
                self.best_model = self.state_dict().copy()
            #print("epoch: " + str(epoch) + " " + str(time.time() - start))
        print("total train time:" + str(time.time() - all_time) + " for epochs: " + str(epoch))
        self.load_state_dict(self.best_model)
        self.best_model = None

    def predict(self, inputs):
        """ Run the trained model on the inputs"""
        return self.forward(inputs)


class MLP(Model):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

    def setup_layers(self):
        out_dim = 2
        in_dim = len(self.X.keys())
        dims = [in_dim] + self.channels
        logging.info("Constructing the network...")

        layers = []
        for c_in, c_out in zip(dims[:-1], dims[1:]):
            layer = nn.Linear(c_in, c_out)
            layers.append(layer)
        self.my_layers = nn.ModuleList(layers)

        if self.channels:
            self.last_layer = nn.Linear(self.channels[-1], out_dim)
        else:
            self.last_layer = nn.Linear(in_dim, out_dim)

        self.my_dropout = torch.nn.Dropout(0.5) if self.dropout else None
        logging.info("Done!")

        torch.manual_seed(self.seed)
        if self.on_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            self.cuda()

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
        self.in_dim = 1
        self.out_dim = 2
        import pdb ;pdb.set_trace()
        self.adj.setdiag(0.)
        D = self.adj.sum(0) + 1e-5
        laplacian = np.eye(D.shape[0]) - np.diag((D**-0.5)).dot(self.adj).dot(np.diag((D**-0.5)))
        self.laplacian = torch.FloatTensor(laplacian)

        # The logistic layer.
        logistic_in_dim = self.nb_nodes * self.in_dim
        logistic_layer = nn.Linear(logistic_in_dim, self.out_dim)
        logistic_layer.register_forward_hook(save_computations)
        self.my_logistic_layers = nn.ModuleList([logistic_layer])

        torch.manual_seed(self.seed)
        if self.on_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            self.cuda()

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
        self.in_dim = 1
        self.out_dim = 2

        # The logistic layer.
        logistic_in_dim = self.nb_nodes * self.in_dim
        logistic_layer = nn.Linear(logistic_in_dim, self.out_dim)
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
        self.master_nodes = 0
        self.in_dim = 1
        self.out_dim = 2
        self.nb_nodes = self.adj.shape[0]

        if self.embedding:
            self.add_embedding_layer()
            self.in_dim = self.emb.emb_size
        self.dims = [self.in_dim] + self.channels
        self.adjs, self.centroids = setup_aggregates(self.adj, self.num_layer, cluster_type=self.pooling)
        self.add_graph_convolutional_layers()
        self.add_logistic_layer()
        self.add_gating_layers()
        self.add_dropout_layers()

        if self.attention_head:
            self.attention_layer = AttentionLayer(self.channels[-1], self.attention_head)
            self.attention_layer.register_forward_hook(save_computations)

        self.grads = {}
        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad.data.cpu().numpy()
            return hook
        self.save_grad = save_grad

        torch.manual_seed(self.seed)
        if self.on_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            self.cuda()

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()

        if self.embedding:
            x = self.emb(x)
            x.register_hook(self.save_grad('emb'))

        for i, [conv, gate, dropout] in enumerate(zip(self.conv_layers, self.gating_layers, self.dropout_layers)):
            for prepool_conv in self.prepool_conv_layers[i]:
                x = prepool_conv(x)

            if self.gating > 0.:
                x = conv(x)
                g = gate(x)
                x = g * x
            else:
                x = conv(x)

            x = F.relu(x)
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


    def add_embedding_layer(self):
        self.emb = EmbeddingLayer(self.nb_nodes, self.embedding)
        self.emb.register_forward_hook(save_computations)

    def add_dropout_layers(self):
        self.dropout_layers = [None] * (len(self.dims) - 1)
        if self.dropout:
            self.dropout_layers = nn.ModuleList([torch.nn.Dropout(int(self.dropout)*min((id_layer+1) / 10., 0.4)) for id_layer in range(len(self.dims)-1)])

    def add_graph_convolutional_layers(self):
        convs = []
        prepool_convs = nn.ModuleList([])
        for i, [c_in, c_out] in enumerate(zip(self.dims[:-1], self.dims[1:])):
            # transformation to apply at each layer.
            extra_layers = []
            for _ in range(self.prepool_extralayers):
                extra_layer = self.graph_layer_type(self.adjs[i], c_in, c_out, self.on_cuda, i, torch.tensor([]), self.exp)
                extra_layer.register_forward_hook(save_computations)
                extra_layers.append(extra_layer)

            prepool_convs.append(nn.ModuleList(extra_layers))

            layer = self.graph_layer_type(self.adjs[i], c_in, c_out, self.on_cuda, i, torch.tensor(self.centroids[i]), self.exp)
            layer.register_forward_hook(save_computations)
            convs.append(layer)
        self.conv_layers = nn.ModuleList(convs)
        self.prepool_conv_layers = prepool_convs

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
        logistic_layers = []
        if self.attention_head > 0:
            logistic_in_dim = [self.attention_head * self.dims[-1]]
        else:
            logistic_in_dim = [self.adjs[-1].shape[0] * self.dims[-1]]
        for d in logistic_in_dim:
            layer = nn.Linear(d, self.out_dim)
            layer.register_forward_hook(save_computations)
            logistic_layers.append(layer)
        self.my_logistic_layers = nn.ModuleList(logistic_layers)

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
        self.graph_layer_type = GCNLayer
        super(GCN, self).__init__(**kwargs)


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    From: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class GCNLayer(nn.Module):
    def __init__(self, adj, in_dim=1, channels=1, cuda=False, id_layer=None, centroids=None, exp=None):
        super(GCNLayer, self).__init__()

        adj.setdiag(np.ones(adj.shape[0]))

        self.my_layers = []
        self.cuda = cuda
        self.nb_nodes = adj.shape[0]
        self.in_dim = in_dim
        self.channels = channels
        self.id_layer = id_layer
        self.adj = adj
        self.centroids = centroids
        self.exp = exp
        edges = torch.LongTensor(np.array(self.adj.nonzero()))
        sparse_adj = torch.sparse.FloatTensor(edges, torch.FloatTensor(self.adj.data), torch.Size([self.nb_nodes, self.nb_nodes]))
        self.dense_adj = sparse_adj.to_dense()
        self.register_buffer('sparse_adj', sparse_adj)

        self.linear = nn.Conv1d(in_channels=self.in_dim, out_channels=int(self.channels/2), kernel_size=1, bias=True)
        self.eye_linear = nn.Conv1d(in_channels=self.in_dim, out_channels=int(self.channels/2), kernel_size=1, bias=True)

        self.sparse_adj = self.sparse_adj.cuda() if self.cuda else self.sparse_adj
        self.centroids = self.centroids.cuda() if self.cuda else self.centroids
        self.dense_adj = self.dense_adj.cuda() if self.cuda else self.dense_adj

    def _adj_mul(self, x, D):
        nb_examples, nb_channels, nb_nodes = x.size()
        x = x.view(-1, nb_nodes)

        # Needs this hack to work: https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
        #x = D.mm(x.t()).t()
        x = SparseMM(D)(x.t()).t()

        x = x.contiguous().view(nb_examples, nb_channels, nb_nodes)
        return x

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()  # from ex, node, ch, -> ex, ch, node

        adj = Variable(self.sparse_adj, requires_grad=False)

        eye_x = self.eye_linear(x)

        x = self._adj_mul(x, adj)

        x = torch.cat([self.linear(x), eye_x], dim=1)
        shape = x.size()
        if self.exp == "sparse_max_pool":
            res = sparse_max_pool(x, self.centroids, self.dense_adj)
        elif self.exp == "max_pool_dense":
            res = max_pool_dense(x, self.centroids, self.dense_adj)
        elif self.exp == "max_pool_dense_iter":
            res = max_pool_dense_iter(x, self.centroids, self.dense_adj)
        elif self.exp == "max_pool_torch_scatter":
            res = max_pool_torch_scatter(x, self.centroids)
        return res
