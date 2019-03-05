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
from models.utils import *
from models.models import Model
from models.gcn_layers import *
import scipy.sparse

class GCN(Model):
    def __init__(self, **kwargs):
        super(GCN, self).__init__(**kwargs)

    def setup_layers(self):
        self.master_nodes = 0
        self.in_dim = 1
        self.out_dim = len(np.unique(self.y))

        if (self.adj is None):
            raise Exception("adj must be specified for GCN")
        self.adj = scipy.sparse.csr_matrix(self.adj)
        self.adjs, self.centroids = setup_aggregates(self.adj, self.num_layer, self.X, aggregation=self.aggregation, agg_reduce=self.agg_reduce, verbose=self.verbose)
        self.nb_nodes = self.X.shape[1]

        if self.embedding:
            self.add_embedding_layer()
            self.in_dim = self.emb.emb_size
        self.dims = [self.in_dim] + self.channels
        self.add_graph_convolutional_layers()
        self.add_logistic_layer()
        self.add_gating_layers()
        self.add_dropout_layers()

        if self.attention_head:
            self.attention_layer = AttentionLayer(self.channels[-1], self.attention_head)

        torch.manual_seed(self.seed)
        if self.on_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            self.cuda()

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()

        if self.embedding:
            x = self.emb(x)

        for i, [conv, gate, dropout] in enumerate(zip(self.conv_layers, self.gating_layers, self.dropout_layers)):
            for prepool_conv in self.prepool_conv_layers[i]:
                x = prepool_conv(x)

            if self.gating > 0.:
                x = conv(x)
                g = gate(x)
                x = g * x
            else:
                x = conv(x)

            if dropout is not None:
                id_to_keep = dropout(torch.FloatTensor(np.ones((x.size(0), x.size(1))))).unsqueeze(2)
                if self.on_cuda:
                    id_to_keep = id_to_keep.cuda()
                x = x * id_to_keep

        # Do attention pooling here
        if self.attention_head:
            x = self.attention_layer(x)[0]

        x = self.my_logistic_layers[-1](x.view(nb_examples, -1))
        return x


    def add_embedding_layer(self):
        self.emb = EmbeddingLayer(self.nb_nodes, self.embedding)

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
                extra_layer = GCNLayer(self.adjs[i], c_in, c_in, self.on_cuda, i, torch.LongTensor(np.array(range(self.adjs[i].shape[0]))))
                extra_layers.append(extra_layer)

            prepool_convs.append(nn.ModuleList(extra_layers))

            layer = GCNLayer(self.adjs[i], c_in, c_out, self.on_cuda, i, torch.tensor(self.centroids[i]))
            convs.append(layer)
        self.conv_layers = nn.ModuleList(convs)
        self.prepool_conv_layers = prepool_convs

    def add_gating_layers(self):
        if self.gating > 0.:
            gating_layers = []
            for c_in in self.channels:
                gate = ElementwiseGateLayer(c_in)
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
            logistic_layers.append(layer)
        self.my_logistic_layers = nn.ModuleList(logistic_layers)

    def get_representation(self):
        def add_rep(layer, name, rep):
            rep[name] = {'input': layer.input[0].cpu().data.numpy(), 'output': layer.output.cpu().data.numpy()}

        representation = {}

        if self.embedding:
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
