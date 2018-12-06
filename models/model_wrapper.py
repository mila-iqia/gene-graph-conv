"""A SKLearn-style wrapper around our PyTorch models (like Graph Convolutional Network and SparseLogisticRegression) implemented in models.py"""

import sklearn
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.neural_network
import sklearn.tree
import numpy as np
import torch
from torch.autograd import Variable

from models.model_layers import GCN, SparseLogisticRegression, LogisticRegression, MLP
from models.graph_layers import get_transform

class Method:
    def __init__(self):
        pass

class WrappedModel(Method):

    def __init__(self, name="GCN", column_names=None, num_epochs=100, channels=16, num_layer=2, embedding=8, gating=False, dropout=False, cuda=False, seed=0, adj=None, graph_name=None, pooling="ignore", prepool_extralayers=0):
        self.name = name
        self.model_type = self.name.split("_")[0]
        self.column_names = column_names
        self.model = None
        self.batch_size = 10
        self.channels = channels
        self.num_layer = num_layer
        self.embedding = embedding
        self.gating = gating
        self.dropout = dropout
        self.on_cuda = cuda
        self.num_epochs = num_epochs
        self.start_patience = 10
        self.attention_head = 0
        self.seed = seed
        self.adj = adj
        self.graph_name = graph_name
        self.train_valid_split = 0.8
        self.prepool_extralayers = prepool_extralayers
        self.pooling = pooling
        self.best_model = None

    def fit(self, X, y, adj=None):
        self.adj = adj
        x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, stratify=y, train_size=self.train_valid_split, test_size=1-self.train_valid_split, random_state=self.seed)

        # pylint: disable=E1101
        x_train = torch.FloatTensor(np.expand_dims(x_train, axis=2))
        x_valid = torch.FloatTensor(np.expand_dims(x_valid, axis=2))
        y_train = torch.FloatTensor(y_train)
        # pylint: enable=E1101

        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')

        if self.model_type == "GCN":
            adj_transform, aggregator_fn = get_transform(adj, self.on_cuda, num_layer=self.num_layer, pooling=self.pooling)
            self.model = GCN(
                input_dim=1,
                channels=[self.channels] * self.num_layer,
                adj=self.adj,
                out_dim=2,
                cuda=self.on_cuda,
                embedding=self.embedding,
                transform_adj=adj_transform,
                aggregate_adj=aggregator_fn,
                gating=self.gating,
                dropout=self.dropout,
                attention_head=self.attention_head,
                prepool_extralayers=self.prepool_extralayers,
                )
        elif self.model_type == "MLP":
            self.model = MLP(
                input_dim=x_train.shape[1],
                channels=[self.channels] * self.num_layer,
                out_dim=2,
                cuda=self.on_cuda,
                dropout=self.dropout)
        elif self.model_type == "SLR":
            self.model = SparseLogisticRegression(
                nb_nodes=x_train.shape[1],
                input_dim=1,
                adj=self.adj,
                out_dim=2,
                cuda=self.on_cuda)
        elif self.model_type == 'LR':
            self.model = LogisticRegression(
                nb_nodes=x_train.shape[1],
                input_dim=1,
                out_dim=2,
                cuda=self.on_cuda)

        if self.on_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            self.model.cuda()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        max_valid = 0
        patience = self.start_patience
        for epoch in range(0, self.num_epochs):
            for i in range(0, x_train.shape[0], self.batch_size):
                inputs, labels = x_train[i:i + self.batch_size], y_train[i:i + self.batch_size]

                inputs = Variable(inputs, requires_grad=False).float()
                if self.on_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                self.model.train()
                y_pred = self.model(inputs)

                targets = Variable(labels, requires_grad=False).long()
                loss = criterion(y_pred, targets)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.model.eval()

            auc = {'train': 0., 'valid': 0.}
            res = []
            for i in range(0, x_train.shape[0], self.batch_size):
                inputs = Variable(x_train[i:i + self.batch_size]).float()
                if self.on_cuda:
                    inputs = inputs.cuda()
                res.append(self.model(inputs)[:, 1].data.cpu().numpy())
            y_hat = np.concatenate(res).ravel()
            auc['train'] = sklearn.metrics.roc_auc_score(y_train.numpy(), y_hat)

            res = []
            for i in range(0, x_valid.shape[0], self.batch_size):
                inputs = Variable(x_valid[i:i + self.batch_size]).float()
                if self.on_cuda:
                    inputs = inputs.cuda()
                res.append(self.model(inputs)[:, 1].data.cpu().numpy())
            y_hat = np.concatenate(res).ravel()
            auc['valid'] = sklearn.metrics.roc_auc_score(y_valid, y_hat)

            patience = patience - 1
            if patience == 0:
                break
            if (max_valid <= auc['valid']) and epoch > 5:
                max_valid = auc['valid']
                patience = self.start_patience
                self.best_model = self.model.state_dict().copy()
        self.model.load_state_dict(self.best_model)

    def predict(self, inputs):
        """ Run the trained model on the inputs"""
        return self.model.forward(inputs)
