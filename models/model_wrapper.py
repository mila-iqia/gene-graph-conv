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


class SkLearn(Method):
    def __init__(self, model, penalty=False):
        self.model = model
        self.penalty = penalty

    def loop(self, dataset, seed, train_size, test_size, adj=None):

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels, stratify=dataset.labels, train_size=train_size, test_size=test_size, random_state=seed)

        if self.model == "LR":
            model = sklearn.linear_model.LogisticRegression()
            if self.penalty:
                model = sklearn.linear_model.LogisticRegression(penalty='l1', tol=0.0001)
        elif self.model == "DT":
            model = sklearn.tree.DecisionTreeClassifier()
        elif self.model == "MLP":
            model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(32, 3), learning_rate_init=0.001, early_stopping=False,  max_iter=1000)
        else:
            print "incorrect label"

        model = model.fit(x_train, y_train)
        return sklearn.metrics.roc_auc_score(y_test, model.predict(x_test))


class WrappedModel(Method):

    def __init__(self, column_names=None, model_name="GCN", num_epochs=100, num_channel=16, num_layer=2, add_emb=8, use_gate=False, dropout=False, cuda=False, seed=0, adj=None, graph_name=None, prepool_extralayers=0):
        self.model_name = model_name
        self.column_names = column_names
        self.model = None
        self.batch_size = 10
        self.num_channel = num_channel
        self.num_layer = num_layer
        self.add_emb = add_emb
        self.use_gate = use_gate
        self.dropout = dropout
        self.cuda = cuda
        self.num_epochs = num_epochs
        self.start_patience = 10
        self.attention_head = 0
        self.seed = seed
        self.adj = adj
        self.graph_name = graph_name
        self.train_valid_split = 0.8
        self.prepool_extralayers = prepool_extralayers
        self.best_model = None

    def fit(self, X, y, adj=None):
        self.adj = adj
        x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, stratify=y, train_size=self.train_valid_split, test_size=1-self.train_valid_split, random_state=self.seed)
        # Return if it's all one class
        if len(set(y_train)) == 1 or len(set(y_valid)) == 1:
            print "Only one class represented."
            return

        # pylint: disable=E1101
        x_train = torch.FloatTensor(np.expand_dims(x_train, axis=2))
        x_valid = torch.FloatTensor(np.expand_dims(x_valid, axis=2))
        y_train = torch.FloatTensor(y_train)
        # pylint: enable=E1101

        criterion = torch.nn.CrossEntropyLoss(size_average=True)

        if self.model_name == "GCN":
            adj_transform, aggregator_fn = get_transform(adj, self.cuda, num_layer=self.num_layer)
            self.model = GCN(
                nb_nodes=x_train.shape[1],
                input_dim=1,
                channels=[self.num_channel] * self.num_layer,
                adj=self.adj,
                out_dim=2,
                on_cuda=self.cuda,
                add_emb=self.add_emb,
                transform_adj=adj_transform,
                aggregate_adj=aggregator_fn,
                use_gate=self.use_gate,
                dropout=self.dropout,
                attention_head=self.attention_head,
                prepool_extralayers=self.prepool_extralayers,
                )
        elif self.model_name == "MLP":
            self.model = MLP(
                input_dim=x_train.shape[1],
                channels=[self.num_channel] * self.num_layer,
                out_dim=2,
                on_cuda=self.cuda,
                dropout=self.dropout)
        elif self.model_name == "SLR":
            self.model = SparseLogisticRegression(
                nb_nodes=x_train.shape[1],
                input_dim=1,
                adj=self.adj,
                out_dim=2,
                on_cuda=self.cuda)
        elif self.model_name == 'LR':
            self.model = LogisticRegression(
                nb_nodes=x_train.shape[1],
                input_dim=1,
                out_dim=2,
                on_cuda=self.cuda)

        if self.cuda:
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
                if self.cuda:
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
                res.append(self.model(inputs)[:, 1].data.cpu().numpy())
            y_hat = np.concatenate(res).ravel()
            auc['train'] = sklearn.metrics.roc_auc_score(y_train.numpy(), y_hat)

            res = []
            for i in range(0, x_valid.shape[0], self.batch_size):
                inputs = Variable(x_valid[i:i + self.batch_size]).float()
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
