# File containing ML Methods (PyTorch, SKLearn Wrapper)

import time
import sklearn
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.neural_network
import sklearn.tree
import numpy as np
import torch
from torch.autograd import Variable

import models

class Method:
    def __init__(self):
        pass


class SkLearn(Method):
    def __init__(self, model, penalty=False):
        self.model = model
        self.penalty = penalty

    def loop(self, dataset, seed, train_size, test_size, adj=None):

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels, stratify=dataset.labels, train_size=train_size, test_size=test_size, random_state=seed)

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

        model = model.fit(X_train, y_train)
        return sklearn.metrics.roc_auc_score(y_test, model.predict(X_test))


class MLMethods(Method):

    def __init__(self, column_names=None, model_name="CGN", num_epochs=100, num_channel=16, num_layer=2, add_emb=8, use_gate=False, dropout=False, cuda=False, seed=0, adj=None, graph_name=None, prepool_extralayers=0):
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
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, stratify=y, train_size=self.train_valid_split, test_size=1-self.train_valid_split, random_state=self.seed)
        # Return if it's all one class
        if len(set(y_train)) == 1 or len(set(y_valid)) == 1:
            print ("Only one class represented.")
            return

        X_train = torch.FloatTensor(np.expand_dims(X_train, axis=2))
        X_valid = torch.FloatTensor(np.expand_dims(X_valid, axis=2))
        y_train = torch.FloatTensor(y_train)
        criterion = torch.nn.CrossEntropyLoss(size_average=True)

        if self.model_name == "CGN":
            adj_transform, aggregate_function = models.graphLayer.get_transform(self.adj, self.cuda, num_layer=self.num_layer)
            self.model = models.CGN(
                nb_nodes=X_train.shape[1],
                input_dim=1,
                channels=[self.num_channel] * self.num_layer,
                adj=self.adj,
                out_dim=2,
                on_cuda=self.cuda,
                add_emb=self.add_emb,
                transform_adj=adj_transform,
                aggregate_adj=aggregate_function,
                use_gate=self.use_gate,
                dropout=self.dropout,
                attention_head=self.attention_head,
                prepool_extralayers=self.prepool_extralayers,
                )
        elif self.model_name == "MLP":
            self.model = models.MLP(
                    input_dim=X_train.shape[1],
                    channels=[self.num_channel] * self.num_layer,
                    out_dim=2,
                    on_cuda=self.cuda,
                    dropout=self.dropout)
        elif self.model_name == "SLR":
            self.model = models.SparseLogisticRegression(
                    nb_nodes=X_train.shape[1],
                    input_dim=1,
                    adj=self.adj,
                    out_dim=2,
                    on_cuda=self.cuda)
        elif self.model_name == "LCG":
            adj_transform, aggregate_function = models.graphLayer.get_transform(adj, graph_name, self.cuda, num_layer=self.num_layer)
            self.model = models.LCG(
                    nb_nodes=X_train.shape[1],
                    input_dim=1,
                    channels=[self.num_channel] * self.num_layer,
                    adj=self.adj,
                    out_dim=2,
                    on_cuda=self.cuda,
                    add_emb=self.add_emb,
                    transform_adj=adj_transform,
                    aggregate_adj=aggregate_function,
                    use_gate=self.use_gate,
                    dropout=self.dropout,
                    attention_head=self.nb_attention_head)
        elif self.model_name == 'LR':
            self.model = models.LogisticRegression(
                    nb_nodes=X_train.shape[1],
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
            for base_x in range(0, X_train.shape[0], self.batch_size):
                inputs, labels = X_train[base_x:base_x + self.batch_size], y_train[base_x:base_x + self.batch_size]

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
            for base_x in range(0, X_train.shape[0], self.batch_size):
                inputs = Variable(X_train[base_x:base_x + self.batch_size]).float()
                res.append(self.model(inputs)[:, 1].data.cpu().numpy())
            y_hat = np.concatenate(res).ravel()
            auc['train'] = sklearn.metrics.roc_auc_score(y_train.numpy(), y_hat)

            res = []
            for base_x in range(0, X_valid.shape[0], self.batch_size):
                inputs = Variable(X_valid[base_x:base_x + self.batch_size]).float()
                res.append(self.model(inputs)[:, 1].data.cpu().numpy())
            y_hat = np.concatenate(res).ravel()
            auc['valid'] = sklearn.metrics.roc_auc_score(y_valid, y_hat)

            patience = patience - 1
            if patience == 0:
		self.model.load_state_dict(self.best_model)
		return self
            if (max_valid <= auc['valid']) and epoch > 5:
                max_valid = auc['valid']
                patience = self.start_patience
                self.best_model = self.model.state_dict().copy()
	self.model.load_state_dict(self.best_model)

    def predict(self, inputs):
        return self.model.forward(inputs)
