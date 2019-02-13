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

class Model(nn.Module):

    def __init__(self, name=None, column_names=None, num_epochs=100, channels=16, num_layer=2, embedding=8, gating=0., dropout=False, cuda=False, seed=0, adj=None, graph_name=None, aggregation=None, prepool_extralayers=0, lr=0.0001, patience=10, agg_reduce=2, scheduler=False, metric=sklearn.metrics.roc_auc_score):
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
        self.aggregation = aggregation
        self.lr = lr
        self.scheduler = scheduler
        self.agg_reduce=agg_reduce
        self.batch_size = 100
        self.start_patience = patience
        self.attention_head = 0
        self.train_valid_split = 0.8
        self.best_model = None
        self.metric = metric
        super(Model, self).__init__()

    def fit(self, X, y, adj=None):
        self.adj = adj
        self.X = X
        self.y = y
        start = time.time()
        self.setup_layers()
        self.adj = None
        self.X = None
        x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, stratify=y, train_size=self.train_valid_split, test_size=1-self.train_valid_split, random_state=self.seed)

        x_train = torch.FloatTensor(np.expand_dims(x_train, axis=2))
        x_valid = torch.FloatTensor(np.expand_dims(x_valid, axis=2))
        y_train = torch.FloatTensor(y_train)

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0001)
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

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
                res.append(self(inputs).data.cpu().numpy())
            y_hat = np.concatenate(res)
            auc['train'] = self.metric(y_train.numpy(), np.argmax(y_hat, axis=1))

            res = []
            for i in range(0, x_valid.shape[0], self.batch_size):
                inputs = Variable(x_valid[i:i + self.batch_size]).float()
                if self.on_cuda:
                    inputs = inputs.cuda()
                res.append(self(inputs).data.cpu().numpy())
            y_hat = np.concatenate(res)
            auc['valid'] = self.metric(y_valid, np.argmax(y_hat, axis=1))
            patience = patience - 1
            if patience == 0:
                break
            if (max_valid < auc['valid']) and epoch > 5:
                max_valid = auc['valid']
                patience = self.start_patience
                self.best_model = self.state_dict().copy()
            print("epoch: " + str(epoch) + ", time: " + "{0:.2f}".format(time.time() - start) + ", valid_metric: " + "{0:.2f}".format(auc['valid']) + ", train_metric: " + "{0:.2f}".format(auc['train']))
            if self.scheduler:
                scheduler.step()
        print("total train time:" + "{0:.2f}".format(time.time() - all_time) + " for epochs: " + str(epoch))
        self.load_state_dict(self.best_model)
        self.best_model = None

    def predict(self, inputs, probs=False):
        """ 
        Run the trained model on the inputs
        
        Args:
        inputs (torch.FloatTensor): Input to the model
        probs (bool): Get probability estimates
        """
        out = self.forward(inputs)
        if probs:
            out = F.softmax(out, dim=1)
        return out
