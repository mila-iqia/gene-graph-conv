import argparse
import logging
import tensorflow as tf  # necessary to import here to avoid segfault
from data.utils import get_dataset, split_dataset
from data.graph import Graph
from models.models import get_model, setup_l1_loss
import torch
import time
from torch.autograd import Variable
from analysis import monitoring
from analysis.metrics import record_metrics_for_epoch, summarize, record_metrics_mse
import optimization as otim
import sys
import time
import copy

from itertools import repeat
import data, data.gene_datasets
import sklearn, sklearn.model_selection, sklearn.metrics, sklearn.linear_model, sklearn.neural_network, sklearn.tree
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import gene_inference
#from gene_inference.infer_genes import infer_all_genes, sample_neighbors
import models, models.graphLayer
from models.models import CGN
import data, data.gene_datasets
from data.graph import Graph
from data.utils import split_dataset
import optimization
import torch
from torch.autograd import Variable
from analysis.metrics import record_metrics_for_epoch
import analysis
reload(analysis.metrics)
reload(gene_inference);
import os


parser = argparse.ArgumentParser()
parser.add_argument('--gene', default="S100A8", type=str)
parser.add_argument('--cuda', action='store_true', help='If we want to run on gpu.')
opt = parser.parse_args()


tcgatissue = data.gene_datasets.TCGATissue()


opt.seed = 0
opt.nb_class = None
opt.nb_examples = None
opt.nb_nodes = None
opt.graph = "pathway"
opt.dataset = tcgatissue
opt.add_self = True
opt.norm_adj = True
opt.add_connectivity = False
opt.cuda = True
opt.pool_graph = "ignore"







graph = Graph()
path = "/data/lisa/data/genomics/graph/pancan-tissue-graph.hdf5"
graph.load_graph(path)
#graph.intersection_with(tcgatissue)
g = nx.from_numpy_matrix(graph.adj)
mapping = dict(zip(range(0, len(tcgatissue.df.columns)), tcgatissue.df.columns))
g = nx.relabel_nodes(g, mapping)




def sample_neighbors(g, gene, num_neighbors, include_self=True):
    results = set([])
    if include_self:
        results = set([gene])
    all_nodes = set(g.nodes)
    first_degree = set(g.neighbors(gene))
    second_degree = set()
    for x in g.neighbors(gene):
        second_degree = second_degree.union(set(g.neighbors(x)))
    while len(results) < num_neighbors:
        if len(first_degree) - len(results) > 0:
            unique = sorted(first_degree - results)
            results.add(unique.pop())
        elif len(second_degree) - len(results) > 0:
            unique = sorted(second_degree - results)
            results.add(unique.pop())
        else:
            unique = sorted(all_nodes - results)
            results.add(unique.pop())
    return results



import sklearn, sklearn.model_selection, sklearn.metrics, sklearn.linear_model, sklearn.neural_network, sklearn.tree
import numpy as np

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
            model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(32,3), learning_rate_init=0.001, early_stopping=False,  max_iter=1000)
        else:
            print "incorrect label"

        model = model.fit(X_train, y_train)
        return sklearn.metrics.roc_auc_score(y_test, model.predict(X_test))


class PyTorch(Method):

    def __init__(self, model, num_epochs=100, num_channel=64, num_layer=3, add_emb=32, use_gate=False, dropout=True, cuda=True, attention_head=0, l1_loss_lambda=0, model_reg_lambda=0, training_mode=None):
        self.model = model
        self.batch_size = 10
        self.num_channel = num_channel
        self.num_layer = num_layer
        self.add_emb = add_emb
        self.use_gate = use_gate
        self.dropout = dropout
        self.cuda = cuda
        self.num_epochs = num_epochs
        self.patience = 10
        self.attention_head = attention_head
        self.l1_loss_lambda = l1_loss_lambda
        self.model_reg_lambda = model_reg_lambda
        self.training_mode = training_mode

    def loop(self, dataset, seed, train_size, test_size, adj=None):

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels, stratify=dataset.labels, train_size=train_size*2, test_size=test_size, random_state=seed)

        #split train into valid and train
        local_X_train, local_X_valid, local_y_train, local_y_valid = sklearn.model_selection.train_test_split(X_train, y_train, stratify=y_train, train_size=0.50, random_state=seed)
        local_X_train = torch.FloatTensor(np.expand_dims(local_X_train, axis=2))
        local_X_valid = torch.FloatTensor(np.expand_dims(local_X_valid, axis=2))
        X_test = torch.FloatTensor(np.expand_dims(X_test, axis=2))

        local_y_train = torch.FloatTensor(local_y_train)

        criterion = optimization.get_criterion(dataset)
        patience = self.patience
        opt.num_layer = self.num_layer
        adj_transform, aggregate_function = models.graphLayer.get_transform(opt, adj)

        if self.model == "CGN":
            model = models.models.CGN(
                    nb_nodes=len(dataset.df.columns),
                    input_dim=1,
                    channels=[self.num_channel] * self.num_layer,
                    adj=adj,
                    out_dim=2,
                    on_cuda=self.cuda,
                    add_emb=self.add_emb,
                    transform_adj=adj_transform,
                    aggregate_adj=aggregate_function,
                    use_gate=self.use_gate,
                    dropout=self.dropout,
                    attention_head=self.attention_head
                    )
        elif self.model == "MLP":
            model = models.models.MLP(
                    len(dataset.df.columns),
                    channels=[self.num_channel] * self.num_layer,
                    out_dim=2,
                    on_cuda=self.cuda,
                    dropout=self.dropout)
        elif self.model == "SLR":
            model = models.models.SparseLogisticRegression(
                    nb_nodes=len(dataset.df.columns),
                    input_dim=1,
                    adj=adj,
                    out_dim=2,
                    on_cuda=self.cuda)
        elif self.model == "LCG":
            model = models.models.LCG(
                    nb_nodes=len(dataset.df.columns),
                    input_dim=1,
                    channels=[self.num_channel] * self.num_layer,
                    adj=adj,
                    out_dim=2,
                    on_cuda=self.cuda,
                    add_emb=self.add_emb,
                    transform_adj=adj_transform,
                    aggregate_adj=aggregate_function,
                    use_gate=self.use_gate,
                    dropout=self.dropout,
                    attention_head=nb_attention_head,
                    training_mode=training_mode)



        if self.cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model.cuda()

        l1_criterion = torch.nn.L1Loss(size_average=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        max_valid = 0
        max_valid_test = 0
        for t in range(0, self.num_epochs):
            start_timer = time.time()

            for base_x in range(0,local_X_train.shape[0], self.batch_size):
                inputs, labels = local_X_train[base_x:base_x+self.batch_size], local_y_train[base_x:base_x+self.batch_size]

                inputs = Variable(inputs, requires_grad=False).float()
                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                model.train()
                y_pred = model(inputs)


                # Compute and print loss
                crit_loss = optimization.compute_loss(criterion, y_pred, labels, self.training_mode)
                model_regularization_loss = model.regularization(self.model_reg_lambda)
                l1_loss = models.models.setup_l1_loss(model, self.l1_loss_lambda, l1_criterion, opt.cuda)
                total_loss = crit_loss + model_regularization_loss + l1_loss


                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                crit_loss.backward()
                optimizer.step()
                model.eval()

            auc = {}
            res = []
            for base_x in range(0,local_X_train.shape[0], self.batch_size):
                inputs = Variable(local_X_train[base_x:base_x+self.batch_size], requires_grad=False).float()
                res.append(model(inputs.cuda())[:,1].data.cpu().numpy())
            auc['train'] = sklearn.metrics.roc_auc_score(local_y_train.numpy(), np.asarray(res).flatten())

            res = []
            for base_x in range(0,local_X_valid.shape[0], self.batch_size):
                inputs = Variable(local_X_valid[base_x:base_x+self.batch_size], requires_grad=False).float()
                res.append(model(inputs.cuda())[:,1].data.cpu().numpy())
            auc['valid'] = sklearn.metrics.roc_auc_score(local_y_valid, np.asarray(res).flatten())

            res = []
            for base_x in range(0,X_test.shape[0], self.batch_size):
                inputs = Variable(X_test[base_x:base_x+self.batch_size], requires_grad=False).float()
                res.append(model(inputs.cuda())[:,1].data.cpu().numpy())
            auc['test'] = sklearn.metrics.roc_auc_score(y_test, np.asarray(res).flatten())


            time_this_epoch = time.time() - start_timer

#eval on cpu
#             auc['train'] = sklearn.metrics.roc_auc_score(local_y_train.numpy(), model(Variable(local_X_train.cpu(), requires_grad=False).float())[:,1].cpu().data.numpy())
#             auc['valid'] = sklearn.metrics.roc_auc_score(local_y_valid, model(Variable(local_X_valid.cpu(), requires_grad=False).float())[:,1].cpu().data.numpy())
#             auc['test'] = sklearn.metrics.roc_auc_score(y_test, model(Variable(X_test.cpu(), requires_grad=False).float())[:,1].cpu().data.numpy())

            summary = [ t, crit_loss.data[0], auc['train'], auc['valid'], time_this_epoch ]
            summary = "epoch {}, cross_loss: {:.03f}, auc_train: {:0.3f}, auc_valid:{:0.3f}, time: {:.02f} sec".format(*summary)
            #print summary

            patience = patience - 1
            if patience == 0:
                return max_valid_test
                break
            if (max_valid < auc['valid']) and t > 5:
                max_valid = auc['valid']
                max_valid_test = auc['test']
                patience = self.patience



def method_comparison(results, dataset, models, gene, search_num_genes, trials, search_train_size, test_size):

    dataset = data.gene_datasets.TCGATissue()
    dataset.df = dataset.df - dataset.df.mean()

    mean = dataset.df[gene].mean()
    dataset.labels = [1 if x > mean else 0 for x in dataset.df[gene]]
    full_df = dataset.df.copy(deep=True)

    for train_size in search_train_size:
        for ex in search_num_genes:

            num_genes = ex
            num_genes = np.min([num_genes, tcgatissue.df.shape[1]])
            print ex, num_genes

            neighbors = sample_neighbors(g, gene, num_genes, include_self=True)
            print "neighbors", len(neighbors), "train_size", train_size

            dataset.df = dataset.df[list(neighbors)]
            dataset.df[gene] = 1
            dataset.data = dataset.df.as_matrix()

            neighborhood = np.asarray(nx.to_numpy_matrix(nx.Graph(g.subgraph(neighbors))))
            for model in models:
                for seed in range(trials):

                    #have we already done it?
                    already_done = results["df"][(results["df"].gene_name == gene) &
                                                 (results["df"].model == model['key']) &
                                                 (results["df"].num_genes == num_genes) &
                                                 (results["df"].seed == seed) &
                                                 (results["df"].train_size == train_size)].shape[0] > 0

                    if already_done:
                        print "already done:", model['key'], "num_genes", num_genes, "train_size", train_size, "seed", seed
                        continue
                    print "doing:", model['key'], "num_genes", num_genes, "train_size", train_size, "seed", seed

                    result = model['method'].loop(dataset=dataset, seed=seed, train_size=train_size, test_size=test_size, adj=neighborhood)

                    experiment = {"gene_name": gene,
                            "model": model['key'],
                            "num_genes": num_genes,
                            "seed":seed,
                            "train_size": train_size,
                            "auc":result
                            }

                    results["df"] = results["df"].append(experiment, ignore_index=True)
                    pickle.dump(results, open("results_" + gene + ".pkl", "wb"))
            dataset.df = full_df





import pickle

m = [
#    {'key': 'LR-L1', 'method': SkLearn("LR", penalty=True)},
#    {'key': 'MLP', 'method': mlp},
#    {'key': 'DT', 'method': SkLearn("DT")},
    {'key': 'CGN_lay3_chan64_emb32_dropout', 'method': PyTorch("CGN", num_layer=3,num_channel=64, add_emb=32 )},
#{'key': 'CGN_3_layer_64_channel_emb_32_dropout', 'method': PyTorch("CGN")},
# #    {'key': 'wRPL5_CGN_3_layer_64_channel_emb_32_dropout_attn3', 'method': PyTorch("CGN", attention_head=3)},
#  #{'key': 'wRPL5_CGN_1_layer_32_channel_emb_32_dropout', 'method': PyTorch("CGN", num_layer=1, num_channel=32)},
#     {'key': 'MLP-dropout', 'method': PyTorch("MLP", dropout=True)},
#     {'key': 'CGN_2_layer_512_channel_emb_32_dropout', 'method': PyTorch("CGN", num_layer=2, num_channel=512, add_emb=32)},
#     {'key': 'CGN_3_layer_512_channel_emb_32_dropout', 'method': PyTorch("CGN", num_layer=3, num_channel=512, add_emb=32)},
#     {'key': 'CGN_1_layer_768_channel_emb_32_dropout', 'method': PyTorch("CGN", num_layer=1, num_channel=768, add_emb=32)},
#     {'key': 'CGN_2_layer_512_channel_emb_128_dropout', 'method': PyTorch("CGN", num_layer=2, num_channel=512, add_emb=128)},
#     {'key': 'CGN_2_layer_512_channel_emb_256_dropout', 'method': PyTorch("CGN", num_layer=2, num_channel=512, add_emb=256)},
#    {'key': 'CGN_2_layer_512_channel_emb_512_dropout', 'method': PyTorch("CGN", num_layer=2, num_channel=512, add_emb=512)},

    #    {'key': 'MLP_2_chan128', 'method': PyTorch("MLP", dropout=False, num_layer=2, num_channel=128)},
#    {'key': 'MLP_2_chan256', 'method': PyTorch("MLP", dropout=False, num_layer=2, num_channel=256)},
    {'key': 'MLP_lay2_chan512_dropout', 'method': PyTorch("MLP", dropout=True, num_layer=2, num_channel=512)},
     {'key': 'MLP_lay2_chan512', 'method': PyTorch("MLP", dropout=False, num_layer=2, num_channel=512)},
#    {'key': 'MLP_2_chan1024', 'method': PyTorch("MLP", dropout=False, num_layer=2, num_channel=1024)},
#    {'key': 'MLP_2', 'method': PyTorch("MLP", dropout=False, num_layer=2)},
#    {'key': 'MLP_1', 'method': PyTorch("MLP", dropout=False, num_layer=1)},
#     {'key': 'MLP-dropout-l1-1', 'method': PyTorch("MLP", dropout=True, l1_loss_lambda=1)},
#     {'key': 'SLR2=lambda1', 'method': PyTorch("SLR", model_reg_lambda=1)},
#    {'key': 'SLR2=lambda100', 'method': PyTorch("SLR", model_reg_lambda=100)},
     {'key': 'SLR_lambda1_l11', 'method': PyTorch("SLR", model_reg_lambda=1, l1_loss_lambda=1)},
#   {'key': 'MLP', 'method': PyTorch("MLP", dropout=False)},
    ]


import pickle

print opt
if not os.path.exists("results_" + opt.gene + ".pkl"):
    results = {"df": pd.DataFrame(columns=['auc','gene_name', 'model', 'num_genes', 'seed', 'train_size'])}
else:
    results = pickle.load(open("results_" + opt.gene + ".pkl", "r"))


#"S100A8
#RPL5
#RPS10

trials = 20

a = method_comparison(results, tcgatissue, m, gene=opt.gene,
                      search_num_genes=[50, 100,200,300,500,1000,2000,4000,8000,16000],
                      trials=trials, search_train_size=[50], test_size=1000)

#a = method_comparison(results, tcgatissue, m, gene=opt.gene,
#                      search_num_genes=[100], trials=trials,
#                      search_train_size=[50, 100, 200, 500, 1000, 2000, 4000], test_size=1000)
