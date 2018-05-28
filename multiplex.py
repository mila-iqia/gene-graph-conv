import sys
import time
import copy
import pickle
import argparse

from scipy.stats import norm
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
import sklearn, sklearn.model_selection, sklearn.metrics, sklearn.linear_model, sklearn.neural_network, sklearn.tree
import numpy as np

# 3 graphs, all genes
# 3 graphs, reduce to only nodes with
# just MLP
# add 1 trial each time
# do a diff on the big results graph, I can do a merge (I think at the end)
# put images in the paper
#do just one trial and add logic to do repeated trials

def sample_neighbors(g, gene, num_neighbors, include_self=True):
    results = set([])
    if include_self:
        results = set([gene])
    all_nodes = set(g.nodes)
    try:
        first_degree = set(g.neighbors(gene))
        second_degree = set()
        for x in g.neighbors(gene):
            second_degree = second_degree.union(set(g.neighbors(x)))
    except:
        first_degree = all_nodes
        second_degree = all_nodes

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

    def __init__(self, model, num_epochs=100, num_channel=16, num_layer=2, add_emb=8, use_gate=False, dropout=False, cuda=False):
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
        self.attention_head = 0

    def loop(self, dataset, seed, train_size, test_size, adj=None):

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels, stratify=dataset.labels, train_size=train_size, test_size=test_size, random_state=seed)

        #split train into valid and train
        if len(set(y_train)) == 1 or len(set(y_test)) == 1:
            return
        try:
            local_X_train, local_X_valid, local_y_train, local_y_valid = sklearn.model_selection.train_test_split(X_train, y_train, stratify=y_train, train_size=0.60, random_state=seed)
        except Exception:
            return
        local_X_train = torch.FloatTensor(np.expand_dims(local_X_train, axis=2))
        local_X_valid = torch.FloatTensor(np.expand_dims(local_X_valid, axis=2))
        X_test = torch.FloatTensor(np.expand_dims(X_test, axis=2))

        local_y_train = torch.FloatTensor(local_y_train)

        criterion = optimization.get_criterion(dataset)
        l1_criterion = torch.nn.L1Loss(size_average=False)

        patience = self.patience

        #opt.num_layer = self.num_layer
        #adj_transform, aggregate_function = models.graphLayer.get_transform(opt, adj)

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
        elif self.model == 'LR':
            model = models.models.LogisticRegression(
                    nb_nodes=len(dataset.df.columns),
                    input_dim=1,
                    out_dim=2,
                    on_cuda=self.cuda)

        if self.cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model.cuda()

        #l1_loss_lambda = 0.0001
        #l1_criterion = torch.nn.L1Loss(size_average=False)

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
                crit_loss = optimization.compute_loss(criterion, y_pred, labels)
                #l1_loss = setup_l1_loss(model, l1_loss_lambda, l1_criterion, False)
                total_loss = crit_loss #+ l1_loss

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                crit_loss.backward()
                optimizer.step()
                model.eval()

            auc = {'train': 0., 'valid': 0., 'test': 0.}
            res = []
            try:
                for base_x in range(0,local_X_train.shape[0], self.batch_size):
                    inputs = Variable(local_X_train[base_x:base_x+self.batch_size], requires_grad=False).float()
                    res.append(model(inputs)[:,1].data.cpu().numpy())
                auc['train'] = sklearn.metrics.roc_auc_score(local_y_train.numpy(), np.asarray(res).flatten())

                res = []
                for base_x in range(0,local_X_valid.shape[0], self.batch_size):
                    inputs = Variable(local_X_valid[base_x:base_x+self.batch_size], requires_grad=False).float()
                    res.append(model(inputs)[:,1].data.cpu().numpy())
                auc['valid'] = sklearn.metrics.roc_auc_score(local_y_valid, np.asarray(res).flatten())

                res = []
                for base_x in range(0,X_test.shape[0], self.batch_size):
                    inputs = Variable(X_test[base_x:base_x+self.batch_size], requires_grad=False).float()
                    res.append(model(inputs)[:,1].data.cpu().numpy())
                auc['test'] = sklearn.metrics.roc_auc_score(y_test, np.asarray(res).flatten())
            except Exception:
                pass

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


def method_comparison(results, dataset, models, gene, num_genes, trials, train_size, test_size, file_to_write=None, g=None, first_degree=False, second_degree=False):
    mean = dataset.df[gene].mean()
    dataset.labels = [1 if x > mean else 0 for x in dataset.df[gene]]
    if num_genes != len(dataset.df.columns):
        neighbors = set([gene])
        all_nodes = set(g.nodes)
        try:
            neighbors = neighbors.union(set(g.neighbors(gene)))
            if second_degree:
                for x in g.neighbors(gene):
                    neighbors = neighbors.union(set(g.neighbors(x)))
        except:
            first_degree = all_nodes
            second_degree = all_nodes
        dataset.df = dataset.df[list(neighbors)]

    dataset.df[gene] = 1
    dataset.data = dataset.df.as_matrix()
    neighborhood=None
    for model in models:
        for seed in range(trials):

            #have we already done it?
            already_done = results["df"][(results["df"].gene_name == gene) &
                                         (results["df"].model == model['key']) &
                                         (results["df"].num_genes == num_genes) &
                                         (results["df"].seed == seed) &
                                         (results["df"].train_size == train_size)].shape[0] > 0

            if already_done:
                print "already done:", model['key'], num_genes, seed
                continue
            print "doing:", model['key'], num_genes, seed
            result = model['method'].loop(dataset=dataset, seed=seed, train_size=train_size, test_size=test_size, adj=neighborhood)

            experiment = {"gene_name": gene,
                    "model": model['key'],
                    "num_genes": num_genes,
                    "seed":seed,
                    "train_size": train_size,
                    "auc":result
                    }

            results["df"] = results["df"].append(experiment, ignore_index=True)
            print results
            pickle.dump(results, open(file_to_write, "wb"))

m = [
    {'key': 'MLP', 'method': PyTorch("MLP", dropout=False, cuda=False)},
    #{'key': 'LR', 'method': PyTorch("LR", dropout=False, cuda=False)},
    ]

def build_parser():
    parser = argparse.ArgumentParser(
        description="Meta learning ")

    parser.add_argument('--bucket', default=0, type=int, help='which bucket is this.')
    parser.add_argument('--exp-dir', default=0, type=str, help='which exp dir is this.')
    parser.add_argument('--graph-path', default=0, type=str, help='which exp dir is this.')
    parser.add_argument('--first-degree', default=False, type=bool, help='which exp dir is this.')
    parser.add_argument('--second-degree', default=False, type=str, help='which exp dir is this.')
    return parser


def parse_args(argv):
    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt


def main(argv=None):
    opt = parse_args(argv)

    #tcgatissue = data.gene_datasets.TCGATissue(data_dir='./genomics/TCGA/', data_file='TCGA_tissue_ppi.hdf5')
    tcgatissue = data.gene_datasets.TCGATissue()

    graph = Graph()
    #path = "genomics/graph/pancan-tissue-graph.hdf5"
    graph.load_graph(opt.graph_path)
    #graph.intersection_with(tcgatissue)
    g = nx.from_numpy_matrix(graph.adj)
    mapping = dict(zip(range(0, len(tcgatissue.df.columns)), tcgatissue.df.columns))
    g = nx.relabel_nodes(g, mapping)

    results_file =  opt.exp_dir + '/results-' + str(opt.bucket) + '.pkl'
    try:
        results = pickle.load(open(results_file, "r"))
    except Exception:
        results = {"df": pd.DataFrame(columns=['auc','gene_name', 'model', 'num_genes', 'seed', 'train_size'])}

    #tcgatissue = data.gene_datasets.TCGATissue(data_dir='./genomics/TCGA/', data_file='TCGA_tissue_ppi.hdf5')
    tcgatissue = data.gene_datasets.TCGATissue()
    tcgatissue.df = tcgatissue.df - tcgatissue.df.mean()

    bucket_size = tcgatissue.df.shape[-1] / 100
    start = opt.bucket * bucket_size
    end = (1 + opt.bucket) * bucket_size

    df = tcgatissue.df.copy(deep=True)
    genes_to_iter = tcgatissue.df.iloc[:, start:end].columns.difference(results['df']['gene_name'].unique())
    for gene in genes_to_iter:
#        tcgatissue.df = df[:]
#        method_comparison(results, tcgatissue, m, gene=gene, num_genes=16300, trials=3, train_size=50, test_size=1000, file_to_write=results_file, g=g)
        tcgatissue.df = df[:]
        method_comparison(results, tcgatissue, m, gene=gene, num_genes=50, trials=3, train_size=50, test_size=1000, file_to_write=results_file, g=g, first_degree=opt.first_degree, second_degree=opt.second_degree)

if __name__ == '__main__':
    main()
