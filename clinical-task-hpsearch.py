#!/usr/bin/env python

import sys
sys.path.append('../')

import meta_dataloader.TCGA

import models.mlp, models.gcn
import numpy as np
import data.gene_graphs
import collections
import sklearn.metrics
import sklearn.model_selection
import pandas as pd


import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-task', type=str, default="histological_type")
parser.add_argument('-study', type=str, default="LGG")
parser.add_argument('-graph', type=str, default="stringdb")
args = parser.parse_args()
print(args)

tasks = meta_dataloader.TCGA.TCGAMeta(download=True, 
                                      min_samples_per_class=10, 
                                      gene_symbol_map_file="genenames_code_map_Feb2019.txt")


# for taskid in tasks.task_ids:
#     if "BRCA" in taskid:
#         print(taskid)


# clinical_M  PAM50Call_RNAseq
task = meta_dataloader.TCGA.TCGATask((args.task, args.study), gene_symbol_map_file="genenames_code_map_Feb2019.txt")

graph = data.gene_graphs.StringDBGraph(datastore="./data")

print(task.id)
print(task._samples.shape)
print(np.asarray(task._labels).shape)
print(collections.Counter(task._labels))



X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(task._samples, 
                                                                            task._labels, 
                                                                            stratify=task._labels,
                                                                            train_size=100,
                                                                            test_size=100,
                                                                            shuffle=True,
                                                                            random_state=args.seed
                                                                             )
X_test, X_valid, y_test, y_valid = sklearn.model_selection.train_test_split(X_test, 
                                                                            y_test, 
                                                                            stratify=y_test,
                                                                            train_size=50,
                                                                            test_size=50,
                                                                            shuffle=True,
                                                                            random_state=args.seed
                                                                           )




import skopt, collections
from skopt.space import Real, Integer, Categorical






def doMLP():
    
    
    skopt_args = collections.OrderedDict()
    skopt_args["lr"]=Integer(2, 6)
    skopt_args["channels"]=Integer(6, 12)
    skopt_args["layers"]=Integer(1, 3)

    optimizer = skopt.Optimizer(dimensions=skopt_args.values(),
                                base_estimator="GP",
                                n_initial_points=3,
                                random_state=args.seed)
    print(skopt_args)

    best_valid_metric = 0
    test_for_best_valid_metric = 0
    best_config = None
    already_done = set()
    for i in range(30):
        suggestion = optimizer.ask()
        if str(suggestion) in already_done:
            continue
        already_done.add(str(suggestion))
        sdict = dict(zip(skopt_args.keys(),suggestion))
        sdict["lr"] = 10**float((-sdict["lr"]))
        sdict["channels"] = 2**sdict["channels"]
        print(sdict)

        model = models.mlp.MLP(name="MLP",
                               num_layer=sdict["layers"], 
                               channels=sdict["channels"], 
                               lr=sdict["lr"],
                               num_epochs=100,
                               patience=50,
                               cuda=True,
                               metric=sklearn.metrics.accuracy_score,
                               verbose=False,
                               seed=args.seed)

        model.fit(X_train, y_train)

        y_valid_pred = model.predict(X_valid)
        valid_metric = sklearn.metrics.accuracy_score(y_valid, np.argmax(y_valid_pred,axis=1))

        opt_results = optimizer.tell(suggestion, - valid_metric) 

        #record metrics to write and plot
        if best_valid_metric < valid_metric:
            best_valid_metric = valid_metric
            best_config = sdict

            y_test_pred = model.predict(X_test)
            test_metric = sklearn.metrics.accuracy_score(y_test, np.argmax(y_test_pred,axis=1))
            test_for_best_valid_metric = test_metric

        print(i,"This result:",valid_metric, sdict)
    print("#Final Results", test_for_best_valid_metric, best_config)
    return test_metric, best_config





results_mlp = doMLP()





def doGGC():
    
    if args.graph == "stringdb":
        graph = data.gene_graphs.StringDBGraph(datastore="./data")
    elif args.graph == "genemania":
        graph = data.gene_graphs.GeneManiaGraph()
    else:
        print("unknown graph")
        sys.exit(1)
    adj = graph.adj()



    import gc
    gc.collect()



    skopt_args = collections.OrderedDict()
    skopt_args["lr"]=Integer(3, 5)
    #skopt_args["channels"]=Integer(4, 5)
    #skopt_args["embedding"]=Integer(4, 5)
    skopt_args["num_layer"]=Integer(2, 3)
    skopt_args["prepool_extralayers"]=Integer(1, 2)

    optimizer = skopt.Optimizer(dimensions=skopt_args.values(),
                                base_estimator="GP",
                                n_initial_points=4,
                                random_state=args.seed)
    print(skopt_args)



    best_valid_metric = 0
    test_for_best_valid_metric = 0
    best_config = None
    already_done = set()
    for i in range(10):
        import gc
        gc.collect()

        suggestion = optimizer.ask()
        
        if str(suggestion) in already_done:
            continue
        already_done.add(str(suggestion))
        sdict = dict(zip(skopt_args.keys(),suggestion))
        sdict["lr"] = 10**float((-sdict["lr"]))
        sdict["channels"] = 32#2**sdict["channels"]
        sdict["embedding"] = 32#2**sdict["embedding"]
        print(sdict)

        model = models.gcn.GCN(name="GCN_lay3_chan64_emb32_dropout_agg_hierarchy", 
                               dropout=False, 
                               cuda=True,
                               num_layer=sdict["num_layer"],
                               prepool_extralayers=sdict["prepool_extralayers"],
                               channels=sdict["channels"], 
                               embedding=sdict["embedding"], 
                               aggregation="hierarchy",
                               lr=sdict["lr"],
                               num_epochs=100,
                               patience=20,
                               verbose=True,
                               seed=args.seed
                              )
        model.fit(X_train, y_train, adj)

        y_valid_pred = model.predict(X_valid)
        valid_metric = sklearn.metrics.accuracy_score(y_valid, np.argmax(y_valid_pred,axis=1))

        opt_results = optimizer.tell(suggestion, - valid_metric) 

        #record metrics to write and plot
        if best_valid_metric < valid_metric:
            best_valid_metric = valid_metric
            print("best_valid_metric", best_valid_metric, sdict)
            best_config = sdict

            y_test_pred = model.predict(X_test)
            test_metric = sklearn.metrics.accuracy_score(y_test, np.argmax(y_test_pred,axis=1))
            test_for_best_valid_metric = test_metric

        print(i,"This result:",valid_metric, sdict)
    print("#Final Results", test_for_best_valid_metric, best_config)
    return test_for_best_valid_metric, best_config


results_ggc = doGGC()


print("####GGC", args,results_ggc)
print("####MLP", args,results_mlp)
