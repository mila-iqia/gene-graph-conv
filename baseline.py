import main as conv_graph
import argparse
import json
import numpy as np
import pandas as pd
import os
import copy
import logging
import sys
from collections import defaultdict
from itertools import product, combinations

# I chose to hardcode the parameters so we can all agree on a baseline and share a record of it.
# Use "default" mode as our shared baseline -- these settings shouldn't really be changed.
# Use "test" mode  to ensure that all the models are working, it will be quick (a minute or two)
# Use "freeplay" mode to mess around with the parameters and try to improve our settings.
default = {"num_experiments": 5,
           "models": ['mlp', 'slr', 'cgn', 'lcg'],
           "datasets": ['random', 'percolate', 'tcga-gbm'],
           "grid_width": 5,
           "vars_to_explore":{
                'slr': [('lr', 1e-2, .2, float), ('weight_decay', 0.0, 0.5, float), ('lambdas', 0.0, .5, float)],
                'mlp': [('lr', .06, .1, float), ('weight_decay', 0.3, 0.5, float), ('num_layer', 1, 3, int)],
                'cgn': [('lr', .02, .1, float), ('weight_decay', 0.2, 0.5, float), ('num_channel', 16, 128, int), ('num_layer', 1, 3, int)]
                },
           "epoch": 100,
           "batch_size": 100,
           "train_ratio": .6,
           }
test = {"num_experiments": 5,
        "models": ['mlp', 'cgn'],
        "datasets": ['tcga-gbm'],
        "grid_width": 3,
        "vars_to_explore":[('weight_decay', 0.0, 0.1, float), ('lr', 1e-7, 1e-3, float), ('num_channel', 1, 64, int)],
        "epoch": 100,
        "batch_size": 10,
        "train_ratio": .666,
        }
freeplay = {"num_experiments": 50,
           "models": ['mlp', 'slr', 'cgn', 'lcg'],
           "datasets": ['random', 'percolate', 'tcga-gbm'],
           "grid_width": 1,
           "vars_to_explore":{
                'slr': [('lr', .048, .2, float), ('weight_decay', 0.3, 0.5, float), ('lambdas', 0.0, .5, float)],
                'mlp': [('lr', .06, .1, float), ('weight_decay', 0.3, 0.5, float), ('num_layer', 1, 3, int)],
                'cgn': [('lr', .02, .1, float), ('weight_decay', 0.2, 0.5, float), ('num_channel', 16, 128, int), ('num_layer', 1, 3, int)]
                },
           "epoch": 100,
           "batch_size": 100,
           "train_ratio": .6,
           }


def build_parser():
    parser = conv_graph.build_parser()
    parser.add_argument('--mode', default="default", help="The type of baseline tests to launch", choices=['default', 'test', 'freeplay'])
    parser.add_argument('--baseline-dataset', help="The type of baseline tests to launch", choices=['random', 'tcga-tissue', 'tcga-brca', 'tcga-label', 'tcga-gbm', 'percolate', 'nslr-syn'])
    parser.add_argument('--baseline-model', help="The type of baseline tests to launch", choices=['cgn', 'mlp', 'lcg', 'sgc', 'slr', 'cnn'])
    return parser

def parse_args(argv):
    opt = build_parser().parse_args(argv)
    return opt

def main(argv=None):
    opt = parse_args(argv)
    mode = globals()[opt.mode]
    setting = vars(opt)

    setting['epoch'] = mode['epoch']
    setting['batch_size'] = mode['batch_size']
    baseline_mode = setting['mode']
    if setting.get('baseline_dataset'):
        mode['datasets'] = [setting['baseline_dataset']]
    if setting.get('baseline_model'):
        mode['models'] = [setting['baseline_model']]
    del setting['mode']
    del setting['baseline_model']
    del setting['baseline_dataset']


    cols = ['model', 'dataset', 'param-1', 'val1', 'param-2', 'val2', 'train', 'valid', 'test', 'total_loss', 'cross_loss', 'seed', 'epoch', 'num_trials', 'valid_mean', 'valid_std', 'train_mean', 'train_std']
    to_print = pd.DataFrame()
    for model, dataset in product(mode['models'], mode['datasets']):
        setting['model'] = model
        setting['dataset'] = dataset
        set_num_channel(model, setting)

        for hyperparam1, hyperparam2 in combinations(mode['vars_to_explore'][model], 2):
            hyper_param_name_1, min_value_1, max_value_1, param_type1 = hyperparam1
            hyper_param_name_2, min_value_2, max_value_2, param_type2 = hyperparam2
            for val1 in np.arange(min_value_1, max_value_1, float(max_value_1 - min_value_1)/mode['grid_width']):
                setting[hyper_param_name_1] = param_type1(val1)
                for val2 in np.arange(min_value_2, max_value_2, float(max_value_2 - min_value_2)/mode['grid_width']):
                    setting[hyper_param_name_2] = param_type2(val2)
                    summaries = pd.DataFrame()
                    for seed in range(0, mode['num_experiments']):
                        setting['seed'] = seed
                        summary = format_summary(conv_graph.main(opt), hyper_param_name_1, hyper_param_name_2, param_type1(val1), param_type2(val2), seed, model, dataset)
                        summaries = summaries.append(pd.DataFrame.from_dict(summary), ignore_index=True)
                    to_print = log_summary(summaries, mode, model, dataset, hyper_param_name_1, param_type1(val1), hyper_param_name_2, param_type2(val2), cols, to_print)
                    print to_print.to_string(index=False)
    print to_print.to_string(index=False)


def format_summary(summary, hyper_var_1, hyper_var_2, hyper_val_1, hyper_val_2, seed, model, dataset):
    summary['param-1'] = [hyper_var_1]
    summary['val1'] = [hyper_val_1]
    summary['param-2'] = [hyper_var_2]
    summary['val2'] = [hyper_val_2]
    summary['train'] = [summary['accuracy']['train']]
    summary['test'] = [summary['accuracy']['tests']]
    summary['valid'] = [summary['accuracy']['valid']]
    summary['seed'] = [seed]
    summary['model'] = [model]
    summary['dataset'] = [dataset]
    del summary['accuracy']
    return summary


def log_summary(summaries, mode, model, dataset, hyper_1, val1, hyper_2, val2, cols, to_print):
    experiment = summaries[summaries['dataset']==dataset][summaries['model']==model][summaries['param-1']==hyper_1][summaries['param-2']==hyper_2][summaries['val1']==val1][summaries['val2']==val2].reset_index()
    max_experiment = experiment.iloc[experiment['valid'].idxmax()]
    max_experiment.loc['valid_mean'] = experiment['valid'].mean()
    max_experiment.loc['valid_std'] = experiment['valid'].std()
    max_experiment.loc['train_mean'] = experiment['train'].mean()
    max_experiment.loc['train_std'] = experiment['train'].std()
    max_experiment.loc['num_trials'] = experiment.shape[0]
    to_print = to_print.append(pd.DataFrame(max_experiment).T[cols])
    return to_print

def set_num_channel(model, setting):
    num_channel = setting['num_channel']
    if model == 'sgc':
        setting['num_channel'] = 1
    else:
        setting['num_channel'] = num_channel

if __name__ == "__main__":
    main()
