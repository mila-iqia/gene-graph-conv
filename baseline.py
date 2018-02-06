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

static_settings = {
                   "num_trials": 1,
                   "epoch": 10,
                   "batch_size": 100,
                   "train_ratio": .6,
                   "model_vars":{
                        'slr': [('lr', .001), ('num_channel', 32), ('weight_decay', 0.0), ('l1-loss-lambda', 0.0), ('lambdas', 0.25), ('num_layer', 1)],
                        'mlp': [('lr', .001), ('num_channel', 32), ('weight_decay', 0.0), ('l1-loss-lambda', 0.0), ('lambdas', 0.0), ('num_layer', 1)],
                        'cgn': [('lr', .015), ('num_channel', 128), ('weight_decay', 0.1), ('l1-loss-lambda', 0.5), ('lambdas', 0.0), ('num_layer', 1)]
                        }
                   }

def build_parser():
    parser = conv_graph.build_parser()
    parser.add_argument('--baseline-dataset', help="The type of baseline tests to launch", choices=['random', 'tcga-tissue', 'tcga-brca', 'tcga-label', 'tcga-gbm', 'percolate', 'percolate-plus', 'nslr-syn'], required=True)
    parser.add_argument('--baseline-model', help="The type of baseline tests to launch", choices=['cgn', 'mlp', 'lcg', 'sgc', 'slr', 'cnn', 'random'], required=True)
    parser.add_argument('--vary-param-name', help="A parameter to vary, if you want", type=str)
    parser.add_argument('--vary-param-value', help="A parameter to vary, if you want", type=int)
    return parser

def parse_args(argv):
    opt = build_parser().parse_args(argv)
    return opt

def main(argv=None):
    opt = parse_args(argv)
    setting = vars(opt)
    setting['epoch'] = static_settings['epoch']
    setting['batch_size'] = static_settings['batch_size']
    setting['train_ratio'] = static_settings['train_ratio']
    setting['dataset'] = setting['baseline_dataset']
    setting['model'] = setting['baseline_model']
    param_to_vary = setting['vary_param_name']
    param_value = setting['vary_param_value']
    del setting['vary_param_name']
    del setting['vary_param_value']
    del setting['baseline_model']
    del setting['baseline_dataset']
    cols = ['model', 'dataset', 'train', 'valid', 'test', 'total_loss', 'cross_loss', 'seed', 'epoch', 'num_trials', 'valid_mean', 'valid_std', 'train_mean', 'train_std']
    for hyper_param_name, value in static_settings['model_vars'][setting['model']]:
        cols.append(hyper_param_name)
        setting[hyper_param_name] = value
    setting[param_to_vary] = param_value
    cols.append(param_to_vary)

    to_print = pd.DataFrame()
    summaries = pd.DataFrame()

    for seed in range(0, static_settings['num_trials']):
        setting['seed'] = seed
        summary = format_summary(conv_graph.main(opt), setting)
        summaries = summaries.append(pd.DataFrame.from_dict(summary), ignore_index=True)
    to_print = log_summary(summaries, setting, cols, to_print, param_to_vary, param_value)
    print to_print.to_string(index=False)


def format_summary(summary, setting):
    for hyper_param_name, value in static_settings['model_vars'][setting['model']]:
        setting[hyper_param_name] = value

    summary['train'] = [summary['accuracy']['train']]
    summary['test'] = [summary['accuracy']['tests']]
    summary['valid'] = [summary['accuracy']['valid']]
    summary['seed'] = [setting['seed']]
    summary['model'] = [setting['model']]
    summary['dataset'] = [setting['dataset']]
    del summary['accuracy']
    return summary


def log_summary(summaries, setting, cols, to_print, param_to_vary, param_value):
    experiment = summaries[summaries['dataset']==setting['dataset']][summaries['model']==setting['model']].reset_index()
    pd.options.mode.chained_assignment = None  # default='warn'
    max_experiment = experiment.iloc[experiment['valid'].idxmax()]
    max_experiment.loc['valid_mean'] = experiment['valid'].mean()
    max_experiment.loc['valid_std'] = experiment['valid'].std()
    max_experiment.loc['train_mean'] = experiment['train'].mean()
    max_experiment.loc['train_std'] = experiment['train'].std()
    max_experiment.loc['num_trials'] = experiment.shape[0]
    for hyper_param_name, value in static_settings['model_vars'][setting['model']]:
        max_experiment.loc[hyper_param_name] = value
    max_experiment.loc[param_to_vary] = param_value
    to_print = to_print.append(pd.DataFrame(max_experiment).T[cols])
    return to_print


if __name__ == "__main__":
    main()
