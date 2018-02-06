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
                   "num_trials": 3,
                   "num_experiments": 50,
                   "epoch": 100,
                   "batch_size": 100,
                   "train_ratio": .6,
                   "vars_to_explore":{
                        'slr': [('lr', .001, .1), ('num_channel', 16, 256), ('num_layer', 0, 0)],
                        'mlp': [('lr', .001, .1), ('num_channel', 32, 256), ('weight_decay', 0.0, 0.5), ('l1-loss-lambda', 0.00, 0.6), ('lambdas', 0.0, 1e-7), ('num_layer', 0, 1)],
                        'cgn': [('lr', .01, .02), ('num_channel', 64, 200), ('weight_decay', 0.05, 0.15), ('l1-loss-lambda', 0.4, 0.6)]
                        }
                   }

def build_parser():
    parser = conv_graph.build_parser()
    parser.add_argument('--search-dataset', help="The type of dataset to search for good hyper-parameters", choices=['random', 'tcga-tissue', 'tcga-brca', 'tcga-label', 'tcga-gbm', 'percolate', 'nslr-syn'], required=True)
    parser.add_argument('--search-model', help="The type of model to search for good hyper-parameters", choices=['cgn', 'mlp', 'lcg', 'sgc', 'slr', 'cnn'], required=True)
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
    setting['dataset'] = setting['search_dataset']
    setting['model'] = setting['search_model']
    del setting['search_model']
    del setting['search_dataset']

    # Model-specific settings, always add self and norm adjacency for CGN
    if setting['model'] == 'cgn':
        setting['add_self'] = ""
        setting['norm_adj'] = ""

    # columns to print out
    cols = ['model', 'dataset', 'train', 'valid', 'test', 'total_loss', 'cross_loss', 'seed', 'epoch', 'num_trials', 'valid_mean', 'valid_std', 'train_mean', 'train_std']
    for hyper_param, min_value, max_value in static_settings['vars_to_explore'][setting['model']]:
        cols.append(hyper_param)
    to_print = pd.DataFrame()
    best_valid_mean = 0
    best_setting = {}
    best_results = {}

    # Search the parameter-space num_experiments times
    for x in range(0, static_settings['num_experiments']):
        summaries = pd.DataFrame()

        # sample the hyperparameters and add them to settings
        for hyper_param in static_settings['vars_to_explore'][setting['model']]:
            hyper_param_name, min_value, max_value = hyper_param
            setting[hyper_param_name] = sample_value(min_value, max_value)

        # run a number of trials with these parameters
        for seed in range(0, static_settings['num_trials']):
            setting['seed'] = seed
            summary = format_summary(conv_graph.main(opt), setting)
            summaries = summaries.append(pd.DataFrame.from_dict(summary), ignore_index=True)
        to_print = log_summary(summaries, setting, cols, to_print)
        print to_print.to_string(index=False)

        # Save the best setting to print at the end
        if 'valid_mean' in to_print.columns and to_print['valid_mean'].max() > best_valid_mean:
            best_valid_mean = to_print['valid_mean'].max()
            best_setting = setting
            best_results = pd.DataFrame.from_dict(summary)

    print "\n"
    print "--------------"
    print "Best Results:"
    print log_summary(best_results, best_setting, cols, to_print).tail(1).to_string(index=False)


def format_summary(summary, setting):
    # Parses the return value from the conv_graph's main.py
    for hyper_param, min_value, max_value in static_settings['vars_to_explore'][setting['model']]:
        summary[hyper_param] = setting[hyper_param]
    summary['train'] = [summary['accuracy']['train']]
    summary['test'] = [summary['accuracy']['tests']]
    summary['valid'] = [summary['accuracy']['valid']]
    summary['seed'] = [setting['seed']]
    summary['model'] = [setting['model']]
    summary['dataset'] = [setting['dataset']]
    del summary['accuracy']
    return summary


def sample_value(min_value, max_value):
    if type(min_value) == int and type(max_value) == int:
        value = min_value + np.random.random_sample() * (max_value - min_value)
        value = int(np.round(value))
    else:
        value = np.exp(np.log(min_value) + (np.log(max_value) - np.log(min_value)) * np.random.random_sample())
        value = float(value)
    return value


def log_summary(summaries, setting, cols, to_print):
    experiment = summaries[summaries['dataset']==setting['dataset']][summaries['model']==setting['model']].reset_index()
    pd.options.mode.chained_assignment = None  # default='warn'
    max_experiment = experiment.iloc[experiment['valid'].idxmax()]
    max_experiment.loc['valid_mean'] = experiment['valid'].mean()
    max_experiment.loc['valid_std'] = experiment['valid'].std()
    max_experiment.loc['train_mean'] = experiment['train'].mean()
    max_experiment.loc['train_std'] = experiment['train'].std()
    max_experiment.loc['num_trials'] = experiment.shape[0]
    for hyper_param, min_value, max_value in static_settings['vars_to_explore'][setting['model']]:
        max_experiment.loc[hyper_param] = setting[hyper_param]

    to_print = to_print.append(pd.DataFrame(max_experiment).T[cols])
    return to_print


if __name__ == "__main__":
    main()
