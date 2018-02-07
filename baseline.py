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
                   "epoch": 300,
                   "batch_size": 100,
                   "train_ratio": .6,
                   "param_to_vary": "extra_ucn",
                   "param_min_val": 0,
                   "param_max_val": 100,
                   "param_step": 10,
                   "model_map": {
                    "lr_no_l1": "mlp",
                    "lr_with_l1": "mlp",
                    "slr_no_l1": "slr",
                    "slr_with_l1": "slr",
                    "mlp": "mlp",
                    "cnn": "cnn",
                    "cgn_pool": "cgn",
                    "cgn_no_pool": "cgn",
                    "sgc": "sgc",
                    "lcg": "lcg",
                   },
                   "model_vars":{
                        'lr_no_l1': [('num_layer', 0)],
                        'lr_with_l1': [('num_layer', 0), ('l1_loss_lambda', 0.5)],
                        'slr_no_l1': [('lambdas', 0.25)],
                        'slr_with_l1': [('lambdas', 0.25), ('l1_loss_lambda', 0.25)],
                        'mlp': [('num_layer', 3), ('num_channel', 64), ("use_emb", 32), ("l1_loss_lambda", .1)],
                        'cnn': [],
                        'cgn_pool': [('num_layer', 3), ('num_channel', 32), ("use_emb", 32), ('pool_graph', 'hierarchy')],
                        'cgn_no_pool': [('num_layer', 3), ('num_channel', 32), ("use_emb", 32)],
                        'sgc': [('num_layer', 3), ('num_channel', 32), ("use_emb", 32)],
                        'lcg': [('num_layer', 3), ('num_channel', 32), ("use_emb", 32)],
                        }
                   }

def build_parser():
    parser = conv_graph.build_parser()
    parser.add_argument('--baseline-dataset', help="The type of baseline tests to launch", required=True)
    parser.add_argument('--baseline-model', help="The type of baseline tests to launch", required=True)
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
    setting['model'] = static_settings['model_map'][setting['baseline_model']]
    model_config_name = setting['baseline_model']
    del setting['baseline_model']
    del setting['baseline_dataset']
    cols = ['model', 'dataset', 'total_loss', 'cross_loss', 'auc_train', 'auc_valid', 'auc_test', 'seed', 'epoch', 'num_trials', 'valid_acc_mean', 'valid_acc_std', 'test_acc_mean', 'test_acc_std',  'valid_auc_mean', 'valid_auc_std', 'test_auc_mean', 'test_auc_std']
    cols.append(static_settings['param_to_vary'])
    for hyper_param_name, value in static_settings['model_vars'][model_config_name]:
        cols.append(hyper_param_name)
        setting[hyper_param_name] = value
    to_print = pd.DataFrame()
    for param_value in range(static_settings['param_min_val'], static_settings['param_max_val'], static_settings['param_step']):
        summaries = pd.DataFrame()
        setting[static_settings['param_to_vary']] = param_value
        for seed in range(0, static_settings['num_trials']):
            setting['seed'] = seed
            summary = format_summary(conv_graph.main(opt), setting, model_config_name)
            summaries = summaries.append(pd.DataFrame.from_dict(summary), ignore_index=True)
        to_print = log_summary(summaries, setting, cols, to_print, static_settings['param_to_vary'], param_value, model_config_name)
        print to_print.to_csv(index=False)


def format_summary(summary, setting, model_config_name):
    for hyper_param_name, value in static_settings['model_vars'][model_config_name]:
        setting[hyper_param_name] = value
    summary['acc_train'] = [summary['accuracy']['train']]
    summary['acc_test'] = [summary['accuracy']['test']]
    summary['acc_valid'] = [summary['accuracy']['valid']]
    summary['auc_train'] = [summary['auc']['train']]
    summary['auc_test'] = [summary['auc']['test']]
    summary['auc_valid'] = [summary['auc']['valid']]
    summary['seed'] = [setting['seed']]
    summary['model'] = [model_config_name]
    summary['dataset'] = [setting['dataset']]
    del summary['accuracy']
    del summary['auc']
    return summary


def log_summary(summaries, setting, cols, to_print, param_to_vary, param_value, model_config_name):
    experiment = summaries[summaries['dataset']==setting['dataset']][summaries['model']==model_config_name].reset_index()
    pd.options.mode.chained_assignment = None  # default='warn'
    max_experiment = experiment.iloc[experiment['auc_valid'].idxmax()]
    max_experiment.loc['valid_acc_mean'] = experiment['acc_valid'].mean()
    max_experiment.loc['valid_acc_std'] = experiment['acc_valid'].std()
    max_experiment.loc['test_acc_mean'] = experiment['acc_test'].mean()
    max_experiment.loc['test_acc_std'] = experiment['acc_test'].std()
    max_experiment.loc['valid_auc_mean'] = experiment['auc_valid'].mean()
    max_experiment.loc['valid_auc_std'] = experiment['auc_valid'].std()
    max_experiment.loc['test_auc_mean'] = experiment['auc_test'].mean()
    max_experiment.loc['test_auc_std'] = experiment['auc_test'].std()
    max_experiment.loc['num_trials'] = experiment.shape[0]
    for hyper_param_name, value in static_settings['model_vars'][model_config_name]:
        max_experiment.loc[hyper_param_name] = value
    max_experiment.loc[param_to_vary] = param_value
    to_print = to_print.append(pd.DataFrame(max_experiment).T[cols])
    return to_print


if __name__ == "__main__":
    main()
