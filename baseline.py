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
    static_settings = json.load(open("settings.json"))

    setting['epoch'] = static_settings['epoch']
    setting['batch_size'] = static_settings['batch_size']
    setting['train_ratio'] = static_settings['train_ratio']
    setting['dataset'] = setting['baseline_dataset']
    setting['model'] = static_settings['model_map'][setting['baseline_model']]
    model_config_name = setting['baseline_model']
    del setting['baseline_model']
    del setting['baseline_dataset']
    cols = ['model', 'dataset', 'total_loss', 'cross_loss', 'auc_train', 'auc_valid', 'auc_test', 'seed', 'epoch', 'num_trials', 'valid_acc_mean', 'valid_acc_std', 'test_acc_mean', 'test_acc_std',  'valid_auc_mean', 'valid_auc_std', 'test_auc_mean', 'test_auc_std']
    cols.append(static_settings['param_to_vary']['name'])
    for hyper_param_name, value in static_settings['model_vars'][model_config_name].iteritems():
        cols.append(hyper_param_name)
        setting[hyper_param_name] = value
    for key, value in static_settings.get('dataset_vars', {}).iteritems():
        cols.append(key)
        setting[key] = value
    to_print = pd.DataFrame()
    for param_value in range(static_settings['param_to_vary']['min'], static_settings['param_to_vary']['max'], static_settings['param_to_vary']['step']):
        summaries = pd.DataFrame()
        setting[static_settings['param_to_vary']['name']] = param_value
        for seed in range(0, static_settings['num_trials']):
            setting['seed'] = seed
            summary = format_summary(conv_graph.main(opt), setting, model_config_name, static_settings)
            summaries = summaries.append(pd.DataFrame.from_dict(summary), ignore_index=True)
        to_print = log_summary(summaries, setting, cols, to_print, static_settings['param_to_vary']['name'], param_value, model_config_name, static_settings)
        print to_print.to_csv(index=False)


def format_summary(summary, setting, model_config_name, static_settings):
    for hyper_param_name, value in static_settings['model_vars'][model_config_name].iteritems():
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


def log_summary(summaries, setting, cols, to_print, param_to_vary, param_value, model_config_name, static_settings):
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
    for hyper_param_name, value in static_settings['model_vars'][model_config_name].iteritems():
        max_experiment.loc[hyper_param_name] = value
    for key, value in static_settings.get('dataset_vars', {}).iteritems():
        max_experiment.loc[key] = value
    max_experiment.loc[param_to_vary] = param_value
    to_print = to_print.append(pd.DataFrame(max_experiment).T[cols])
    return to_print


if __name__ == "__main__":
    main()
