import main as conv_graph
import argparse
import json
import numpy as np
import pandas as pd
import os
import copy
import logging
from collections import defaultdict
import itertools

# I chose to hardcode the parameters so we can all agree on a baseline and share a record of it.
# Use "default" mode as our shared baseline -- these settings shouldn't really be changed.
# Use "test" mode  to ensure that all the models are working, it will be quick (a minute or two)
# Use "freeplay" mode to mess around with the parameters and try to improve our settings.
default = {"num_experiments": 10,
           "models": ['mlp', 'sgc', 'slr', 'cgn', 'lcg'],
           "datasets": ['random', 'percolate', 'tcga-gbm'],
           "vars_to_explore": [('lr', (1e-5, 1e-3))],
           "epoch": 10,
           "batch_size": 10
           }
test = {"num_experiments": 1,
        "models": ['mlp', 'sgc', 'slr', 'cgn', 'lcg'],
        "datasets": ['random', 'percolate', 'tcga-gbm'],
        "vars_to_explore": [('lr', (1e-5, 1e-3))],
        "epoch": 1,
        "batch_size": 10
        }
freeplay = {"num_experiments": 3,
            "models": ['mlp', 'cgn'],
            "datasets": ['tcga-gbm'],
            "vars_to_explore":[('lr', (1e-5, 1e-3))],
            "epoch": 2,
            "batch_size": 100}


def build_parser():
    parser = conv_graph.build_parser()
    parser.add_argument('--mode', default="default", help="The type of baseline tests to launch", choices=['default', 'test', 'freeplay'])
    return parser

def parse_args(argv):
    opt = build_parser().parse_args(argv)
    return opt

def main(argv=None):
    # Setup logger
    hdlr = logging.FileHandler('baseline.log')
    hdlr.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger('baseline')
    logger.addHandler(hdlr)
    logger.setLevel('INFO')

    opt = parse_args(argv)
    mode = globals()[opt.mode]
    setting = vars(opt)
    setting['epoch'] = mode['epoch']
    setting['batch_size'] = mode['batch_size']
    baseline_mode = setting['mode']
    del setting['mode']
    summaries = pd.DataFrame()

    for model, dataset, var_to_explore in itertools.product(mode['models'], mode['datasets'], mode['vars_to_explore']):
        setting['model'] = model
        setting['dataset'] = dataset
        set_num_channel(model, setting)
        sample_and_set_hyper_param(var_to_explore, setting)

        for seed in range(0, mode['num_experiments']):
            setting['seed'] = seed
            summary = conv_graph.main(opt)
            summaries = record_summary(var_to_explore[0], setting[var_to_explore[0]], seed, model, dataset, summary, summaries)

    if baseline_mode != 'test':
        summaries = log_summary(summaries, mode)

def record_summary(hyper_var, hyper_val, seed, model, dataset, summary, summaries):
    summary['hyper-parameter-name'] = [hyper_var]
    summary['hyper-parameter-value'] = [hyper_val]
    summary['train'] = [summary['accuracy']['train']]
    summary['test'] = [summary['accuracy']['tests']]
    summary['valid'] = [summary['accuracy']['valid']]
    summary['seed'] = [seed]
    summary['model'] = [model]
    summary['dataset'] = [dataset]
    del summary['accuracy']

    if summaries.empty:
        summaries = pd.DataFrame.from_dict(summary)
    else:
        summaries = summaries.append(pd.DataFrame.from_dict(summary), ignore_index=True)
    return summaries


def log_summary(summaries, mode):
    logger = logging.getLogger('baseline')
    to_print = pd.DataFrame()

    for model, dataset, var_to_explore in itertools.product(mode['models'], mode['datasets'], mode['vars_to_explore']):
        experiment = summaries[summaries['dataset'].isin([dataset])][summaries['model'].isin([model])][summaries['hyper-parameter-name'].isin([var_to_explore[0]])].reset_index()
        max_experiment = experiment.iloc[experiment['valid'].idxmax()]
        max_experiment.loc['valid_set_mean'] = experiment['valid'].mean()
        max_experiment.loc['valid_set_variance'] = experiment['valid'].var()
        to_print = to_print.append(pd.DataFrame(max_experiment).T)
    cols = ['model', 'dataset', 'hyper-parameter-name', 'hyper-parameter-value', 'train', 'valid', 'test', 'valid_set_mean', 'valid_set_variance', 'total_loss', 'cross_loss', 'seed', 'epoch']
    logger.info(to_print[cols].to_string())


def sample_and_set_hyper_param(var_to_explore, setting):
    variable, bound = var_to_explore
    min_value, max_value = bound
    if min_value > max_value:
        raise ValueError("The minimum value is bigger than the maxium value for {}, {}".format(value, bound))

    # sampling the value
    if type(min_value) == int and type(max_value) == int:
        value = min_value + np.random.random_sample() * (max_value - min_value)
        value = int(np.round(value))
    else:
        value = np.exp(np.log(min_value) + (np.log(max_value) - np.log(min_value)) * np.random.random_sample())
        value = float(value)

    if variable not in setting:
        raise ValueError("The parameter {} is nor defined.".format(variable))
    setting[variable] = value

    return value


def set_num_channel(model, setting):
    num_channel = setting['num_channel']
    if model == 'sgc':
        setting['num_channel'] = 1
    else:
        setting['num_channel'] = num_channel

if __name__ == "__main__":
    main()
