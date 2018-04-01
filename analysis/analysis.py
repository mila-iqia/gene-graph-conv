import pickle
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def filter_exps(experiments):
    accumulators = []
    for exp in experiments.values():
        if exp.values()[0].get('event_accumulator') is not None:
            accumulators.append(exp)
    return accumulators


def find_best_epoch(events):
    # Takes a tensorflow event_accuulator object as input and returns the best epoch/auc_valid value
    x = np.array([e.value for e in events])
    best_epoch = x.argmax(axis=0)
    best_value = x[best_epoch]
    return best_epoch, best_value


def get_all_tf_metadata(ex_dir='../experiments/experiments', filtered=True):
    # format: {'path_to_exp': {trial#: {'experiment_hash': hash_str, opts: {}, 'event_accumulator': tf_event_accumulator }}}
    all_experiments = {}
    for root, dirs, files in os.walk(ex_dir):
        experiment = defaultdict(dict)
        for f in files:
            key = '/'.join(root.split('/')[1:-1])
            trial = root.split('/')[-1]
            if f == "options.pkl":
                opts = vars(pickle.load(open(os.path.join(root, f), 'rb')))
                experiment = all_experiments.get(key, defaultdict(dict))
                experiment[trial]['opts'] = opts
                all_experiments[key] = experiment
            elif f.startswith("event"):
                experiment = all_experiments.get(key, defaultdict(dict))
                experiment[trial]['event_accumulator'] = EventAccumulator(os.path.join(root, f))
                experiment[trial]['hash'] = root.split('/')[-2]
                experiment[trial]['path'] = '/'.join(root.split('/')[:-1])
                all_experiments[key] = experiment
    if filtered:
        all_experiments = filter_exps(all_experiments)
    return all_experiments


def accumulator_to_df(event_accumulator,
                      experiment_opts=None,
                      e_hash='N/A',
                      e_path='N/A',
                      tags=['auc_train', 'auc_valid', 'auc_test', 'acc_train', 'acc_valid', 'acc_test'],
                      opt_tags=['weight_decay', 'epoch', 'lr', 'train_ratio', 'dataset', 'dropout', 'batch_size', 'pool_graph', 'use_gate', 'num_channel', 'add_self', 'l1_loss_lambda', 'num_layer', 'graph', 'model', 'nb_nodes']):

    df = pd.DataFrame(columns=tags + opt_tags)

    # Get the epoch with the best AUC for the valid set
    best_epoch, value = find_best_epoch(event_accumulator.Scalars("auc_valid"))
    row_number = len(df.index)
    df.loc[row_number, "auc_valid"] = value
    df.loc[row_number, "best_epoch"] = best_epoch
    df.loc[row_number, "hash"] = e_hash
    df.loc[row_number, "path"] = e_path

    # set the other values from that epoch
    for tag in tags:
        df.loc[row_number, tag] = event_accumulator.Scalars(tag)[best_epoch].value
        df[tag] = df[tag].astype(float)
    for tag in opt_tags:
        df.loc[row_number, tag] = experiment_opts[tag]
    return df


def metadata_to_experiments(metadata):
    df = pd.DataFrame()
    for experiment in metadata:
        for key, trial in experiment.iteritems():
            event_accumulator = trial['event_accumulator'].Reload()
            temp_df = accumulator_to_df(event_accumulator, trial['opts'], trial['hash'], trial['path'])
            df = df.append(temp_df)
    return df
