import main as conv_graph
import argparse
import json
import numpy as np
import os

# I chose to hardcode the parameters so we can all agree on a baseline and share a record of it.
# Use "default" mode as our shared baseline -- these settings shouldn't really be changed.
# Use "test" mode  to ensure that all the models are working, it will be quick (a minute or two)
# Use "freeplay" mode to mess around with the parameters and try to improve our settings.

default = {"num_experiments": 10,
           "models": ['mlp', 'sgc', 'slr', 'cgn', 'lcg'],
           "vars_to_explore": [('lr', (1e-5, 1e-3))],
           "epoch": 10
           }
test = {"num_experiments": 1,
        "models": ['mlp', 'sgc', 'slr', 'cgn', 'lcg'],
        "vars_to_explore": [('lr', (1e-5, 1e-3))],
        "epoch": 1
        }
freeplay = {"num_experiments": 10,
            "models": ['mlp', 'sgc', 'slr', 'cgn', 'lcg'],
            "vars_to_explore":[('lr', (1e-5, 1e-3))],
            "epoch": 10}


def build_parser():
    parser = conv_graph.build_parser()
    parser.add_argument('--mode', default="default", help="The type of baseline tests to launch", choices=['default', 'test', 'freeplay'])
    return parser

def parse_args(argv):
    opt = build_parser().parse_args(argv)
    return opt

def main(argv=None):
    opt = parse_args(argv)
    mode = globals()[opt.mode]
    setting = vars(opt)
    setting['epoch'] = mode['epoch']
    del setting['mode']

    for model in mode['models']:
        setting['model'] = model
        for seed in range(0, mode['num_experiments']):
            setting['seed'] = seed
            for variable, bound in mode['vars_to_explore']:
                min_value, max_value = bound
                if min_value > max_value:
                    raise ValueError("The minimum value is bigger than the maxium value for {}, {}".format(value, bound))
                set_num_channel(model, setting)

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
                #launch an experiment:
                print "Will launch the experiment with the following hyper-parameters: {}".format(setting)
                conv_graph.main(opt)

def set_num_channel(model, setting):
    num_channel = setting['num_channel']
    if model == 'sgc':
        setting['num_channel'] = 1
    else:
        setting['num_channel'] = num_channel

if __name__ == "__main__":
    main()
