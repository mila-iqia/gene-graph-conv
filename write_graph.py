import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args(argv):
    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv
    return opt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='The directory with the slurm files.')
    parser.add_argument('-v', type=str, help='The name of the column that we are varying.')
    opt = parser.parse_args(argv)

    files = [os.path.join(opt.d, f) for f in os.listdir(opt.d) if os.path.isfile(os.path.join(opt.d, f))]
    for f in files:
        df = pd.read_csv(f)

        df.test_auc_mean.name = df.model[0]
        plt.errorbar(df[opt.v], df.test_auc_mean, df.test_auc_std)

    if opt.v == "extra_ucn":
        plt.xlabel('# Additional Unconnected Nodes')
        plt.ylabel('Mean AUROC')
        plt.title("Percolate Task with Extra Unconnected Nodes")
    elif opt.v == "extra_cn":
        plt.xlabel('# Additional Connected Nodes')
        plt.ylabel('Mean AUROC')
        plt.title("Percolate Task with Extra Connected Nodes")
    elif opt.v == "disconnected":
        plt.xlabel('# Disconnected Nodes')
        plt.ylabel('Mean AUROC')
        plt.title("Percolate Task with Disconnected Informative Nodes")

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
