import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fileinput

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
    parser.add_argument('-p', type=bool, help="Parses the multiply printed logs")
    opt = parser.parse_args(argv)
    files = [os.path.join(opt.d, f) for f in os.listdir(opt.d) if os.path.isfile(os.path.join(opt.d, f)) and 'slurm' in f]
    for f in sorted(files):
        for line in fileinput.input(f, inplace=1):
            if "CANCELLED" in line or "Exceeded" in line or "\n" == line:
                continue
            else:
                print line

        df = pd.read_csv(f)
        if "dropout" in df:
            df = df.drop(['dropout'], axis=1)
        if df.valid_acc_mean.dtype != 'float64':
            df = df.dropna(axis=0)

            last_copy = df[df.valid_acc_mean.str.contains('valid_acc_mean')].tail(1).index.values[0] + 1
            df = df.drop(df.index[:last_copy])
            df = df.reset_index()
            df = df.apply(pd.to_numeric, errors='ignore')
        df.test_auc_mean.name = df.model[0]
        plt.errorbar(df[opt.v], df.test_auc_mean, df.test_auc_std)

    if opt.v == "extra_ucn":
        plt.xlabel('# Additional Unconnected Nodes')
        plt.ylabel('Mean AUROC')
        plt.title("Percolate Task with Extra Unconnected Nodes")
        plt.legend()
    elif opt.v == "extra_cn":
        plt.xlabel('# Additional Connected Nodes')
        plt.ylabel('Mean AUROC')
        plt.title("Percolate Task with Extra Connected Nodes")
        plt.legend()
    elif opt.v == "disconnected":
        plt.xlabel('# of Informative Edges Removed')
        plt.ylabel('Mean AUROC')
        plt.title("Percolate Task with Removed Informative Edges")
        plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
