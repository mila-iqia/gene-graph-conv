import os
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import fileinput

def main(argv=None):
    dfs = []
    df = pd.DataFrame()
    params = [{'column_name': 'extra_cn', 'pretty_name': 'Percolate Task \n with Connected Uninformative Nodes'}, {'column_name': 'extra_ucn', 'pretty_name': 'Percolate Task \n with Unconnected Uninformative Nodes'}, {'column_name': 'disconnected', 'pretty_name': 'Percolate Task \n with Removed Edges'}]
    model_names = {'lr_with_l1': 'Lasso', 'lr_no_l1': 'Logistic Regression','slr_no_l1': 'SLR (no L1)', 'random': 'Random', 'slr_with_l1': 'SLR', 'mlp': 'MLP', 'cgn_no_pool': 'GCN', 'cgn_pool': 'GCN (Pooling)', 'cgn_dropout': 'GCN (Dropout)'}
    x_axis = {'extra_cn': '# of Connected Uninformative Nodes', 'extra_ucn': '# of Unconnected Uninformative Nodes', 'disconnected': '# of Edges Removed'}
    linestyles = {'random': ':', 'slr_with_l1': '--', 'mlp': '--', 'cgn_no_pool': '-.', 'cgn_pool': ':', 'cgn_dropout': '-'}
    dirs = ['data/cn/iclr_data/', 'data/ucn/iclr_data/', 'data/disconnected/iclr_data/']
    fig, axes = plt.subplots(nrows=1, ncols=3)
    for d in dirs:
        files = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and 'slurm' in f]
        for f in sorted(files):
            print f
            df = pd.read_csv(f)
            if "dropout" in df:
                df = df.drop(['dropout'], axis=1)
            if df.valid_acc_mean.dtype != 'float64':
                df = df.dropna(axis=0)

                last_copy = df[df.valid_acc_mean.str.contains('valid_acc_mean')].tail(1).index.values[0] + 1
                df = df.drop(df.index[:last_copy])
                df = df.reset_index()
                df = df.apply(pd.to_numeric, errors='ignore')

            for index, v in enumerate(params):
                if v['column_name'] in df.columns:
                    df_index = df[v['column_name']]
                    if v['column_name'] == "extra_cn":
                        new_index = []
                        for i in df_index:
                            new_index.append(calculate_cn(df_index[:i]))
                        df_index = np.array(new_index)
                    df = df.set_index(df_index)
                    if df.model[0] == "lr_with_l1":
                        continue
                    df.test_auc_mean.name = model_names[df.model[0]]
                    dashes = []
                    if df.model[0] == "slr_with_l1":
                        df.test_auc_mean.plot(ax=axes[index], yerr=df.test_auc_std, sharey=True, legend=False, markersize=0, linestyle=linestyles[df.model[0]], dashes=(5, 2), fontsize=13)
                    else:
                        df.test_auc_mean.plot(ax=axes[index], yerr=df.test_auc_std, sharey=True, legend=False, markersize=0, linestyle=linestyles[df.model[0]], fontsize=13)

                    axes[index].set_title(v['pretty_name'], fontsize=22);
                    axes[index].set_xlabel(x_axis[v['column_name']], fontsize=20, labelpad=14)
                    axes[index].set_ylabel('Mean AUROC', fontsize=20, labelpad=14)
    plt.legend(bbox_to_anchor=(-1.66, 1.18, 2.0, .122), loc=3, ncol=5, mode="expand", borderaxespad=0., fontsize=20, frameon=True)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.8, wspace=0.05, hspace=1.1)
    plt.show()

def calculate_cn(index):
    if index.tolist() == []:
        return 0
    else:
        return ((2 * (6 + index[len(index)-1])) + 1) + calculate_cn(index[:len(index)-1])

if __name__ == '__main__':
    main()
