import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fileinput

def main(argv=None):
    opt = parser.parse_args(argv)
    dfs = []
    df = pd.DataFrame()
    params = [{'column_name': 'extra_cn', 'pretty_name': 'Percolate Task with Extra Connected Nodes'}, {'column_name': 'extra_ucn', 'pretty_name': 'Percolate Task with Extra Unconnected Nodes'}, {'column_name': 'disconnected', 'pretty_name': 'Percolate Task with Removed Informative Edges'}]
    model_names = {'lr_with_l1': 'Lasso', 'slr_with_l1': 'SLR', 'mlp': 'MLP', 'cgn_no_pool': 'GCN', 'cgn_pool': 'GCN (Pooling)', 'cgn_dropout': 'GCN (Dropout)'}
    x_axis = {'extra_cn': '# of Additional Connected Nodes', 'extra_ucn': '# of Additional Unconnected Nodes', 'disconnected': '# of Additional Nodes with no Connections'}
    dirs = ['data/cn/2018-02-101726/', 'data/ucn/2018-02-101727/', 'data/disconnected/2018-02-101728/']
    fig, axes = plt.subplots(nrows=1, ncols=3)

    for d in dirs:
        print d
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
                    df = df.reindex(df[v['column_name']])
                    df.test_auc_mean.name = model_names[df.model[0]]
                    legend=False
                    if v['column_name'] == "extra_cn":
                        legend=True
                    df.test_auc_mean.plot(ax=axes[index], yerr=df.test_auc_std, sharey=True, legend=legend, sort_columns=True)
                    axes[index].set_title(v['pretty_name']);
                    axes[index].set_xlabel(x_axis[v['column_name']])
                    axes[index].set_ylabel('Mean AUROC')
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=0.05, hspace=0.001)
    plt.show()

    plt.errorbar(df[opt.v], df.test_auc_mean, df.test_auc_std)


    plt.show()

if __name__ == '__main__':
    main()
