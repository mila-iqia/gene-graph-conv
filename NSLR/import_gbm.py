import os
import pandas as pd
import h5py
import numpy as np
from pandas import HDFStore, read_hdf

graph_file = os.path.join('/data/lisa/data/genomics/TCGA/BRCA_coexpr.hdf5')
f_brca = h5py.File(graph_file, 'r')

data_brca = np.array(f_brca['expression_data'])
nb_nodes_brca = data_brca.shape[1]
labels_brca = f_brca['labels_data']
adj_brca = np.array(f_brca['graph_data']).astype('float32')
sample_names_brca = f_brca['sample_names']
node_names_brca = np.array(f_brca['gene_names'])

f = h5py.File("/data/lisa/data/genomics/TCGA/gbm.hdf5")
exp = pd.read_csv("gbm_exp.csv")
survival = pd.read_csv("gbm_survival.csv")

survival.index = survival['bcr_patient_barcode']
labels_gbm = survival.loc[survival['lable1'] != 3]
labels_gbm = labels_gbm['lable1']
labels_gbm.subtract(1)
data_gbm = exp.drop(exp.columns[0], axis=1).T
data_gbm.index = survival['bcr_patient_barcode']
data_gbm = data_gbm[data_gbm.index.isin(labels_gbm.index)]
nb_nodes_gbm = data_gbm.shape[1]
adj_gbm = pd.read_table('pathway_commons_adj.csv.gz', sep=' ').fillna(0.0)
sample_names_gbm = pd.Series(exp.columns.values[1:])
node_names_gbm = exp[exp.columns[0]]

np.array(adj_gbm).astype('float32').sum()

f.create_dataset('expression_data', data=data_gbm)
f.create_dataset('labels_data', data=labels_gbm)
f['labels_data'].attrs['0'] = "died"
f['labels_data'].attrs['1'] = "survived"

f.create_dataset('graph_data', data=adj_gbm)
f.create_dataset('gene_names', data=node_names_gbm, dtype=h5py.special_dtype(vlen=str))
f.create_dataset('sample_names', data=sample_names_gbm, dtype=h5py.special_dtype(vlen=str))

f.close()
