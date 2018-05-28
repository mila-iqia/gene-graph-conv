import pandas as pd
import h5py
import numpy as np

df = pd.read_csv('~/Desktop/GSE83533_Dx_Rel_AML_RNAseq_rpkm.csv', sep='\t')
del df['chromosome']
df.index = df['gene']
del df['gene']
df = df.filter(like='Dx', axis=1)
df = df.reindex_axis(sorted(df.columns), axis=1).T

times = np.array([203, 298, 1744, 236, 340, 269, 331,149, 637,104,417,607,216,638,665,507,466, 636,158])
labels = [1 if x > times.mean() else 0 for x in times ]
import pdb; pdb.set_trace()
f = h5py.File("/Users/martinweiss/Desktop/aml.hdf5", 'w')
f.create_dataset("expression_data", data=df)
f.create_dataset("labels_data", data=labels)
f.create_dataset("sample_names", data=df.index.values.tolist())
f.create_dataset("gene_names", data=df.columns.values.tolist())
f.close()
