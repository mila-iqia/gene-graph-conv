import os
import sys
import pandas as pd
import numpy as np
from datasets import GeneDataset

separators = {'.tsv' : '\t', '.txt': '\t', '.csv': ','}

class CustomDataset(GeneDataset):
    def __init__(self, name, expr_path, label_path, label_name):
        self.name = name
        self.expr_path = expr_path
        self.label_path = label_path
        self.label_name = label_name
        super(CustomDataset, self).__init__()
        
    def load_data(self):
        # Load expression and label files, samples as rows and genes/label names as columns
        sep = separators[os.path.splitext(self.expr_path)[1]]
        self.df = pd.read_csv(self.expr_path, sep=sep, index_col=0)
        sep = separators[os.path.splitext(self.label_path)[1]]
        self.lab = pd.read_csv(self.label_path, sep=sep, index_col=0)
        self.node_names = self.df.columns.values
        self.sample_names =  self.df.index.values
        self.nb_nodes = self.df.shape[1]
        self.data = self.df.values
        self.labels = self.lab[self.label_name].values
        
    def __getitem__(self, idx):
        # label : class of the sample, # sample for all genes
        sample = self.df.iloc[idx,:].values
        sample = np.expand_dims(sample, axis=-1)
        label = self.labels[idx]
        sample = {'sample': sample, 'labels': label}
        return sample
    
    def __len__(self):
        pass
