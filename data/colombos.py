import os
import pandas as pd
import collections
import logging
import numpy as np
from datasets import Dataset

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import colombos_load_data


class EcoliDataset(Dataset):

    def __init__(self, name="EcoliDataset", opt=None):
        super(EcoliDataset, self).__init__(name=name, opt=opt)

    def load_data(self):
        

        data =  colombos_load_data.load("ecoli", False)
        
        self.data = data[0]
        self.nb_nodes = self.data.shape[1]
        #self.labels = self.file['labels_data']
        self.sample_names = data[1]
        self.node_names = data[2]
        self.df = pd.DataFrame(self.data)
        self.df.columns = self.node_names
        self.nb_class = self.nb_class if self.nb_class is not None else len(self.labels[0])
        #self.label_name = self.labels.attrs

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = np.expand_dims(sample, axis=-1)
        label = self.labels[idx]
        sample = [sample, label]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def labels_name(self, l):
        if type(self.label_name[str(l)]) == np.ndarray:
            return self.label_name[str(l)][0]
        return self.label_name[str(l)]
