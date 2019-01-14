import os
import h5py
import pandas as pd
from torch.utils.data import Dataset
import academictorrents as at


DIR = '/Users/martinweiss/code/academic/conv-graph/data/clinical/'

class TCGADataset(Dataset):
    def __init__(self, normalize=False, limit=100000):
        self.labels = pd.read_csv(DIR + 'tcga_labels.csv', low_memory=False)  # reading TCGA labels
        self.labels.drop_duplicates('sampleID', keep='first', inplace=True)
        self.labels.index = self.labels['sampleID']
        #del self.labels['sampleID']

        f = h5py.File(at.get("4070a45bc7dd69584f33e86ce193a2c903f0776d"))
        self.data = pd.DataFrame(f['expression_data'][:limit])
        self.data.columns = f['gene_names'][:]
        self.data.index = f['sample_names'][:limit]
        self.data.index = [x.decode('utf-8') for x in self.data.index]

        # remove the data that don't have gene expression.
        self.labels = self.labels.loc[self.data.index]

        if normalize:
            self.data = ((self.data - self.data.mean(0)) / ((self.data - self.data.mean(0)).std() + 1e-6))

class Task(Dataset):
    def __init__(self, fulldataset, task_id, limit=100):  # tcga id blaclist: blacklist
        label_name, tissue_type = task_id.split('-')
        df = fulldataset.labels.loc(axis=1)[label_name, 'clinicalMatrix'].dropna()  # tissue type
        label_indices = df[df['clinicalMatrix'] == tissue_type].index
        intersection = [x for x in label_indices.tolist() if x in fulldataset.data.index]
        #print("intersection:{}" .format(intersection))
        self.metadata = None

        self.labels = fulldataset.labels.loc[intersection][label_name][:limit].astype('category').cat.codes
        self.num_labels = len(self.labels.unique())
        self.num_all_labels = len(fulldataset.labels.loc[intersection][label_name].astype('category').cat.codes.unique())
        self.data = fulldataset.data.loc[intersection][:limit]
        self.attrs = [label_name, tissue_type]
        self.id = label_name + '-' + tissue_type

    def get_num_examples(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :].as_matrix()
        sample = [sample, self.labels.iloc[idx]]
        return sample

    def __len__(self):
        return self.data.shape[0]
