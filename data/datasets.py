import h5py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import academictorrents as at


class GeneDataset(Dataset):
    """Gene Expression Dataset."""

    def __init__(self, file_path=None, at_hash=None, nb_class=2, nb_examples=None):
        self.nb_examples = nb_examples
        self.nb_class = nb_class
        if at_hash:
            file_path = at.get(at_hash)
        self.file = h5py.File(file_path, 'r')
        self.nb_examples = nb_examples
        self.data = np.array(self.file['expression_data'][:self.nb_examples])
        self.nb_nodes = self.data.shape[1]
        self.labels = self.file['labels_data']
        self.sample_names = self.file['sample_names']
        self.node_names = np.array(self.file['gene_names'])
        self.df = pd.DataFrame(self.data)
        self.df.columns = self.node_names[:len(self.df.columns)]
        self.label_name = self.node_names[len(self.df.columns)+1:]

        if self.labels.shape != self.labels[:].reshape(-1).shape:
            print "Converting one-hot labels to integers"
            self.labels = np.argmax(self.labels[:], axis=1)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = np.expand_dims(sample, axis=-1)
        label = self.labels[idx]
        sample = {'sample': sample, 'labels': label}
        return sample

    def labels_name(self, l):
        if type(self.label_name[str(l)]) == np.ndarray:
            return self.label_name[str(l)][0]
        return self.label_name[str(l)]
