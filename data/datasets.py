import os
import h5py
import pandas as pd
import collections
import logging
import numpy as np
import academictorrents as at
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, name, seed, nb_class, nb_examples, nb_nodes, nb_master_nodes=0):

        self.name = name
        self.seed = seed
        self.nb_class = nb_class
        self.nb_examples = nb_examples
        self.nb_nodes = nb_nodes
        self.load_data()
        self.df = self.df - self.df.mean(0)
        self.nb_master_nodes = nb_master_nodes

        for master in range(nb_master_nodes):
            self.df.insert(0, 'master_{}'.format(master), 1.)
        self.data = self.df.as_matrix()

    def load_data(self):
        raise NotImplementedError()

    def labels_name(self, l):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __len__(self):
        return self.data.shape[0]

    def get_adj(self):
        return self.adj

class GeneDataset(Dataset):
    """Gene Expression Dataset."""

    def __init__(self, data_dir=None, data_file=None, sub_class=None, name=None, seed=None, nb_class=None, nb_examples=None, nb_nodes=None, **kwargs):
        """
        Args:
            data_file (string): Path to the hdf5 file.
        """

        self.sub_class = sub_class
        self.data_dir = data_dir
        self.data_file = data_file

        super(GeneDataset, self).__init__(name=name, seed=seed, nb_class=nb_class, nb_examples=nb_examples, nb_nodes=nb_nodes, **kwargs)

    def load_data(self):
        try:
            data_file = os.path.join(self.data_dir, self.data_file)
            self.file = h5py.File(data_file, 'r')
            self.data = np.array(self.file['expression_data'][:self.nb_examples])
        except Exception as e:
            data_file = at.get('4070a45bc7dd69584f33e86ce193a2c903f0776d')
            self.file = h5py.File(data_file, 'r')
            self.data = np.array(self.file['expression_data'][:self.nb_examples])
        self.nb_nodes = self.data.shape[1]
        try:
            self.labels = self.file['labels_data']
        except Exception:
            self.labels = np.array([])
        try:
            self.sample_names = self.file['sample_names']
        except Exception:
            self.sample_names = pd.DataFrame([])
        self.node_names = np.array(self.file['gene_names'])
        self.df = pd.DataFrame(self.data)
        self.df.columns = self.node_names[:len(self.df.columns)]
        self.nb_class = self.nb_class if self.nb_class is not None else 2
        self.label_name = self.node_names[len(self.df.columns)+1:]
        self.transform = None

        if self.labels.shape != self.labels[:].reshape(-1).shape:
            print "Converting one-hot labels to integers"
            self.labels = np.argmax(self.labels[:], axis=1)

        # Take a number of subclasses
        if self.sub_class is not None:
            self.data = self.data[[i in self.sub_class for i in self.labels]]
            self.labels = self.labels[[i in self.sub_class for i in self.labels]]
            self.nb_class = len(self.sub_class)
            for i, c in enumerate(np.sort(self.sub_class)):
                self.labels[self.labels == c] = i

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


class TCGATissue(GeneDataset):
    """TCGA Dataset. We predict tissue."""
    def __init__(self, data_dir='data/gene_expression/', data_file='TCGA_tissue_ppi.hdf5', **kwargs):
        super(TCGATissue, self).__init__(data_dir=data_dir, data_file=data_file, name='TCGATissue', **kwargs)

class TCGAGeneInference(GeneDataset):
    """TCGA Dataset. We predict tissue."""
    def __init__(self, data_dir='data/gene_expression/', data_file='TCGA_tissue_ppi.hdf5', **kwargs):
        super(TCGAGeneInference, self).__init__(data_dir=data_dir, data_file=data_file, name='TCGATissue', **kwargs)

        total_gene = 5000
        all_genes = np.arange(total_gene)
        np.random.shuffle(all_genes)

        # Do something not stupid here.
        self.gene_to_keep = all_genes[:total_gene/2]
        self.gene_to_infer = all_genes[total_gene/2:total_gene]
        self.df = self.df.iloc[:, :total_gene]
        self.data = self.data[:, :total_gene]
        self.nb_nodes = total_gene
        self.node_names = self.df.columns


    def __getitem__(self, idx):
        sample = self.data[idx]
        sample[self.gene_to_infer] = 0.
        sample[self.gene_to_keep] += 1e-8

        sample = np.expand_dims(sample, axis=-1)

        label = self.data[idx]
        label[self.gene_to_keep] = 0.
        label[self.gene_to_infer] += 1e-8

        sample = {'sample': sample, 'labels': label}
        return sample

class TCGAForLabel(GeneDataset):
    """TCGA Dataset."""
    def __init__(self,
                 data_dir='data/gene_expression/',
                 data_file='TCGA_tissue_ppi.hdf5',
                 clinical_file="PANCAN_clinicalMatrix.gz",
                 clinical_label="gender",
                 **kwargs):
        super(TCGAForLabel, self).__init__(data_dir=data_dir, data_file=data_file, name='TCGAForLabel', **kwargs)

        dataset = self

        clinical_raw = pd.read_csv(data_dir + clinical_file, compression='gzip', header=0, sep='\t', quotechar='"')
        clinical_raw = clinical_raw.set_index("sampleID")

        logging.info("Possible labels to select from ", clinical_file, " are ", list(clinical_raw.columns))
        logging.info("Selected label is ", clinical_label)

        clinical = clinical_raw[[clinical_label]]
        clinical = clinical.dropna()

        data_raw = pd.DataFrame(dataset.data, index=dataset.sample_names)
        data_raw.index.name = "sampleID"
        samples = pd.DataFrame(np.asarray(dataset.sample_names), index=dataset.sample_names)
        samples.index.name = "sampleID"

        data_joined = data_raw.loc[clinical.index]
        data_joined = data_joined.dropna()
        clinical_joined = clinical.loc[data_joined.index]

        logging.info("clinical_raw", clinical_raw.shape, ", clinical", clinical.shape, ", clinical_joined", clinical_joined.shape)
        logging.info("data_raw", data_raw.shape, "data_joined", data_joined.shape)
        logging.info("Counter for " + clinical_label + ": ")
        logging.info(collections.Counter(clinical_joined[clinical_label].as_matrix()))

        self.labels = pd.get_dummies(clinical_joined).as_matrix().astype(np.float)
        self.data = data_joined.as_matrix()
