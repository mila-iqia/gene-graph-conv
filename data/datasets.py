"""Imports Datasets"""
import glob
import os
import urllib
import zipfile

import h5py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import academictorrents as at


class GeneDataset(Dataset):
    """Gene Expression Dataset."""
    def __init__(self):
        self.load_data()

    def load_data(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class TCGADataset(GeneDataset):
    def __init__(self, nb_examples=None, at_hash="4070a45bc7dd69584f33e86ce193a2c903f0776d", datastore=""):
        self.at_hash = at_hash
        self.datastore = datastore
        self.nb_examples = nb_examples # In case you don't want to load the whole dataset from disk
        super(TCGADataset, self).__init__()

    def load_data(self):
        self.file_path = at.get(self.at_hash, datastore=self.datastore)
        self.file = h5py.File(self.file_path, 'r')
        self.data = np.array(self.file['expression_data'][:self.nb_examples])
        self.nb_nodes = self.data.shape[1]
        self.labels = self.file['labels_data']
        self.sample_names = self.file['sample_names']
        self.node_names = np.array(self.file['gene_names']).astype("str")
        self.df = pd.DataFrame(self.data)
        self.df.columns = self.node_names[:len(self.df.columns)]
        self.label_name = self.node_names[len(self.df.columns)+1:]

        if self.labels.shape != self.labels[:].reshape(-1).shape:
            print("Converting one-hot labels to integers")
            self.labels = np.argmax(self.labels[:], axis=1)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = np.expand_dims(sample, axis=-1)
        label = self.labels[idx]
        sample = {'sample': sample, 'labels': label}
        return sample


class DatasetFromCSV(GeneDataset):
    def __init__(self, name, expr_path, label_path, label_name):
        self.name = name
        self.expr_path = expr_path
        self.label_path = label_path
        self.label_name = label_name
        super(CustomDataset, self).__init__()

    def load_data(self):
        # Load expression and label files, samples as rows and genes/label names as columns
        separators = {'.tsv' : '\t', '.txt': '\t', '.csv': ','}
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

class EcoliDataset(GeneDataset):
    def __init__(self):
        super(EcoliDataset, self).__init__()

    def load_data(self):
        # Go and prepare the actual Data
        data_dir = "colombos_data"
        organism = "ecoli"
        source = "http://www.colombos.net/cws_data/compendium_data"
        zipfname = "%s_compendium_data.zip" %(organism)

        # Download data if necessary.
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.isfile(data_dir + "/%s" %(zipfname)):
            print("Downloading %s data..." %(organism))
            urllib.urlretrieve("%s/%s" %(source, zipfname), data_dir + "/%s" %(zipfname))

            # Extract data.
            fh = zipfile.ZipFile(data_dir + "/%s" %(zipfname))
            fh.extractall(data_dir)
            fh.close()

        # Prepare data for later processing.
        print("Preparing %s data..." %(organism))
        expfname = glob.glob(data_dir + "/colombos_%s_exprdata_*.txt" %(organism))[0]
        refannotfname = glob.glob(data_dir + "/colombos_%s_refannot_*.txt" %(organism))[0]
        testannotfname = glob.glob(data_dir + "/colombos_%s_testannot_*.txt"
                                   %(organism))[0]

        df = pd.read_table(expfname, skiprows = 5, header = 1)
        df = df.fillna(0.0)
        genes = df["Gene name"].values
        expressions = df.iloc[:, 3:len(df.columns)].values
        contrasts = np.array(open(expfname,
                                     "r").readline().strip().split('\t')[1:],
                                dtype = object)
        lines = open(refannotfname, "r").readlines()
        refannot = {}
        for line in lines[1:]:
            contrast, annot = line.strip().split("\t")
            refannot.setdefault(contrast, set())
            refannot[contrast].add(annot)
        lines = open(testannotfname, "r").readlines()
        testannot = {}
        for line in lines[1:]:
            contrast, annot = line.strip().split("\t")
            testannot.setdefault(contrast, set())
            testannot[contrast].add(annot)

        # Transpose and standardize expressions.
        expressions = expressions.T
        expressions_mean = np.mean(expressions, axis = 0)
        expressions_std = np.std(expressions, axis = 0)
        expressions = (expressions - expressions_mean) / expressions_std
        expressions = np.nan_to_num(expressions)

        # Make the imported data accessible on the dataset object
        self.data = expressions
        self.nb_nodes = self.data.shape[1]
        self.sample_names = contrasts
        self.node_names = genes
        self.df = pd.DataFrame(self.data)
        self.df.columns = self.node_names

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
