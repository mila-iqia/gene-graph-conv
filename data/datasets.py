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
    def __init__(self, nb_examples=None, at_hash_or_path="4070a45bc7dd69584f33e86ce193a2c903f0776d"):
        self.at_hash_or_path = at_hash_or_path
        self.nb_examples = nb_examples # In case you don't want to load the whole dataset from disk
        super(TCGADataset, self).__init__()

    def load_data(self):
        # You could replace the value of self.hash with a path to a local copy of your graph and AT can handle that.
        self.file_path = at.get(self.at_hash_or_path)
        self.file = h5py.File(self.file_path, 'r')
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
