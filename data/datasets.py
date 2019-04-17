"""Imports Datasets"""
import csv
import glob
import os
import urllib
import zipfile
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import academictorrents as at
import data.utils
from data.utils import symbol_map, ensg_to_hugo_map

class GeneDataset(Dataset):
    """Gene Expression Dataset."""
    def __init__(self):
        self.load_data()

    def load_data(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class TCGADataset(GeneDataset):
    def __init__(self, nb_examples=None, at_hash="e4081b995625f9fc599ad860138acf7b6eb1cf6f", datastore=""):
        self.at_hash = at_hash
        self.datastore = datastore
        self.nb_examples = nb_examples # In case you don't want to load the whole dataset from disk
        super(TCGADataset, self).__init__()

    def load_data(self):
        csv_file = at.get(self.at_hash, datastore=self.datastore)
        hdf_file = csv_file.split(".gz")[0] + ".hdf5"
        if not os.path.isfile(hdf_file):
            print("We are converting a CSV dataset of TCGA to HDF5. Please wait a minute, this only happens the first "
                  "time you use the TCGA dataset.")
            df = pd.read_csv(csv_file, compression="gzip", sep="\t")
            df = df.set_index('Sample')
            df = df.transpose()
            df.to_hdf(hdf_file, key="data", complevel=5)
        self.df = pd.read_hdf(hdf_file)
        self.df.rename(symbol_map(self.df.columns), axis="columns", inplace=True)
        self.df = self.df - self.df.mean(axis=0)
        #self.df = self.df / self.df.variance()
        self.sample_names = self.df.index.values.tolist()
        self.node_names = np.array(self.df.columns.values.tolist()).astype("str")
        self.nb_nodes = self.df.shape[1]
        self.labels = [0 for _ in range(self.df.shape[0])]

    def __getitem__(self, idx):
        sample = np.array(self.df.iloc[idx])
        sample = np.expand_dims(sample, axis=-1)
        label = self.labels[idx]
        sample = {'sample': sample, 'labels': label}
        return sample


class DatasetFromCSV(GeneDataset):
    def __init__(self, name, expr_path, label_path, label_name=None):
        self.name = name
        self.expr_path = expr_path
        self.label_path = label_path
        self.label_name = label_name
        super(DatasetFromCSV, self).__init__()

    def load_data(self):
        # Load expression and label files, samples as rows and genes/label names as columns
        separators = {'.tsv' : '\t', '.txt': '\t', '.csv': ','}
        sep = data.utils.get_file_separator(self.expr_path)
        self.df = pd.read_csv(self.expr_path, sep=sep, index_col=0)

        sep = data.utils.get_file_separator(self.label_path)
        self.lab = pd.read_csv(self.label_path, sep=sep, index_col=0)

        self.node_names = self.df.columns.values
        self.sample_names = self.df.index.values
        self.nb_nodes = self.df.shape[1]
        self.data = self.df.values

        if self.label_name in self.lab.columns:
            self.labels_ = self.lab[self.label_name].values
            self.labels = pd.Categorical(self.labels_).codes
        else : 
            self.labels_ = []
            self.labels = []

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
        self.df = pd.DataFrame(expressions)
        self.df.columns = self.node_names
        self.nb_nodes = self.df.shape[1]
        self.sample_names = contrasts
        self.node_names = genes

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
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


class GTexDataset(GeneDataset):
    """
    You should download the following files and store them in data/datastore/ :
    - https://cbcl.ics.uci.edu/public_data/D-GEX/GTEx_RNASeq_RPKM_n2921x55993.gctx
    - https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=md_ensembl_id&status=Approved&status=Entry
    %20Withdrawn&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit
    """
    def __init__(self, nb_examples=None, data_path="data/datastore/GTEx_RNASeq_RPKM_n2921x55993.gctx", normalize=False):
        self.data_path = data_path
        self.nb_examples = nb_examples  # In case you don't want to load the whole dataset from disk
        self.normalize = normalize
        super(GTexDataset, self).__init__()

    def load_data(self):
        from cmapPy.pandasGEXpress.parse import parse
        self.df = parse(self.data_path).data_df.T
        # Map gene names
        eh_map = ensg_to_hugo_map()
        columns_to_drop = [i for i in self.df.columns if str(i)[str(i).find('ENS'):].split('.')[0] not in eh_map.keys()]
        self.df = self.df.drop(columns_to_drop, axis=1)  # Drop columns whose gene is not covered by the map

        self.df.columns = [eh_map[str(i)[str(i).find('ENS'):].split('.')[0]] for i in self.df.columns]  # Rename columns
        self.df = self.df.loc[:, (self.df != self.df.iloc[0]).any()]
        self.df = self.df - self.df.mean(axis=0)

        if self.normalize:
            self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min()) * 20 - 10

    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :].values
        sample = np.expand_dims(sample, axis=-1)
        sample = {'sample': sample}
        return sample


class GEODataset(GeneDataset):

    def __init__(self, file_path, seed=0, load_full=False, nb_examples=1200, normalize=True):
        """
        Args:
            file_path: Path to the HDF5 file
            load_full: Load the entire dataset into memory or not
            nb_examples: Number of examples to load if load_full is False.

        If load_full is False, the object will load only a randomly sampled dataframe
        with the length nb_examples. The randomize_dataset method can be used to generate
        a new dataset with a specified seed.
        """
        self.file_path = file_path
        self.load_full = load_full
        self.nb_examples = nb_examples
        self.normalize = normalize
        self.seed = seed
        super(GEODataset, self).__init__()

    def load_data(self):
        self.hdf5 = h5py.File(name=self.file_path, mode='r')
        self.expression_data = self.hdf5['expression_data']
        self.nrows, self.ncols = self.expression_data.shape

        # Load all gene names to memory
        self.genes = [x.decode() for x in self.hdf5['gene_names'][()].tolist()]

        if self.load_full:
            self.df = pd.DataFrame(data=self.expression_data[()], columns=self.genes)
        else:
            self.df = self._load_nb_examples()
        self.df.rename(symbol_map(self.df.columns), axis="columns", inplace=True)
        if self.normalize:
            self.df = self.df - self.df.mean(axis=0)


    def randomize_dataset(self, new_seed):
        """
        Sample a new self.df of the same length, but with a new seed.
        """
        self.df = self._load_nb_examples(seed=new_seed)
        self.df.rename(symbol_map(self.df.columns), axis="columns", inplace=True)

        if self.normalize:
            self.df = self.df - self.df.mean(axis=0)

    def _load_nb_examples(self):
        np.random.seed(self.seed)
        indices = np.sort(np.random.choice(self.nrows, size=(self.nb_examples), replace=False))

        # This indexing is very slow
        data = self.expression_data[indices.tolist()]
        return pd.DataFrame(data=data, columns=self.genes)

    def __getitem__(self, idx):
        sample = self.expression_data[idx]
        sample = np.expand_dims(sample, axis=-1)
        return sample
