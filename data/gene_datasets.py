import os
import h5py
import pandas as pd
import collections
import logging
import numpy as np
from datasets import Dataset


class GeneDataset(Dataset):
    """Gene Expression Dataset."""

    def __init__(self, data_dir=None, data_file=None, sub_class=None, name=None, opt=None):
        """
        Args:
            data_file (string): Path to the h5df file.
        """

        self.sub_class = sub_class
        self.data_dir = data_dir
        self.data_file = data_file
        super(GeneDataset, self).__init__(name=name, opt=opt)

    def load_data(self):
        data_file = os.path.join(self.data_dir, self.data_file)
        self.file = h5py.File(data_file, 'r')
        self.data = np.array(self.file['expression_data'])
        self.nb_nodes = self.data.shape[1]
        self.labels = self.file['labels_data']
        self.sample_names = self.file['sample_names']
        self.node_names = self.file['gene_names']
        self.df = pd.DataFrame(np.array(self.file['expression_data']))
        self.df.columns = self.node_names
        self.nb_class = self.nb_class if self.nb_class is not None else len(self.labels[0])
        self.label_name = self.labels.attrs

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
    def __init__(self, data_dir='/data/lisa/data/genomics/TCGA/', data_file='TCGA_tissue_ppi.hdf5', **kwargs):
        super(TCGATissue, self).__init__(data_dir=data_dir, data_file=data_file, name='TCGATissue', **kwargs)


def get_high_var_genes(df):
    return df.var().sort_values()[-5000:]

def get_neighbors(df):
    import pdb; pdb.set_trace()

class TCGAInference(GeneDataset):
    """TCGA Dataset. We predict tissue."""
    def __init__(self, data_dir='/data/lisa/data/genomics/TCGA/', data_file='TCGA_tissue_ppi.hdf5', **kwargs):
        super(TCGAInference, self).__init__(data_dir=data_dir, data_file=data_file, name='TCGAInference', **kwargs)

    def load_data(self):
        data_file = os.path.join(self.data_dir, self.data_file)
        self.file = h5py.File(data_file, 'r')
        self.data = np.array(self.file['expression_data'])
        self.labels = self.file['labels_data']
        self.sample_names = self.file['sample_names']
        self.node_names = self.file['gene_names']
        self.df = pd.DataFrame(np.array(self.file['expression_data']))
        self.df.columns = self.node_names
        self.nb_class = self.nb_class if self.nb_class is not None else len(self.labels[0])
        self.label_name = self.labels.attrs

        # determine the variance in gene expression for each gene
        candidates = get_high_var_genes(self.df)

        # Make there be one set of labels which is the expression value of the target gene
        self.candidate_names = candidates.index.values.tolist()
        self.gene_to_infer = self.candidate_names[-4800]

        self.labels = [1 if x > self.df[self.gene_to_infer].mean() else 0 for x in self.df[self.gene_to_infer]]
        self.df = self.df.drop(self.gene_to_infer, axis=1)
        self.data = self.df.values
        self.nb_nodes = self.data.shape[1]

        # Do LR
        # also, reduce the gene dataset to just the first degree neighbors of the target gene
        # if self.labels.shape != self.labels[:].reshape(-1).shape:
        #    print "Converting one-hot labels to integers"
        #    self.labels = np.argmax(self.labels[:], axis=1)

        # Take a number of subclasses
        if self.sub_class is not None:
            self.data = self.data[[i in self.sub_class for i in self.labels]]
            self.labels = self.labels[[i in self.sub_class for i in self.labels]]
            self.nb_class = len(self.sub_class)
            for i, c in enumerate(np.sort(self.sub_class)):
                self.labels[self.labels == c] = i


class TCGAForLabel(GeneDataset):
    """TCGA Dataset."""
    def __init__(self,
                 data_dir='/data/lisa/data/genomics/TCGA/',
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


class BRCACoexpr(GeneDataset):
    """Breast cancer, with coexpression graph. """
    def __init__(self, data_dir='/data/lisa/data/genomics/TCGA/', data_file='BRCA_coexpr.hdf5', nb_class=2, **kwargs):

        # For this dataset, when we chose 2 classes, it's the 'Infiltrating Ductal Carcinoma'
        # and the 'Infiltrating Lobular Carcinoma'

        sub_class = None
        if nb_class == 2:
            nb_class = None
            sub_class = [0, 7]

        super(BRCACoexpr, self).__init__(data_dir=data_dir, data_file=data_file,
                                         nb_class=nb_class, sub_class=sub_class, name='BRCACoexpr',
                                         **kwargs)


class GBMDataset(GeneDataset):
    " Glioblastoma Multiforme dataset"
    def __init__(self, data_dir="/data/lisa/data/genomics/TCGA/", data_file="gbm.hdf5", opt=None):
        super(GBMDataset, self).__init__(data_dir=data_dir, data_file=data_file, name='GBMDataset', opt=opt)


class NSLRSyntheticDataset(GeneDataset):
    " SynMin dataset"
    def __init__(self, data_dir="/data/lisa/data/genomics/TCGA/", data_file="syn_nslr.hdf5", nb_class=2, **kwargs):
        super(NSLRSyntheticDataset, self).__init__(data_dir=data_dir, data_file=data_file, nb_class=nb_class, name='NSLRSyntheticDataset', **kwargs)
