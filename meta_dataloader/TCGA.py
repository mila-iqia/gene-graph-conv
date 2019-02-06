from torch.utils.data import Dataset, DataLoader
import sys
import os
import numpy as np
import pandas as pd
import h5py


class TCGAMeta(Dataset):
    """ Meta_TCGA Dataset.

    """

    def __init__(self, data_dir=None, dataset_transform=None, transform=None, target_transform=None, download=False, preload=True):
        self.dataset_transform = dataset_transform
        self.target_transform = target_transform
        self.transform = transform
        # specify a default data directory
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')

        if download:
            with open(os.path.join(os.path.dirname(__file__), 'cancers')) as f:
                cancers = f.readlines()
            # remove whitespace
            cancers = [x.strip() for x in cancers]

            _download(data_dir, cancers)

        self.task_ids = get_TCGA_task_ids(data_dir)

        if preload:
            try:
                with h5py.File(os.path.join(data_dir, 'TCGA_tissue_ppi.hdf5'), 'r') as f:
                    self.gene_expression_data = f['expression_data'][()]

                    gene_ids_file = os.path.join(data_dir, 'gene_ids')
                    all_sample_ids_file = os.path.join(data_dir, 'all_sample_ids')
                    with open(gene_ids_file, 'r') as file:
                        gene_ids = file.readlines()
                        self.gene_ids = [x.strip() for x in gene_ids]
                    with open(all_sample_ids_file, 'r') as file:
                        all_sample_ids = file.readlines()
                        self.all_sample_ids = [x.strip() for x in all_sample_ids]

                    self.preloaded = (self.all_sample_ids, self.gene_ids, self.gene_expression_data)
            except:
                print('TCGA_tissue_ppi.hdf5 could not be read from the data_dir.')
                sys.exit()

        else:
            self.preloaded=None

    # convenience method to be used with torch dataloaders
    @staticmethod
    def collate_fn(data):
        """
        Args:
            task (Dataset) : A task from the TCGA Metadataset.

        Returns:
            dataset: the argument dataset unchanged.

            This function performs no operation. It is used to overwrite the default collate_fn of torchs
            DataLoader because it is not compatible with a batch of Datasets.
        """
        return data

    def get_dataloader(self, *args, **kwargs):
        """
        Args:
            *args : The conventional dataset arg which will be supressed
            **kwargs : The conventional kwargs of a torch Dataloader with exception of dataset and collate_fn

            Returns:
                Meta_TCGA_loader (DataLoader): a configured dataloader for the MetaTCGA dataset.

                A convenience function for creating a dataloader which handles passing the right collate_fn
                and the dataset.
        """

        # Delete those kwargs if the have been passed in error
        kwargs.pop('collate_fn', None)
        kwargs.pop('dataset', None)
        return DataLoader(self, **kwargs, collate_fn=TCGAMeta.collate_fn)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dataset: a dataset which represents a specific task from the set of TCGA tasks.

            A task is defined by a target variable, which should be predicted from the gene expression data of a patient.
            The target variable is a combination of a clinical attribute and one of 39 types of cancer.
            An example of a target variable is: 'gender-BRCA', where we predict gender for breast cancer(BRCA) patients.
        """
        dataset = TCGATask(self.task_ids[index], transform=self.transform, target_transform=self.target_transform, download=False, preloaded=self.preloaded)

        if self.target_transform is not None:
            dataset = self.dataset_transform(dataset)
        return dataset

    def __len__(self):
        return len(self.task_ids)


class TCGATask(Dataset):
    def __init__(self, task_id, data_dir=None, transform=None, target_transform=None, download=False, preloaded=None):
        self.id = task_id
        self.transform = transform
        self.target_transform = target_transform

        task_variable, cancer = task_id

        # specify a default data directory
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')

        if download:
            _download(data_dir, [cancer])

        if preloaded is None:
            try:
                with h5py.File(os.path.join(data_dir, 'TCGA_tissue_ppi.hdf5'), 'r') as f:
                    gene_ids_file = os.path.join(data_dir, 'gene_ids')
                    all_sample_ids_file = os.path.join(data_dir, 'all_sample_ids')
                    with open(gene_ids_file, 'r') as file:
                        gene_ids = file.readlines()
                        self.gene_ids = [x.strip() for x in gene_ids]
                    with open(all_sample_ids_file, 'r') as file:
                        all_sample_ids = file.readlines()
                        self.all_sample_ids = [x.strip() for x in all_sample_ids]

            except:
                print('TCGA_tissue_ppi.hdf5 could not be read from the data_dir.')
                sys.exit()
        else:
            self.all_sample_ids, self.gene_ids, self.data = preloaded

        # load the cancer specific matrix
        matrix = pd.read_csv(os.path.join(data_dir, 'clinicalMatrices', cancer + '_clinicalMatrix'), delimiter='\t')
        # TODO: verify we don't need this
        #matrix.drop_duplicates(subset=['sampleID'], keep='first', inplace=True)
        ids = matrix['sampleID']
        attribute = matrix[task_variable]

        # filter all elements where the clinical variable is not available or the associated gene expression data
        available_elements = attribute.notnull() & matrix['sampleID'].isin(self.all_sample_ids)
        sample_ids = ids[available_elements].tolist()
        filtered_attribute = attribute[available_elements].astype('category').cat
        self._labels = filtered_attribute.codes.tolist()
        self.categories = filtered_attribute.categories.tolist()
        self.num_classes = len(self.categories)

        # generator to retrieve the specific indices we need
        indices_to_load = (sample_ids.index(sample_id) for sample_id in sample_ids)

        # lazy loading or loading from preloaded data if available
        if preloaded is None:
            with h5py.File(os.path.join(data_dir, 'TCGA_tissue_ppi.hdf5'), 'r') as f:
                self._samples = f['expression_data'][indices_to_load, :]
        else:
            self._samples = self.data[np.array(list(indices_to_load), dtype=int), :]

        self.input_size = self._samples.shape[1]

    def __getitem__(self, index):
        sample = self._samples[index, :]
        label = self._labels[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (sample, label)

    def __len__(self):
        return self._samples.shape[0]


def get_TCGA_task_ids(data_dir=None, min_samples=250, max_samples=sys.maxsize, task_variables_file=None):
    # specify a default data directory

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')

    try:
        with h5py.File(os.path.join(data_dir, 'TCGA_tissue_ppi.hdf5'), 'r') as f:
            all_sample_ids = [x.decode('utf-8') for x in f['sample_names']]
    except:
        print('TCGA_tissue_ppi.hdf5 could not be read from the data_dir.')
        sys.exit()

    if task_variables_file is None:
        task_variables_file = os.path.join(os.path.dirname(__file__), 'task_variables')

    with open(task_variables_file) as f:
        task_variables = f.readlines()
    # remove whitespace
    task_variables = [x.strip() for x in task_variables]

    task_ids = []
    for filename in os.listdir(os.path.join(data_dir, 'clinicalMatrices')):

        matrix = pd.read_csv(os.path.join(data_dir, 'clinicalMatrices', filename), delimiter='\t')
        for task_variable in task_variables:
            try:
                # if this task_variable exists for this cancer find the sample_ids for this task
                filter_clinical_variable_present = matrix[task_variable].notnull()
                # filter out all sample_ids for which no valid value exists
                potential_sample_ids = matrix['sampleID'][filter_clinical_variable_present]
                # filter out all sample_ids for which no gene expression data exists
                task_sample_ids = [sample_id for sample_id in potential_sample_ids if sample_id in all_sample_ids]
            except KeyError:
                continue

            task_id = (task_variable, filename.split('_')[0])

            num_samples = len(task_sample_ids)
            # only add this task for the specified range of number of samples
            if min_samples < num_samples < max_samples:
                task_ids.append(task_id)
    return task_ids


def _download(data_dir, cancers):
    import academictorrents as at
    from six.moves import urllib
    import gzip

    # download files
    try:
        os.makedirs(os.path.join(data_dir, 'clinicalMatrices'))
    except OSError as e:
        if e.errno == 17:
            pass
        else:
            raise

    for cancer in cancers:
        filename = '{}_clinicalMatrix'.format(cancer)
        file_path = os.path.join(data_dir, 'clinicalMatrices', filename)
        decompressed_file_path = file_path.replace('.gz', '')

        if os.path.isfile(file_path):
            continue

        file_path += '.gz'

        url = 'https://tcga.xenahubs.net/download/TCGA.{}.sampleMap/{}_clinicalMatrix.gz'.format(cancer, cancer)

        print('Downloading ' + url)
        data = urllib.request.urlopen(url)

        with open(file_path, 'wb') as f:
            f.write(data.read())
        with open(decompressed_file_path, 'wb') as out_f, gzip.GzipFile(file_path) as zip_f:
            out_f.write(zip_f.read())
        os.unlink(file_path)

        if os.stat(decompressed_file_path).st_size == 0:
            os.remove(decompressed_file_path)
            error = IOError('Downloading {} from {} failed.'.format(filename, url))
            error.strerror = 'Downloading {} from {} failed.'.format(filename, url)
            error.errno = 5
            error.filename = decompressed_file_path
            raise error

    gene_expression_data = os.path.join(data_dir, 'TCGA_tissue_ppi.hdf5')
    if not os.path.isfile(gene_expression_data):
        print('Downloading TCGA_tissue_ppi.hdf5 using academictorrents')
        at.get("4070a45bc7dd69584f33e86ce193a2c903f0776d", datastore=data_dir)

    gene_ids_file = os.path.join(data_dir, 'gene_ids')
    all_sample_ids_file = os.path.join(data_dir, 'all_sample_ids')

    if not os.path.isfile(gene_ids_file):
        print('Processing...')

        with h5py.File(gene_expression_data, 'r') as f:
            gene_ids = [x.decode('utf-8') for x in f['gene_names']]
            all_sample_ids = [x.decode('utf-8') for x in f['sample_names']]

        with open(gene_ids_file, "w") as text_file:
            for gene_id in gene_ids:
                text_file.write('{}\n'.format(gene_id))

        if not os.path.isfile(all_sample_ids_file):
            with open(all_sample_ids_file, "w") as text_file:
                for sample_id in all_sample_ids:
                    text_file.write('{}\n'.format(sample_id))

        print('Done!')
