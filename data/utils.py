import logging
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from gene_datasets import TCGATissue, TCGAGeneInference
from datasets import RandomDataset
import data


def split_dataset(dataset, batch_size=100, random=False, train_ratio=0.8, seed=1993, nb_samples=None, nb_per_class=None):
    logger = logging.getLogger()
    all_idx = range(len(dataset))

    if random:
        np.random.seed(seed)
        np.random.shuffle(all_idx)

    if nb_samples is not None:
        all_idx = all_idx[:nb_samples]
        nb_example = len(all_idx)
        logger.info("Going to subsample to {} examples".format(nb_example))

    # If we want to keep a specific number of examples per class in the training set.
    # Since the
    if nb_per_class is not None:

        idx_train = []
        idx_rest = []

        idx_per_class = {i: [] for i in range(dataset.nb_class)}
        nb_finish = np.zeros(dataset.nb_class)

        last = None
        for no, i in enumerate(all_idx):

            last = no
            label = dataset[i]['labels']
            label = np.argmax(label)

            if len(idx_per_class[label]) < nb_per_class:
                idx_per_class[label].append(i)
                idx_train.append(i)
            else:
                nb_finish[label] = 1
                idx_rest.append(i)

            if nb_finish.sum() >= dataset.nb_class:
                break

        idx_rest += all_idx[last+1:]

        idx_valid = idx_rest[:len(idx_rest)/2]
        idx_test = idx_rest[len(idx_rest)/2:]

        logger.info("Keeping {} examples in training set total.".format(len(idx_train)))

    else:
        nb_example = len(all_idx)
        nb_train = int(nb_example * train_ratio)
        nb_rest = (nb_example - nb_train) / 2
        nb_valid = nb_train + int(nb_rest)
        nb_test = nb_train + 2 * int(nb_rest)

        idx_train = all_idx[:nb_train]
        idx_valid = all_idx[nb_train:nb_valid]
        idx_test = all_idx[nb_valid:nb_test]

    train_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(idx_train))
    test_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(idx_test))
    valid_set = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(idx_valid))
    logger.info("Our sets are of length: train={}, valid={}, tests={}".format(len(idx_train), len(idx_valid), len(idx_test)))
    return train_set, valid_set, test_set


def get_dataset(data_dir, data_file, seed, nb_class, nb_examples, nb_nodes, dataset, nb_master_nodes, opt):
    """
    Get a dataset based on the options.
    :param opt:
    :return:
    """
    if dataset == 'random':
        logging.info("Getting a random dataset")
        dataset = RandomDataset(seed, nb_class, nb_examples, nb_nodes)
    elif dataset == 'tcga-tissue':
        logging.info("Getting TCGA tissue type")
        dataset = TCGATissue(seed=seed, nb_class=nb_class, nb_examples=nb_examples, nb_nodes=nb_nodes, nb_master_nodes=nb_master_nodes)
    elif opt.dataset == 'tcga-tissue-gene-inference':
        logging.info("TCGA tissue gene inference")
        dataset = TCGAGeneInference(seed=seed, nb_class=nb_class, nb_examples=nb_examples, nb_nodes=nb_nodes, nb_master_nodes=nb_master_nodes)

    else:
        raise ValueError
    return dataset


def subsample_graph(adj, percentile=100):
    # if we want to sub-sample the edges, based on the edges value
    if percentile < 100:
        # small trick to ignore the 0.
        nan_adj = np.ma.masked_where(adj == 0., adj)
        nan_adj = np.ma.filled(nan_adj, np.nan)

        threshold = np.nanpercentile(nan_adj, 100 - percentile)
        logging.info("We will remove all the edges that has a value smaller than {}".format(threshold))

        to_keep = adj >= threshold  # throw away all the edges that are bigger than what we have.
        return adj * to_keep


class InpaintingGraph(object):
    def __init__(self, graph, keep_original=True):
        self.graph = graph
        self.epsilon = 1e-8
        self.keep_original = keep_original

    def __call__(self, sample):

        # Could do node2vec or something here.
        inputs, labels = sample
        nb_nodes = inputs.shape[0]

        to_predict_idx = np.random.randint(nb_nodes)

        to_predict = np.zeros(inputs.shape)
        to_predict[to_predict_idx] = inputs[to_predict_idx] + self.epsilon

        inputs_supervised = inputs
        inputs_unsupervised = inputs.copy()
        inputs_unsupervised[to_predict_idx] = 0.

        if self.keep_original:
            return np.concatenate([inputs_supervised, inputs_unsupervised], -1), [labels, to_predict]
        else:
            return inputs_unsupervised, to_predict
