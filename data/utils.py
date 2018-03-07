import logging
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from graph import Graph
from gene_datasets import BRCACoexpr, GBMDataset, TCGATissue, NSLRSyntheticDataset
from datasets import RandomDataset, PercolateDataset


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


def get_dataset(opt):
    """
    Get a dataset based on the options.
    :param opt:
    :return:
    """
    graph = Graph(opt)

    if opt.dataset == 'random':
        logging.info("Getting a random graph")
        dataset = RandomDataset(graph, opt)

    elif opt.dataset == 'tcga-tissue':
        logging.info("Getting TCGA tissue type")
        dataset = TCGATissue(graph, opt)

    elif opt.dataset == 'tcga-brca':
        logging.info("Getting TCGA BRCA type")
        dataset = BRCACoexpr(graph, opt)

    elif opt.dataset == 'percolate':
        dataset = PercolateDataset(graph, opt)

    elif opt.dataset == 'tcga-gbm':
        logging.info("Getting TCGA GBM Dataset")
        dataset = GBMDataset(graph=graph, opt=opt)

    elif opt.dataset == 'nslr-syn':
        logging.info("Getting NSLR Synthetic Dataset")
        dataset = NSLRSyntheticDataset(graph, opt)

    elif opt.dataset == 'percolate-plus':
        logging.info("Getting percolate-plus Dataset")
        pdata = PercolateDataset(graph, opt)
        dataset = Graph.add_noise(dataset=pdata, num_added_nodes=opt.extra_ucn)

    else:
        raise ValueError
    return dataset
