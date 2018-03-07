import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque
from tqdm import tqdm
import argparse
import h5py


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Percolation dataset')
    parser.add_argument('--dataset', help='Dataset filename')
    args = parser.parse_args()

    fmy = h5py.File(args.dataset,"r")
    data = fmy["expression_data"]
    labels = fmy["labels_data"]
    for i in range(0, data.shape[0]):
        print np.sum(data[i,:]), labels[i]