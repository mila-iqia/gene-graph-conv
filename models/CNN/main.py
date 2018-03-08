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

from model import CNN_naive
import torch
from torch.autograd import Variable

if __name__=='__main__':
    model = CNN_naive()

    
    fmy = h5py.File("test.hdf5", "r")
    data = fmy["expression_data"]
    labels = fmy["labels_data"]
    

    correct = 0
    for i in range(0,data.shape[0]):
        x = torch.FloatTensor(16*16).copy_(torch.from_numpy(data[i,:]))
        x.resize_(1,1,16,16)
        x = Variable(x)
        y = model(x)
        print y.data[0], labels[i]
        if y.data[0] == labels[i]:
            correct += 1
    
    print 'Percentage correct = ', correct/100.0
    
    
