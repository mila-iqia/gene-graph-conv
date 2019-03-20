import logging
from models.models import Model
from models.utils import *
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np


class MLP(Model):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

    def setup_layers(self):
        self.out_dim = len(np.unique(self.y))
        in_dim = self.X.shape[1]
        dims = [in_dim] + self.channels
        logging.info("Constructing the network...")

        layers = []
        for c_in, c_out in zip(dims[:-1], dims[1:]):
            layer = nn.Linear(c_in, c_out)
            layers.append(layer)
        self.my_layers = nn.ModuleList(layers)

        if self.channels:
            self.last_layer = nn.Linear(self.channels[-1], self.out_dim)
        else:
            self.last_layer = nn.Linear(in_dim, self.out_dim)

        self.my_dropout = torch.nn.Dropout(0.5) if self.dropout else None
        logging.info("Done!")

        torch.manual_seed(self.seed)
        if self.on_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            self.cuda()

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.permute(0, 2, 1).contiguous()  # from ex, node, ch, -> ex, ch, node
        for layer in self.my_layers:
            x = F.relu(layer(x.view(nb_examples, -1)))  # or relu, sigmoid...
            if self.dropout:
                x = self.my_dropout(x)
        x = self.last_layer(x.view(nb_examples, -1))
        return x
