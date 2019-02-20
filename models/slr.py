import logging
from models.models import Model
from models.utils import *
import torch.nn.functional as F
import torch
from torch import nn

class SLR(Model):
    def __init__(self, **kwargs):
        super(SLR, self).__init__(**kwargs)

    def setup_layers(self):
        self.nb_nodes = self.X.shape[1]
        self.in_dim = 1
        self.out_dim = 2
        self.adj.setdiag(0.)
        self.laplacian = torch.FloatTensor(norm_laplacian(self.adj).toarray())

        # The logistic layer.
        logistic_in_dim = self.nb_nodes * self.in_dim
        logistic_layer = nn.Linear(logistic_in_dim, self.out_dim)
        logistic_layer.register_forward_hook(save_computations)
        self.my_logistic_layers = nn.ModuleList([logistic_layer])

        torch.manual_seed(self.seed)
        if self.on_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            self.cuda()

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(nb_examples, -1)
        x = self.my_logistic_layers[-1](x)
        return x

    def regularization(self, reg_lambda):
        laplacian = torch.Variable(self.laplacian, requires_grad=False)
        if self.on_cuda:
            laplacian = laplacian.cuda()
        weight = self.my_logistic_layers[-1].weight
        reg = torch.abs(weight).mm(laplacian) * torch.abs(weight)
        return reg.sum() * reg_lambda
