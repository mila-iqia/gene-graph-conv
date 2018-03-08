import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
import os


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        print type(layer)
        print layer.weight.size()[1]
        if layer.weight.size()[1] == 1:
            # upper horizontal
            layer.weight.data[0, 0, 0, 0] = 1.0
            layer.weight.data[0, 0, 0, 1] = 1.0
            layer.weight.data[0, 0, 1, 0] = 0.0
            layer.weight.data[0, 0, 1, 1] = 0.0
            # left vertical
            layer.weight.data[1, 0, 0, 0] = 1.0
            layer.weight.data[1, 0, 0, 1] = 0.0
            layer.weight.data[1, 0, 1, 0] = 1.0
            layer.weight.data[1, 0, 1, 1] = 0.0
            # lower horizontal
            layer.weight.data[2, 0, 0, 0] = 0.0
            layer.weight.data[2, 0, 0, 1] = 0.0
            layer.weight.data[2, 0, 1, 0] = 1.0
            layer.weight.data[2, 0, 1, 1] = 1.0
            # right vertical
            layer.weight.data[3, 0, 0, 0] = 0.0
            layer.weight.data[3, 0, 0, 1] = 1.0
            layer.weight.data[3, 0, 1, 0] = 0.0
            layer.weight.data[3, 0, 1, 1] = 1.0

            layer.bias.data.fill_(-1.5)


def init_weights_naive(layer):
    if type(layer) == nn.Conv2d:
        layer.weight.data.fill_(0.0)
        if layer.weight.size()[1] == 1:
            layer.weight.data[0, 0, 0, 0] = 1.0
            layer.weight.data[0, 0, 0, 1] = 1.0
            layer.bias.data[0] = -1.5

            layer.weight.data[1, 0, 1, 0] = 1.0
            layer.weight.data[1, 0, 1, 1] = 1.0
            layer.bias.data[1] = -1.5

            layer.weight.data[2, 0, 1, 0] = 1.0
            layer.weight.data[2, 0, 1, 1] = 1.0
            layer.weight.data[2, 0, 0, 0] = 1.0
            layer.bias.data[2] = -2.5

            layer.weight.data[3, 0, 0, 1] = 1.0
            layer.weight.data[3, 0, 1, 1] = 1.0
            layer.weight.data[3, 0, 0, 0] = 1.0
            layer.bias.data[3] = -2.5

            layer.weight.data[4, 0, 0, 0] = 1.0
            layer.weight.data[4, 0, 0, 1] = 1.0
            layer.weight.data[4, 0, 1, 0] = 1.0
            layer.bias.data[4] = -2.5

            layer.weight.data[5, 0, 1, 0] = 1.0
            layer.weight.data[5, 0, 0, 1] = 1.0
            layer.weight.data[5, 0, 1, 1] = 1.0
            layer.bias.data[5] = -2.5

            layer.weight.data[6, 0, 0, 0] = 1.0
            layer.weight.data[6, 0, 1, 0] = 1.0
            layer.weight.data[6, 0, 0, 1] = 1.0
            layer.weight.data[6, 0, 1, 1] = 1.0
            layer.bias.data[6] = -3.5


class CNN_naive(nn.Module):
    def __init__(self):
        super(CNN_naive, self).__init__()

        self.cnn = nn.Conv2d(1, 7, kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()

        init_weights_naive(self.cnn)

    def forward(self, x):
        x_dim = x.size(2)
        n = 4
        for i in range(0, 4):
            y = self.relu(self.cnn(x))
            y = y * 2.0
            x = torch.unsqueeze(torch.max(y, dim=1)[0], dim=0)

        return torch.squeeze(x)
