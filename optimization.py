import torch
from data import utils
from torch.autograd import Variable

def get_criterion(dataset):
    return torch.nn.CrossEntropyLoss(size_average=True)


def compute_loss(criterions, y_pred, targets):
    targets = Variable(targets, requires_grad=False).long()
    return criterions(y_pred, targets)
