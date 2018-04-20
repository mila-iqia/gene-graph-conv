import torch
from data import utils
from torch.autograd import Variable

def get_criterion(dataset, training_mode=None):
    criterions = None

    if training_mode is None:
        criterions = torch.nn.CrossEntropyLoss(size_average=True)
    if training_mode == 'semi':
        print "Adding a semi supervised loss."
        criterions = [torch.nn.CrossEntropyLoss(size_average=True), torch.nn.MSELoss(size_average=True)]
        #do transform
        dataset.transform = utils.InpaintingGraph(dataset.graph)
    if training_mode == 'unsupervised':
        print "Adding a unsupervised loss."
        criterions = torch.nn.MSELoss(size_average=True)
        #do transform
        dataset.transform = utils.InpaintingGraph(dataset.graph, keep_original=False)

    if training_mode == 'gene-inference':
        criterions = torch.nn.MSELoss(size_average=True)

    return criterions



def compute_loss(criterions, y_pred, targets, training_mode=None, semi_mse_lambda=0):

    if training_mode is None:
        targets = Variable(targets, requires_grad=False).long()
        return criterions(y_pred, targets)
    elif training_mode == 'unsupervised':

        targets_uns = Variable(targets, requires_grad=False)
        mse = criterions
        to_check = targets_uns != 0.
        y_pred = y_pred * to_check.float()
        #import ipdb; ipdb.set_trace()
        mse_loss = mse(y_pred.float(), targets_uns.float()) * y_pred.size(1)
        return mse_loss.mean()

    elif training_mode == 'semi':

        targets_sup = Variable(targets[0], requires_grad=False).long()
        targets_semi = Variable(targets[1], requires_grad=False)

        ce, mse = criterions
        ce_loss = ce(y_pred[0].float(), targets_sup)

        to_check = targets_semi != 0.
        y_pred[1] = y_pred[1] * to_check.float()
        mse_loss = mse(y_pred[1].float(), targets_semi.float())
        return ce_loss.mean() + semi_mse_lambda * mse_loss.mean()

    elif training_mode == 'gene-inference':

        #import ipdb; ipdb.set_trace()

        targets_uns = Variable(targets, requires_grad=False).squeeze(-1)
        mse = criterions
        to_check = targets_uns != 0.
        y_pred = y_pred * to_check.float()
        #import ipdb; ipdb.set_trace()
        mse_loss = mse(y_pred.float(), targets_uns.float()) * y_pred.size(1)
        return mse_loss.mean()
