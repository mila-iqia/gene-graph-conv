import numpy as np
from torch.autograd import Variable


def format_mini(mini, model, on_cuda):
    inputs = Variable(mini['sample'], requires_grad=False).float()
    targets = Variable(mini['labels'], requires_grad=False).float()

    if on_cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()

    targets = targets.data.cpu().long().numpy()
    preds = model(inputs).max(dim=1)[1].data.cpu().long().numpy()
    return preds, targets


def accuracy(data, model, no_class = None, on_cuda=False):
    acc = 0.
    total = 0.

    for mini in data:
        preds, targets = format_mini(mini, model, on_cuda)

        id_to_keep = np.ones_like(targets)
        if no_class is not None:
            id_to_keep = targets == no_class

        acc += ((targets == preds) * id_to_keep).sum()
        total += sum(id_to_keep)

    acc = acc / float(total)
    return acc

def recall(preds, gts, cl):
    """How many revelant item are selected?"""
    tmp = ((gts == preds) * (gts == cl)).sum()
    total = sum(gts == cl)
    return tmp / float(total)

def precision(preds, gts, cl):
    """How many selected item are revelant?"""
    tmp = ((gts == preds) * (gts == cl)).sum()
    total = sum(cl == preds)
    return tmp / float(total)

def f1_score(preds, gts, cl):
    re = recall(preds, gts, cl)
    pre = precision(preds, gts, cl)
    return 2 * re * pre / (re + pre)

def auc(preds, targets, cl):
    #import pdb; pdb.set_trace()
    pass

def I(pred, target):
    #import pdb; pdb.set_trace()
    pass

def compute_metrics_per_class(data, model, nb_class, idx_to_str, on_cuda=False,
                     metrics_foo={'recall': recall,
                                  'precision': precision,
                                  'f1_score': f1_score,
                                  'auc': auc}):

    metrics = {k: {} for k in metrics_foo.keys()}

    # Get the predictions
    all_preds, all_targets = [], []
    for mini in data:
        preds, targets = format_mini(mini, model, on_cuda)
        all_preds = np.concatenate([all_preds, preds])
        all_targets = np.concatenate([all_targets, targets])

    # Get the class specific
    for cl in range(nb_class):
        for i, m in metrics_foo.iteritems():
            metrics[i][idx_to_str(cl)] = m(all_preds, all_targets, cl)

    return metrics
