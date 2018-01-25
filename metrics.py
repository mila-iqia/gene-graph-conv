import numpy as np
from torch.autograd import Variable


def accuracy(data, model, no_class = None, on_cuda=False):
    acc = 0.
    total = 0.

    for mini in data:

        inputs = Variable(mini['sample'], requires_grad=False).float()
        targets = Variable(mini['labels'], requires_grad=False).float()

        if on_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        if len(targets.size()) > 2:
            max_index_target = targets.max(dim=1)[1].data.cpu().long().numpy()
        else:
            max_index_target = targets.data.cpu().long().numpy()

        max_index_pred = model(inputs).max(dim=1)[1].data.cpu().long().numpy()


        id_to_keep = np.ones_like(max_index_target)
        if no_class is not None:
            id_to_keep = max_index_target == no_class

        acc += ((max_index_target == max_index_pred) * id_to_keep).sum()
        total += sum(id_to_keep)

    acc = acc / float(total)
    return acc

def recall(preds, gts, cl):

    # How many revelant item are selected?

    ids_from_that_class = gts == cl # ids_to_keep total number in class

    tmp = ((gts == preds) * ids_from_that_class).sum()
    total = sum(ids_from_that_class)
    return tmp / float(total)

def precision(preds, gts, cl):

    # How many selected item are revelant?

    ids_from_that_class = gts == cl  # total number predicted for that class

    tmp = ((gts == preds) * ids_from_that_class).sum()
    total = sum(cl == preds)
    return tmp / float(total)

def f1_score(preds, gts, cl):

    re = recall(preds, gts, cl)
    pre = precision(preds, gts, cl)

    return 2 * re * pre / (re + pre)

# TODO: move all of that to it's own file
def compute_metrics_per_class(data, model, nb_class, idx_to_str, on_cuda=False,
                     metrics_foo={'recall': recall,
                                  'precision': precision,
                                  'f1_score': f1_score}):

    metrics = {k: {} for k in metrics_foo.keys()}

    all_target = None
    all_pred = None

    # Get the predictions
    for mini in data:

        inputs = Variable(mini['sample'], requires_grad=False).float()
        targets = Variable(mini['labels'], requires_grad=False).float()

        if on_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()


        if len(targets.size()) > 2:
            max_index_target = targets.max(dim=1)[1].data.cpu().long().numpy()
        else:
            max_index_target = targets.data.cpu().long().numpy()

        max_index_pred = model(inputs).max(dim=1)[1].data.cpu().long().numpy()

        if all_target is None:
            all_target = max_index_target
        else:
            all_target = np.concatenate([all_target, max_index_target])

        if all_pred is None:
            all_pred = max_index_pred
        else:
            all_pred = np.concatenate([all_pred, max_index_pred])

    # Get the class specific
    for cl in range(nb_class):

        for i, m in metrics_foo.iteritems():
            metrics[i][idx_to_str(cl)] = m(all_pred, all_target, cl)

    return metrics
