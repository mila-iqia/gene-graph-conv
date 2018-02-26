import numpy as np
from torch.autograd import Variable
from sklearn import metrics


def format_mini(mini, model, on_cuda):
    inputs = Variable(mini['sample'], requires_grad=False).float()
    targets = Variable(mini['labels'], requires_grad=False).float()

    if on_cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
    targets = targets.data.cpu().numpy()
    preds = model(inputs)
    return preds, targets


def accuracy(data, model, no_class=None, on_cuda=False):
    acc = 0.
    total = 0.

    for mini in data:
        preds, targets = format_mini(mini, model, on_cuda)
        preds = preds.max(dim=1)[1].data.cpu().long().numpy()
        targets = targets.astype(long)
        id_to_keep = np.ones_like(targets)
        if no_class is not None:
            id_to_keep = targets == no_class

        acc += ((targets == preds) * id_to_keep).sum()
        total += sum(id_to_keep)

    acc = acc / float(total)
    return acc


def auc(data, model, no_class=None, on_cuda=False):
    all_preds, all_targets = [], []

    for mini in data:
        preds, targets = format_mini(mini, model, on_cuda)
        preds = [x[1] for x in preds.data.cpu().numpy()]
        all_preds = np.concatenate([all_preds, preds])
        all_targets = np.concatenate([all_targets, targets])

    return metrics.roc_auc_score(all_targets, all_preds)


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


def compute_metrics_per_class(data, model, nb_class, idx_to_str, on_cuda=False,
                              metrics_foo={'recall': recall,
                                           'precision': precision,
                                           'f1_score': f1_score}):

    metrics = {k: {} for k in metrics_foo.keys()}

    # Get the predictions
    all_preds, all_targets = [], []
    for mini in data:
        preds, targets = format_mini(mini, model, on_cuda)
        preds = preds.max(dim=1)[1].data.cpu().long().numpy()
        all_preds = np.concatenate([all_preds, preds])
        all_targets = np.concatenate([all_targets, targets])

    # Get the class specific
    for cl in range(nb_class):
        for i, m in metrics_foo.iteritems():
            metrics[i][idx_to_str(cl)] = m(all_preds, all_targets, cl)

    return metrics


def record_metrics_for_epoch(writer, cross_loss, total_loss, t, time_this_epoch, train_set, valid_set, test_set, my_model, nb_class, dataset, on_cuda):
    # Add some metric for tensorboard
    # Loss
    if writer is not None:
        writer.scalar_summary('cross_loss', cross_loss.data[0], t)
        # writer.scalar_summary('other_loss', other_loss.data[0], t)
        writer.scalar_summary('total_loss', total_loss.data[0], t)

    if writer is not None:
        writer.scalar_summary('time', time_this_epoch, t)

    # compute the metrics for all the sets, for all the classes. right now it's precision/recall/f1-score, for train and valid.
    acc = {}
    auc_dict = {}
    for my_set, set_name in zip([train_set, valid_set, test_set], ['train', 'valid', 'test']):
        acc[set_name] = accuracy(my_set, my_model, on_cuda=on_cuda)
        auc_dict[set_name] = auc(my_set, my_model, on_cuda=on_cuda)

        if writer is not None:
            writer.scalar_summary('accuracy_{}'.format(set_name), acc[set_name], t)

        # accuracy for a different class
        metric_per_class = compute_metrics_per_class(my_set, my_model, nb_class, lambda x: dataset.labels_name(x), on_cuda=on_cuda)

        if writer is not None:
            for m, value in metric_per_class.iteritems():
                for cl, v in value.iteritems():
                    writer.scalar_summary('{}/{}/{}'.format(m, set_name, cl), v, t)  # metric/set/class
    return acc, auc_dict


def summarize(epoch, cross_loss, total_loss, accuracy, auc):
    summary = {
        "epoch": epoch,
        "cross_loss": cross_loss,
        "total_loss": total_loss,
        "accuracy": accuracy,
        "auc": auc
    }
    return summary
