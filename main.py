import argparse
import datasets
import numpy as np
import models
import torch
import time
from torch.autograd import Variable
import os
import pickle
import monitoring

def accuracy(data, model, no_class = None, on_cuda=False):
    acc = 0.
    total = 0.

    for mini in data:

        inputs = Variable(mini['sample'], requires_grad=False).float()
        targets = Variable(mini['labels'], requires_grad=False).float()

        if on_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        #import ipdb; ipdb.set_trace()

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



def build_parser():
    parser = argparse.ArgumentParser(
        description="Model for convolution-graph network (CGN)")

    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=1993, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=100, type=int, help="The batch size.")
    parser.add_argument('--tensorboard', default='./testing123/', help='The folder where to store the experiments. Will be created if not already exists.')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=0., type=float, help='weight decay (L2 loss).')
    parser.add_argument('--l1-loss', default=0., type=float, help='L1 loss.')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--data-dir', default='/data/milatmp1/dutilfra/transcriptome/graph/', help='The folder contening the dataset.')
    parser.add_argument('--dataset', choices=['random', 'tcga-tissue', 'tcga-brca', "tcga-label", 'percolate'], default='random', help='Which dataset to use.')
    parser.add_argument('--clinical-file', type=str, default='PANCAN_clinicalMatrix.gz', help='File to read labels from')
    parser.add_argument('--clinical-label', type=str, default='gender', help='Label to join with data')
    parser.add_argument('--scale-free', action='store_true', help='If we want a scale-free random adjacency matrix for the dataset.')
    parser.add_argument('--cuda', action='store_true', help='If we want to run on gpu.')
    parser.add_argument('--norm-adj', action='store_true', help="If we want to normalize the adjancy matrix.")
    parser.add_argument('--make-it-work-for-Joseph', action='store_true', help="Don't store anything in tensorboard, otherwise a segfault can happen.")
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")

    # Model specific options
    parser.add_argument('--num-channel', default=32, type=int, help='Number of channel in the model.')
    parser.add_argument('--skip-connections', action='store_true', help='If we want to add skip connection from every layer to the last.')
    parser.add_argument('--model', default='cgn', choices=['cgn', 'mlp', 'lcg', 'sgc'], help='Number of channel in the CGN.')
    parser.add_argument('--num-layer', default=1, type=int, help='Number of convolution layer in the CGN.')
    parser.add_argument('--nb-class', default=None, type=int, help="Number of class for the dataset (won't work with random graph).")
    parser.add_argument('--nb-examples', default=None, type=int, help="Number of samples to train on.")
    parser.add_argument('--nb-per-class', default=None, type=int, help="Number of samples per class.")
    parser.add_argument('--train-ratio', default=0.8, type=float, help="The ratio of data to be used in the training set.")
    parser.add_argument('--percentile', default=100, type=float, help="How many edges to keep.")
    parser.add_argument('--add-self', action='store_true', help="Add self references in the graph.")
    parser.add_argument('--attention-layer', default=0, type=int, help="The number of attention layer to add to the last layer. Only implemented for CGN.")
    parser.add_argument('--prune-graph', action='store_true', help="If we want to prune the graph.")
    parser.add_argument('--use-emb', default=None, type=int, help="If we want to add node embeddings.")

    return parser

def parse_args(argv):

    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt

def main(argv=None):

    opt = parse_args(argv)

    batch_size = opt.batch_size
    epoch = opt.epoch
    seed = opt.seed
    learning_rate = opt.lr
    weight_decay = opt.weight_decay
    momentum = opt.momentum
    on_cuda = opt.cuda
    tensorboard = opt.tensorboard
    nb_examples = opt.nb_examples
    nb_per_class = opt.nb_per_class
    train_ratio = opt.train_ratio
    l1_loss = opt.l1_loss # TODO: add

    # The experiment unique id.
    param = vars(opt).copy()
    # Removing a bunch of useless tag
    del param['data_dir']
    del param['tensorboard']
    del param['cuda']
    del param['make_it_work_for_Joseph']
    del param['train_ratio']
    del param['epoch']
    del param['batch_size']
    del param['clinical_file']
    del param['clinical_label']
    v_to_delete = []
    for v in param:
        if param[v] is None:
            v_to_delete.append(v)
    for v in v_to_delete:
        del param[v]

    exp_name = '_'.join(['{}={}'.format(k, v) for k, v, in param.iteritems()])
    print vars(opt)

    # seed
    if on_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    # creating the dataset
    print "Getting the dataset..."
    dataset, nb_class = datasets.get_dataset(opt)

    # dataset loader
    train_set, valid_set, test_set = datasets.split_dataset(dataset, batch_size=batch_size, seed=seed,
                                                            nb_samples=nb_examples, train_ratio=train_ratio, nb_per_class=nb_per_class)

    # Creating a model
    print "Getting the model..."
    my_model = models.get_model(opt, dataset, nb_class)
    print "Our model:"
    print my_model

    # Train the cgn
    criterion = torch.nn.CrossEntropyLoss(size_average=True)
    #optimizer = torch.optim.SGD(my_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if on_cuda:
        print "Putting the model on gpu..."
        my_model.cuda()

    # For tensorboard
    writer = None
    exp_dir = None
    if not opt.make_it_work_for_Joseph:
        from logger import Logger

        if not os.path.exists(tensorboard):
            os.mkdir(tensorboard)

        exp_dir = os.path.join(tensorboard, exp_name)
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)

        # dumping the options
        pickle.dump(opt, open(os.path.join(exp_dir, 'options.pkl'), 'wb'))
        writer = Logger(exp_dir)
        print "We will log everything in ", exp_dir
    else:
        print "Nothing will be log, everything will only be shown on screen."

    # The training.
    for t in range(epoch):

        start_timer = time.time()

        for no_b, mini in enumerate(train_set):

            inputs, targets = mini['sample'], mini['labels']

            inputs = Variable(inputs, requires_grad=False).float()
            targets = Variable(targets, requires_grad=False).long()
            #print targets.sum(dim=1)

            if on_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = my_model(inputs).float()

            # Compute and print loss
            loss = criterion(y_pred, targets)

            if epoch == 1:
                print "Done minibatch {}".format(no_b)
                print(t, loss.data[0])

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Add some metric for tensorboard
        # Loss
        if writer is not None:
            writer.scalar_summary('loss', loss.data[0], t)

        # time
        time_this_epoch = time.time() - start_timer

        if writer is not None:
            writer.scalar_summary('time', time_this_epoch, t)

        # compute the metrics for all the sets, for all the classes. right now it's precision/recall/f1-score, for train and valid.
        acc = {}
        for my_set, set_name in zip([train_set, valid_set, test_set], ['train', 'valid']):#, 'tests']):
            acc[set_name] = accuracy(my_set, my_model, on_cuda=on_cuda)

            if writer is not None:
                writer.scalar_summary('accuracy_{}'.format(set_name), acc[set_name], t)

            # accuracy for a different class
            metric_per_class = compute_metrics_per_class(my_set, my_model, nb_class, lambda x: dataset.labels_name(x), on_cuda=on_cuda)

            if writer is not None:
                for m, value in metric_per_class.iteritems():
                   for cl, v in value.iteritems():
                        writer.scalar_summary('{}/{}/{}'.format(m, set_name, cl), v, t) # metric/set/class

        # small summary.
        print "epoch {}, loss: {:.03f}, precision train: {:0.2f} precision valid: {:0.2f}, time: {:.02f} sec".format(t,
                                                                                                         loss.data[0],
                                                                                                         acc['train'],
                                                                                                         acc['valid'],
                                                                                                         time_this_epoch)

    print "Done!"

    if not opt.make_it_work_for_Joseph:
        monitoring.monitor_everything(my_model, dataset, opt, exp_dir)

if __name__ == '__main__':
    main()
