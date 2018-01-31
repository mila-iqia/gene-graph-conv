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
from metrics import accuracy, recall, f1_score, precision, compute_metrics_per_class, auc


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
    parser.add_argument('--dataset', choices=['random', 'tcga-tissue', 'tcga-brca', 'tcga-label', 'tcga-gbm', 'percolate'], default='random', help='Which dataset to use.')
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
    parser.add_argument('--model', default='cgn', choices=['cgn', 'mlp', 'lcg', 'sgc', 'slr', 'cnn'], help='Number of channel in the CGN.')
    parser.add_argument('--num-layer', default=1, type=int, help='Number of convolution layer in the CGN.')
    parser.add_argument('--nb-class', default=None, type=int, help="Number of class for the dataset (won't work with random graph).")
    parser.add_argument('--nb-examples', default=None, type=int, help="Number of samples to train on.")
    parser.add_argument('--nb-per-class', default=None, type=int, help="Number of samples per class.")
    parser.add_argument('--train-ratio', default=0.8, type=float, help="The ratio of data to be used in the training set.")
    parser.add_argument('--percentile', default=100, type=float, help="How many edges to keep.")
    parser.add_argument('--add-self', action='store_true', help="Add self references in the graph.")
    parser.add_argument('--attention-layer', default=0, type=int, help="The number of attention layer to add to the last layer. Only implemented for CGN.")
    parser.add_argument('--pool-graph', default=None, choices=['random', 'grid'], help="If we want to pool the graph.")
    parser.add_argument('--use-emb', default=None, type=int, help="If we want to add node embeddings.")
    parser.add_argument('--use-gate', default=0., type=float, help="The lambda for the gate pooling/striding. is ignore if = 0.")
    parser.add_argument('--lambdas', default=[], type=float, nargs='*', help="A list of lambda for the specified models.")

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
    del param['attention_layer']
    del param['clinical_label']
    del param['nb_per_class']
    del param['lambdas']
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
    dataset = datasets.get_dataset(opt)

    # dataset loader
    train_set, valid_set, test_set = datasets.split_dataset(dataset, batch_size=batch_size, seed=seed,
                                                            nb_samples=nb_examples, train_ratio=train_ratio, nb_per_class=nb_per_class)
    nb_class = dataset.nb_class
    # Creating a model
    print "Getting the model..."
    my_model = models.get_model(opt, dataset)
    print "Our model:"
    print my_model

    # Train the cgn
    criterion = torch.nn.CrossEntropyLoss(size_average=True)
    l1loss = torch.nn.L1Loss()
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

            # The l1 loss
            # l1_crit = torch.nn.L1Loss(size_average=False)
            # reg_loss = 0
            # for param in my_model.parameters():
            #     reg_loss += l1_crit(param, torch.FloatTensor(0.))
            # other_loss = l1_loss * reg_loss

            # Compute and print loss
            cross_loss = criterion(y_pred, targets)
            other_loss = sum([r * l for r, l in zip(my_model.regularization(), opt.lambdas)])
            total_loss = cross_loss + other_loss


            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Add some metric for tensorboard
        # Loss
        if writer is not None:
            writer.scalar_summary('cross_loss', cross_loss.data[0], t)
            # writer.scalar_summary('other_loss', other_loss.data[0], t)
            writer.scalar_summary('total_loss', total_loss.data[0], t)

        # time
        time_this_epoch = time.time() - start_timer

        if writer is not None:
            writer.scalar_summary('time', time_this_epoch, t)

        # compute the metrics for all the sets, for all the classes. right now it's precision/recall/f1-score, for train and valid.
        acc = {}
        auc_dict = {}
        for my_set, set_name in zip([train_set, valid_set, test_set], ['train', 'valid']):#, 'tests']):
            acc[set_name] = accuracy(my_set, my_model, on_cuda=on_cuda)
            #auc_dict[set_name] = auc(my_set, my_model, on_cuda=on_cuda)

            if writer is not None:
                writer.scalar_summary('accuracy_{}'.format(set_name), acc[set_name], t)

            # accuracy for a different class
            metric_per_class = compute_metrics_per_class(my_set, my_model, nb_class, lambda x: dataset.labels_name(x), on_cuda=on_cuda)

            if writer is not None:
                for m, value in metric_per_class.iteritems():
                   for cl, v in value.iteritems():
                        writer.scalar_summary('{}/{}/{}'.format(m, set_name, cl), v, t) # metric/set/class

        # small summary.
        print "epoch {}, cross_loss: {:.03f}, total_loss: {:.03f}, precision_train: {:0.3f} precision_valid: {:0.3f}, time: {:.02f} sec".format(
            t,
            cross_loss.data[0],
            total_loss.data[0],
            acc['train'],
            acc['valid'],
            #auc_dict['train'],
            #auc_dict['valid'],
            time_this_epoch)

    print "Done!"

    if not opt.make_it_work_for_Joseph:
        monitoring.monitor_everything(my_model, valid_set, opt, exp_dir)

if __name__ == '__main__':
    main()
