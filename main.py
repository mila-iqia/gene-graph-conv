import argparse
import logging
import tensorflow as tf  # necessary to import here to avoid segfault
from data.utils import get_dataset, split_dataset
from data.graph import Graph, get_hash
from models.models import get_model, setup_l1_loss
import torch
import time
from torch.autograd import Variable
from analysis import monitoring
from analysis.metrics import record_metrics_for_epoch, summarize
import optimization as otim

def build_parser():
    parser = argparse.ArgumentParser(
        description="Model for convolution-graph network (CGN)")

    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=1993, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=100, type=int, help="The batch size.")
    parser.add_argument('--tensorboard-dir', default='./experiments/experiments/', help='The folder where to store the experiments. Will be created if not already exists.')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=0., type=float, help='weight decay (L2 loss).')
    parser.add_argument('--l1-loss-lambda', default=0., type=float, help='L1 loss lambda.')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--dataset', choices=['tcga-tissue', 'tcga-tissue-gene-inference'], default='tcga-tissue', help='Which dataset to use.')
    parser.add_argument('--cuda', action='store_true', help='If we want to run on gpu.')
    parser.add_argument('--norm-adj', default=True, type=bool, help="If we want to normalize the adjancy matrix.")
    parser.add_argument('--log', choices=['console', 'silent'], default='console', help="Determines what kind of logging you get")
    parser.add_argument('--name', type=str, default='testing123', help="If we want to add a random str to the folder.")
    parser.add_argument('--load-folder', type=str, default=None, help="Folder where to load the network and resume training.")
    parser.add_argument('--load-checkpoint', type=bool, default=False, help="Should we load the checkpoint?")
    parser.add_argument('--neighborhood', choices=['all', 'first', 'second'], default='all', help="Should we look at the full dataset, or neighborhood for the gene to infer?")
    parser.add_argument('--master-nodes', type=int, default=0, help="Number of master node to add in the graph. All nodes are connected to it.")

    # Model specific options
    parser.add_argument('--num-channel', default=32, type=int, help='Number of channel in the model.')
    parser.add_argument('--dropout', default=False, type=bool, help='If we want to perform dropout in the model..')
    parser.add_argument('--add-connectivity', default=False, type=bool, help='If we want to augment the connectivity after each convolution layer after the first one.')
    parser.add_argument('--model', default='cgn', choices=['cgn', 'mlp', 'lcg', 'sgc', 'slr', 'cnn', 'random', 'lr', 'cgn+mlp'], help='Which model to use.')
    parser.add_argument('--num-layer', default=1, type=int, help='Number of convolution layer in the CGN.')
    parser.add_argument('--nb-class', default=None, type=int, help="Number of class for the dataset (won't work with random graph).")
    parser.add_argument('--nb-examples', default=None, type=int, help="Number of samples to train on.")
    parser.add_argument('--nb-per-class', default=None, type=int, help="Number of samples per class.")
    parser.add_argument('--train-ratio', default=0.6, type=float, help="The ratio of data to be used in the training set.")
    parser.add_argument('--percentile', default=100, type=float, help="How many edges to keep.")
    parser.add_argument('--add-self', default=True, type=bool, help="Add self references in the graph.")
    parser.add_argument('--attention-layer', default=0, type=int, help="The number of attention layer to add to the last layer. Only implemented for CGN.")
    parser.add_argument('--pool-graph', default=None, choices=['ignore', 'hierarchy'], help="If we want to pool the graph.")
    parser.add_argument('--use-emb', default=None, type=int, help="If we want to add node embeddings.")
    parser.add_argument('--nb-attention-head', default=0, type=int, help="The number of attention head to use for graph network.")
    parser.add_argument('--use-gate', default=0., type=float, help="The lambda for the gate pooling/striding. is ignore if = 0.")
    parser.add_argument('--model_reg_lambda', default=0.0, type=float, nargs='*', help="A lambda for the regularization on a specific model.")
    parser.add_argument('--size-perc', default=4, type=int, help="The size of the connected percolate graph in percolate-plus datsaet")
    parser.add_argument('--extra-cn', default=0, type=int, help="The number of extra nodes with edges in the percolate-plus dataset.")
    parser.add_argument('--extra-ucn', default=0, type=int, help="The number of extra nodes without edges in the percolate-plus dataset")
    parser.add_argument('--disconnected', default=0, type=int, help="The number of disconnected nodes from the perc subgraph without edges in percolate-plus")
    parser.add_argument('--center', default=False, type=bool, help="center the data (subtract mean from each element)?")
    parser.add_argument('--graph', default='genemania', choices=['regnet', 'genemania'], help="Which graph with which to prior")
    parser.add_argument('--approx-nb-edges', default=100, type=int, help="If we have a randomly generated graph, this is the approx nb of edges")
    parser.add_argument('--nb-nodes', default=None, type=int, help="If we have a randomly generated graph, this is the nb of nodes")
    parser.add_argument('--data-dir', default=None, type=str, help="where is your dataset located?")
    parser.add_argument('--data-file', default=None, type=str, help="where is your dataset located?")
    return parser


def parse_args(argv):
    print argv
    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv
    return opt


def setup_logger(opt):
    logging.basicConfig(format="%(message)s")
    logger = logging.getLogger()
    if opt.log != 'silent':
        logger.setLevel('INFO')


def main(argv=None):
    opt = parse_args(argv)
    setup_logger(opt)
    logging.info(vars(opt))
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
    torch.manual_seed(opt.seed)

    logging.info("Getting the dataset...")
    dataset = get_dataset(opt.seed, opt.nb_class, opt.nb_examples, opt.nb_nodes, opt.dataset, opt.master_nodes, opt)
    train_set, valid_set, test_set = split_dataset(dataset, batch_size=opt.batch_size, seed=opt.seed,
                                                   nb_samples=opt.nb_examples, train_ratio=opt.train_ratio, nb_per_class=opt.nb_per_class)


    logging.info("Getting the graph...")
    graph = None
    if opt.graph == "random":
        graph = Graph()
        graph.load_random_adjacency(nb_nodes=opt.nb_nodes, approx_nb_edges=opt.approx_nb_edges, scale_free=opt.scale_free)
    elif opt.graph == 'train-corr':
        graph = Graph()
        graph.build_correlation_graph(train_set.dataset.data[train_set.sampler.indices])
    elif opt.graph is not None:
        graph = Graph()
        graph.load_graph(get_hash(opt.graph))

    # Adding the master nodes
    graph.add_master_nodes(opt.master_nodes)
    if graph is not None:
        graph.intersection_with(dataset)

    writer, exp_dir = monitoring.setup_tensorboard_log(opt)

    logging.info("Getting the model...")
    my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt, dataset, graph)

    logging.info("Our model:")
    logging.info(my_model)

    # Setup the loss
    criterions = otim.get_criterion(dataset)
    l1_criterion = torch.nn.L1Loss(size_average=False)

    if opt.cuda:
        logging.info("Putting the model on gpu...")
        my_model.cuda()

    max_valid = 0
    best_summary = {}
    patience = 20

    # The training.
    for t in range(epoch, opt.epoch):

        start_timer = time.time()
        for no_b, mini in enumerate(train_set):
            inputs, targets = mini['sample'], mini['labels']
            import pdb; pdb.set_trace()
            inputs = Variable(inputs, requires_grad=False).float()
            #targets = Variable(targets, requires_grad=False).long()

            if opt.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Forward pass: Compute predicted y by passing x to the model
            my_model.train()

            y_pred = my_model(inputs)

            # Compute and print loss
            crit_loss = otim.compute_loss(criterions, y_pred, targets)
            model_regularization_loss = my_model.regularization(opt.model_reg_lambda)
            l1_loss = setup_l1_loss(my_model, opt.l1_loss_lambda, l1_criterion, opt.cuda)
            total_loss = crit_loss + model_regularization_loss + l1_loss

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            my_model.eval()

        time_this_epoch = time.time() - start_timer

        acc, auc = record_metrics_for_epoch(writer, crit_loss, total_loss, t, time_this_epoch, train_set, valid_set, test_set, my_model, dataset, opt.cuda)
        summary = [
            t,
            crit_loss.data[0],
            acc['train'],
            acc['valid'],
            auc['train'],
            auc['valid'],
            time_this_epoch
        ]
        summary = "epoch {}, cross_loss: {:.03f}, acc_train: {:0.3f}, acc_valid: {:0.3f}, auc_train: {:0.3f}, auc_valid:{:0.3f}, time: {:.02f} sec".format(*summary)
        logging.info(summary)

        patience = patience - 1
        if patience == 0:
            break
        if max_valid < auc['valid'] and t > 5:
            max_valid = auc['valid']
            best_summary = summarize(t, crit_loss.data[0], total_loss.data[0], acc, auc)
            patience = 1000

        # Saving the checkpoint
        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)

    logging.info("Done!")
    #monitoring.monitor_everything(my_model, valid_set, opt, exp_dir)
    return best_summary

if __name__ == '__main__':
    main()
