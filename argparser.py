import argparse

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=0, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=10, type=int, help="The batch size.")
    parser.add_argument('--agg-reduce', default=2, type=int, help="amount to reduce the graph at each conv layer.")
    parser.add_argument('--tensorboard-dir', default='./experiments/experiments/', help='The folder where to store the experiments. Will be created if not already exists.')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=0., type=float, help='weight decay (L2 loss).')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--dataset', choices=['tcga'], default='random', help='Which dataset to use.')
    parser.add_argument('--cuda', action='store_true', help='If we want to run on gpu.')

    # Model specific options
    parser.add_argument('--channels', default=64, type=int, help='Number of channel in the model.')
    parser.add_argument('--dropout', default=True, type=bool, help='If we want to perform dropout in the model.')
    parser.add_argument('--add-connectivity', default=False, type=bool, help='If we want to augment the connectivity after each convolution layer after the first one.')
    parser.add_argument('--model', default='gcn', choices=['gcn', 'mlp', 'slr'], help='Which model to use.')
    parser.add_argument('--num-layer', default=3, type=int, help='Number of convolution layer in the GCN.')
    parser.add_argument('--train-size', default=50, type=int, help="Number of samples to train on.")
    parser.add_argument('--test-size', default=300, type=int, help="Number of samples to test on.")
    parser.add_argument('--train-ratio', default=0.8, type=float, help="The ratio of data to be used in the training set.")
    parser.add_argument('--aggregation', default='hierarchy', choices=['kmeans', 'hierarchy', 'random'], help="If we want to pool the graph.")
    parser.add_argument('--embedding', default=64, type=int, help="If we want to add node embeddings.")
    parser.add_argument('--graph', default='genemania', choices=['genemania', 'regnet'], help="Which graph with which to prior")
    return parser


def parse_args(argv):
    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv
    return opt
