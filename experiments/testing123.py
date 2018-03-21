import argparse
def build_parser():
    parser = argparse.ArgumentParser(
        description="Model for convolution-graph network (CGN)")

    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=1993, type=int, help='Seed for random initialization and stuff.')
    return parser


def parse_args(argv):
    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv
    return opt



def main(argv=None):
    opt = parse_args(argv)

    print "opt!: ", opt

if __name__ == '__main__':
    main()