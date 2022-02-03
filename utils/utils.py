import argparse
from pprint import pprint


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='../configs/mutag.json',
        help='The Configuration file')

    argparser.add_argument('--dev', type=str, default='cuda', help='')

    argparser.add_argument('--n_graph', type=int, default=4, help='num of grpahs')
    argparser.add_argument('--seed', type=int, default=1, help='random seeds')
    argparser.add_argument('--name', type=str, default='graphon', help='types of graph',
                           choices=['er0.01', 'er0.05', 'er0.1', 'sbm', 'graphon', 'graphonPW2', 'graphonPW3'])
    argparser.add_argument('--sample_method', type=str, default='grid_sample', choices=['grid_sample', 'random_sample', 'grid_Gsample', 'random_Gsample'])
    argparser.add_argument('--estimate', action='store_true', help='estimate edge probability')

    # n_graph = 10, name = 'er', sample_method = 'grid_Gsample', estimate = False

    args = argparser.parse_args()
    return args

def process_config2(config, args):
    keys = ['n_graph', 'name', 'sample_method', 'estimate', 'seed']
    args = vars(args)
    for k in keys:
        config[k] = args[k]
    return config
