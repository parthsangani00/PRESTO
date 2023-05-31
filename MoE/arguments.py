import argparse

def myArgParse(parse_args=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action = 'store_true')
    parser.add_argument('--eval', action = 'store_true')
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--state', type = int, default = -1)
    parser.add_argument('--cuda', type = int, default = '-1')
    parser.add_argument('--cluster_init', type = str, default = None) #EqKMeans / FL
    parser.add_argument('--n_bins', type = int, default = 4)
    parser.add_argument('--ckpt', type = str, default = 'default_ckpt')
    parser.add_argument('--baseline', action = 'store_true')
    parser.add_argument('--clustering_baseline', action = 'store_true')
    parser.add_argument('--dataset', type = str, default = 'pathmnist')
    parser.add_argument('--lee_way', type = int, default = 3000)
    parser.add_argument('--get_param_count', action='store_true')

    if parse_args:
        args = parser.parse_args()
        return args

    else:
        return parser