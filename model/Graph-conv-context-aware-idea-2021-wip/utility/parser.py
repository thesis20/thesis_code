import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CSGCN.")
    parser.add_argument('--load', type=int, default=1,
                        help='1 = Load saved data. 0 = Save new data.')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Seed for randoms.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--dataset', nargs='?', default='yelpnc',
                        help='Choose a dataset from {ml100k, ml1m, frappe, yelpnc, yelpon}')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of layers.')
    parser.add_argument('--mess_dropout', nargs='?', default='[1.0, 1.0, 1.0, 1.0]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 0: no dropout.')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='Learning rate.')
    parser.add_argument('--decay', type=float, default=1e-5,
                        help='Decay for BPR.')
    parser.add_argument('--keep_prob', type=float, default=1.0,
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--ks', nargs='?', default='[20, 50]',
                        help='Top k(s) to recommend')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Interval between evaluations (epochs)')
    parser.add_argument('--initializer', nargs='?', default='xavier',
                    help='Choose an initializer from {xavier, normal, glorot, glorot_normal}')
    parser.add_argument('--optimizer', nargs='?', default='adam',
                        help='Choose an optimizer from {adam, adagrad, RMSProp, Adadelta}')
    parser.add_argument('--eval_method', nargs='?', default='fold',
                        help='Choose an evaluation method from {fold, loo}')
    return parser.parse_args()