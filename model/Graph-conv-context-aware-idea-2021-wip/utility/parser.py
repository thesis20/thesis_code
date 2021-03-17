import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="CSGCN.")
    parser.add_argument('--seed', type=int, default=5,
                        help='Seed for randoms.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--dataset', nargs='?', default='ml100k',
                        help='Choose a dataset from {ml100k, ml100k}')
    parser.add_argument('--load', type=int, default=1,
                        help='1 = Load saved data. 0 = Save new data.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument('--layer_size', nargs='?', default='[64, 64, 64, 64]',
                        help='Output sizes of every layer')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.2, 0.1, 0.2]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--batch', type=int, default=95,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='Learning rate.')
    parser.add_argument('--decay', type=float, default=1e-5,
                        help='Decay for BPR.')
    parser.add_argument('--keep_prob', type=float, default='0.8',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--ks', nargs='?', default='[20, 50]',
                        help='Top k(s) to recommend')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Interval between evaluations (epochs)')
    parser.add_argument('--initializer', nargs='?', default='glorot',
                        help='Choose an initializer from {xavier, normal, glorot, glorot_normal}')
    parser.add_argument('--optimizer', nargs='?', default='adam',
                        help='Choose a dataset from {adam, adagrad, RMSProp, Adadelta}')
    return parser.parse_args()
