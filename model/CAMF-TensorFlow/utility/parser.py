import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CAMF")
    parser.add_argument('--load', type=int, default=1,
                        help='1 = Load saved data. 0 = Save new data.')
    parser.add_argument('--seed', type=int, default=42069,
                        help='Seed for randoms.')
    parser.add_argument('--embed_size', type=int, default=10,
                        help='Embedding size.')
    parser.add_argument('--dataset', nargs='?', default='ml100k',
                        help='Choose a dataset from {ml100k, ml1m, frappe}')
    parser.add_argument('--epoch', type=int, default=2500,
                        help='Number of epochs.')
    parser.add_argument('--batch', type=int, default=95,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='Learning rate.')
    parser.add_argument('--decay', type=float, default=1e-5,
                        help='Decay for BPR.')
    parser.add_argument('--ks', nargs='?', default='[20, 50]',
                        help='Top k(s) to recommend')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Interval between evaluations (epochs)')
    parser.add_argument('--initializer', nargs='?', default='xavier',
                    help='Choose an initializer from {xavier, normal, glorot, glorot_normal}')
    parser.add_argument('--optimizer', nargs='?', default='adam',
                        help='Choose an optimizer from {adam, adagrad, RMSProp, Adadelta, GradientDescent}')
    parser.add_argument('--eval_method', nargs='?', default='fold',
                        help='Choose an evaluation method from {fold, loo}')
    parser.add_argument('--alg_type', nargs='?', default='camf-c',
                    help='Choose an alg type from {camf-c, camf-ci}')
    return parser.parse_args()