import argparse


class ArgumentsBase(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        parser = self.parser
        parser.add_argument('-ddir', '--data-directory',
                            help='data directory of the image dataset', metavar='DIR')

        # gpu ids
        parser.add_argument('-gs', '--gpu-ids', type=int, nargs='+',
                            default=None, help='ids of gpu devices to train the network')

        # the data loading setting
        parser.add_argument('-ds', '--dataset-name', type=str,
                            help='dataset name used for training')
        parser.add_argument('-b', '--batch-size', required=True,
                            type=int, help='mini-batch size')
        parser.add_argument('-nw', '--num-workers', default=8,
                            type=int, help='workers for loading data synchronously')

        # model info
        parser.add_argument('-model', '--model-name', type=str, required=True,
                            help='model name which defines the structure of model backbones')

        # training/testing configuration version
        parser.add_argument('-vs', '--version', type=str, default='baseline',
                            help='defines model architecture, training settings and metrics')

    def parse_args(self):
        return self.parser.parse_args()
