from .arguments_base import ArgumentsBase


class ArgumentsTrainVal(ArgumentsBase):
    def __init__(self):
        super(ArgumentsTrainVal, self).__init__()

        parser = self.parser
        parser.add_argument('-fg', '--flag', type=str, default='train')

        # log info
        parser.add_argument('-logf', '--log-freq', type=int,
                            default=100, help='log frequency over iterations')

        # dataset loading info
        parser.add_argument('-vb', '--validation-batch-size',
                            type=int, default=256, help='validation batch size')

        # model save info
        parser.add_argument('-cptf', '--checkpoint-freq', default=1, type=int,
                            help='the frequency of saving model over epoches')

        # training control parameters
        parser.add_argument('-lr', '--learning-rate', required=True, type=float, help='learning rate')
        parser.add_argument('-me', '--maxepoch', required=True,
                            type=int, help='the number of epochs to train')
        parser.add_argument('-re', '--resume-epoch', type=int,
                            default=0, help='the epoch model resume from')
        parser.add_argument('-ri', '--resume-iteration', type=int,
                            default=0, help='the iteration model resume from')

        # model info
        parser.add_argument('-rf', '--restore-file', default=None,
                            help='resume model file', metavar='FILE')

        # log files info
        parser.add_argument('-lcd', '--check-log-dir', metavar='DIR', required=True,
                            help='checkpoints and logs directory')
