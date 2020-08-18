from .arguments_base import ArgumentsBase


class ArgumentsTest(ArgumentsBase):
    def __init__(self):
        super(ArgumentsTest, self).__init__()

        parser = self.parser
        parser.add_argument('-fg', '--flag', type=str, default='test')

        # model info
        parser.add_argument('-rf', '--restore-file', required=True,
                            help='resume model file', metavar='FILE')
