from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--load_dir', type=str, default='/data/chwang/Log/ordinal',
                            help='models are loaded from here')
        self.isTrain = False
        return parser
