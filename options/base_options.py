import argparse

from .option_utils import *


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--drop_ratio', default=0,
                            help='Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1')

        parser.add_argument('--dataset', type=str, default='smci+pmci', help='dataset')
        parser.add_argument('--group', type=str, default='smci+pmci', help='group type')
        parser.add_argument('--gpu_ids', default='7', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')

        parser.add_argument('--class_num', type=int, default=3,
                            help='the number of class')
        parser.add_argument('--cls_type', type=str, default='resnet3d')
        parser.add_argument('--m', type=float, default=0.999, help='ema momentum decay for prototype update scheme')
        parser.add_argument('--checkpoints_dir', type=str, default='/data/chwang/Log/ordinal',
                            help='models are saved here')
        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test
        if self.isTrain:
            self.print_options(opt)
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

        print(opt.gpu_ids)
        self.opt = opt
        return self.opt
