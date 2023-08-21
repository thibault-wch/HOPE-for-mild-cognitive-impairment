import torch
from models.Resnet import *
from scipy import ndimage
from torch import optim
from torch.nn import init
from torch.optim import lr_scheduler


def get_scheduler(optimizer, opt):
    """
    scheduler definition
    :param optimizer:  original optimizer
    :param opt: corresponding parameters
    :return: corresponding scheduler
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    elif opt.lr_policy == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    """
    initialize the network weights
    :param net: the network
    :param init_type:  initialized method
    :param gain: corresponding gain
    :return: the initialized network
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=''):
    """
    initial the network
    :param net:  to be initialized network
    :param init_type:  initialized method
    :param gain: corresponding gain
    :param gpu_ids: the gpu ids
    :return: the initialized network
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net)
        net.cuda()
    init_weights(net, init_type, gain=init_gain)
    return net


def define_Cls(netCls, class_num=4, init_type='normal', init_gain=0.02, m=0.99, gpu_ids=[]):
    """
    define the corresponding network
    :param netCls: define type
    :param class_num: the class number
    :param init_type:  initialized method
    :param gain: corresponding gain
    :param m: the momentum decay value for online prototype update scheme
    :param gpu_ids: the gpu ids
    :return: the initialized network
    """
    if netCls == 'resnet3d':
        net = resnet18(spatial_size=128, sample_duration=128, num_classes=class_num, m=m)
    return init_net(net, init_type, init_gain, gpu_ids)


def resize_volume(img, desired_depth, desired_width, desired_height):
    '''
    resize the images
    :param img: original image
    :param desired_depth: the desired depth number
    :param desired_width: the desired width number
    :param desired_height: the desired height number
    :return: resized image
    '''

    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]

    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height

    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    return img
