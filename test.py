import torch.nn as nn

from options.test_options import TestOptions
from utils.Dataset import *
from utils.test_data import *
from utils.tools import *
from utils.train_data import *

if __name__ == '__main__':
    # -----  Loading the init options -----
    opt = TestOptions().parse()
    model = define_Cls(opt.cls_type, class_num=opt.class_num, init_type=opt.init_type, init_gain=opt.init_gain, m=opt.m,
                       gpu_ids=opt.gpu_ids)

    # criterion preparation
    criterion = nn.CrossEntropyLoss()

    # dataset preparation
    test_dataset = Dataset(mode="test")
    nacc_dataset = Dataset(mode="nacc")

    # test loader (Internal ADNI testing set)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=int(opt.workers / 2), pin_memory=True)
    # nacc loader (External NACC testing set)
    nacc_loader = torch.utils.data.DataLoader(
        nacc_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=int(opt.workers / 2), pin_memory=True)

    # model loading
    state_dict = torch.load(opt.load_dir)
    model.load_state_dict(state_dict, strict=False)
    # ema prototype
    model.prototypes = state_dict['prototypes']
    model.cuda()
    print("loading weights from {}".format(opt.load_dir))
    # test data on the internal ADNI testing set
    print("Testing on the internal ADNI testing set")
    test_data(model, test_loader, criterion)

    print("Testing on the external NACC testing set")
    test_data(model, nacc_loader, criterion)
