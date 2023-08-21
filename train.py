import os

from models.BasicComputing import BasicComputing
from models.ranking import RankLoss
from options.train_options import TrainOptions
from utils.Dataset import *
from utils.tools import *
from utils.train_data import *

if __name__ == '__main__':
    # -----  Loading the init options -----
    opt = TrainOptions().parse()
    wandb.init(project="your project",
               entity="your entity",
               name=opt.name,
               config={
                   "batch_size": opt.batch_size,
                   "group": opt.group,
                   "dataset": opt.dataset,
                   "learning_rate": opt.lr,
                   "architecture": opt.cls_type,
                   "epoch": opt.epoch_count

               })
    model = define_Cls(opt.cls_type, class_num=opt.class_num, init_type=opt.init_type, init_gain=opt.init_gain, m=opt.m,
                       gpu_ids=opt.gpu_ids)
    epochs = opt.epoch_count
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = get_scheduler(optimizer, opt)

    # criterion preparation
    basiccomputing = BasicComputing(class_num=opt.class_num, gpu_ids=opt.gpu_ids)
    criterion = nn.CrossEntropyLoss()
    criterionRank = RankLoss(opt.interpolation_lambda)

    # dataset preparation
    total_cn_dataset = Dataset(mode="total_cn")
    total_ad_dataset = Dataset(mode="total_ad")
    total_mci_dataset = Dataset(mode="total_mci")
    valid_dataset = Dataset(mode="valid")

    # training loader (random sample data in a stratified manner)
    total_cn_loader = torch.utils.data.DataLoader(
        total_cn_dataset, batch_size=int(opt.batch_size / 4), shuffle=True,
        num_workers=int(opt.workers / 4), pin_memory=True, drop_last=True)
    total_ad_loader = torch.utils.data.DataLoader(
        total_ad_dataset, batch_size=int(opt.batch_size / 4), shuffle=True,
        num_workers=int(opt.workers / 4), pin_memory=True, drop_last=True)
    total_mci_loader = torch.utils.data.DataLoader(
        total_mci_dataset, batch_size=int(opt.batch_size / 2), shuffle=True,
        num_workers=int(opt.workers / 2), pin_memory=True, drop_last=True)

    # valid loader
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=int(opt.workers / 2), pin_memory=True)

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)

    # train data
    train_data(model, total_cn_loader, total_ad_loader, total_mci_loader,
               valid_loader, epochs, optimizer, scheduler,
               basiccomputing, criterion, criterionRank, expr_dir, opt.print_freq,
               opt.save_epoch_freq)
    wandb.finish()
