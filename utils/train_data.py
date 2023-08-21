import time

import numpy as np
import torch
import wandb
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, precision_score
from tqdm import tqdm


def train_data(model, total_cn_loader, total_ad_loader, total_mci_loader,
               valid_dataloaders, epochs, optimizer, scheduler,
               basiccomputing, criterion, criterionRank,
               expr_dir, print_freq, save_epoch_freq,
               ):
    '''
    train process
    :param model: the corresponding model
    :param total_cn_loader: the cn train loader
    :param total_ad_loader: the ad train loader
    :param total_mci_loader: the mci train loader
    :param valid_dataloaders: the smci/pmci valid loader
    :param epochs: epochs
    :param optimizer: the optimizer
    :param scheduler: the scheduler
    :param basiccomputing: basic computing component
    :param criterion: CE criterion
    :param criterionRank: Rank criterion
    :param expr_dir: the saved directory
    :param print_freq: print frequency
    :param save_epoch_freq: saving frequency
    :return:
    '''
    start = time.time()
    steps = 0
    for e in tqdm(range(1, epochs + 1)):
        model.train()
        train_loss = 0.
        train_loss_CE = 0.
        train_loss_ins2ins = 0.
        train_loss_ins2cls = 0.
        train_loss_cls2cls = 0.
        train_loss_hyb = 0.
        y_train_true = []
        y_train_pred = []
        cn_iterator = iter(total_cn_loader)
        mci_iterator = iter(total_mci_loader)
        for ii, (imgs_ad, labels_ad) in enumerate(tqdm(total_ad_loader)):
            steps += 1
            try:
                imgs_cn, labels_cn = next(cn_iterator)
                imgs_mci, labels_mci = next(mci_iterator)
            except StopIteration:
                cn_iterator = iter(total_cn_loader)
                mci_iterator = iter(total_mci_loader)
                imgs_cn, labels_cn = next(cn_iterator)
                imgs_mci, labels_mci = next(mci_iterator)

            imgs = torch.cat((imgs_cn, imgs_mci, imgs_ad))
            labels = torch.cat((labels_cn, labels_mci, labels_ad))
            images = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            features, outputs, _ = model.forward(images)

            # basic computing
            compactness_loss, separation_loss, mus = basiccomputing(features, labels)

            # CE loss
            loss_CE = criterion(outputs, labels)

            # Hybrid-granularity ordinal loss
            loss_ins2ins = criterionRank(features, labels)  # instance-to-instance loss
            loss_ins2cls = compactness_loss / features.shape[1]  # instance-to-class loss
            loss_cls2cls = features.shape[1] / separation_loss + criterionRank(mus, torch.tensor(
                [0, 1, 2]).cuda())

            loss_hyb = loss_ins2ins + loss_ins2cls + loss_cls2cls

            # total loss
            lambda_hyb = e * (1 / epochs)
            loss = loss_CE + lambda_hyb * loss_hyb

            # backward
            loss.backward()
            optimizer.step()

            # prototype online update
            model.update(features, labels)

            # loss logging
            train_loss_CE += loss_CE.item()
            train_loss_ins2ins += loss_ins2ins.item()
            train_loss_ins2cls += loss_ins2cls.item()
            train_loss_cls2cls += loss_cls2cls.item()
            train_loss_hyb += loss_hyb.item()
            train_loss += loss.item()

            _, train_predicted = torch.max(outputs.data, 1)
            y_train_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
            y_train_pred.extend(np.ravel(np.squeeze(train_predicted.cpu().detach().numpy())).tolist())

        if scheduler:
            scheduler.step()

        val_loss = 0.
        y_val_true = []
        y_val_pred = []
        val_prob_all = []
        with torch.no_grad():
            model.eval()
            for ii, (images, labels) in enumerate(tqdm(valid_dataloaders)):
                images, labels = images.cuda(), labels.cuda()
                _, _, outputs = model(images)
                _, val_predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                y_val_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
                y_val_pred.extend(np.ravel(np.squeeze(val_predicted.cpu().detach().numpy())).tolist())
                val_prob_all.extend(outputs[:, 1].cpu().detach().numpy())

        # loss logging
        train_loss_CE = train_loss_CE / len(total_ad_loader)
        train_loss_ins2ins = train_loss_ins2ins / len(total_ad_loader)
        train_loss_ins2cls = train_loss_ins2cls / len(total_ad_loader)
        train_loss_cls2cls = train_loss_cls2cls / len(total_ad_loader)
        train_loss_hyb = train_loss_hyb / len(total_ad_loader)
        train_loss = train_loss / len(total_ad_loader)
        val_loss = val_loss / len(valid_dataloaders)
        train_acc = accuracy_score(y_train_true, y_train_pred)
        val_acc = accuracy_score(y_val_true, y_val_pred)
        val_f1_score = f1_score(y_val_true, y_val_pred, average='weighted')
        val_recall = recall_score(y_val_true, y_val_pred, average='weighted')
        val_spe = recall_score(y_val_true, y_val_pred, pos_label=0, average='binary')
        val_precision = precision_score(y_val_true, y_val_pred, average='weighted')
        val_auc = roc_auc_score(y_val_true, val_prob_all, average='weighted')
        train_f1_score = f1_score(y_train_true, y_train_pred, average='weighted')
        train_recall = recall_score(y_train_true, y_train_pred, average='weighted')
        if e % print_freq == 0:
            wandb.log({"train_loss": train_loss,
                       "train_loss_hyb": train_loss_hyb,
                       "train_loss_CE": train_loss_CE,
                       "train_loss_ins2ins": train_loss_ins2ins,
                       "train_loss_ins2cls": train_loss_ins2cls,
                       "train_loss_cls2cls": train_loss_cls2cls,
                       "train_acc": train_acc,
                       "train_f1": train_f1_score,
                       "train_sen": train_recall,
                       "val_loss": val_loss,
                       "val_acc": val_acc, "val_f1": val_f1_score,
                       "val_sen": val_recall, "val_spe": val_spe,
                       "val_precision": val_precision, "val_auc": val_auc
                       })
            print('Epochs: {}/{}...'.format(e + 1, epochs),
                  'Trian Loss:{:.3f}...'.format(train_loss),
                  'Trian Loss_CE:{:.3f}...'.format(train_loss_CE),
                  'Trian Loss_hyb:{:.3f}...'.format(train_loss_hyb),
                  'Trian Loss_ins2ins:{:.3f}...'.format(train_loss_ins2ins),
                  'Trian Loss_ins2cls:{:.3f}...'.format(train_loss_ins2cls),
                  'Trian Loss_cls2cls:{:.3f}...'.format(train_loss_cls2cls),
                  'Trian Accuracy:{:.3f}...'.format(train_acc),
                  'Trian F1 Score:{:.3f}...'.format(train_f1_score),
                  'Trian SEN:{:.3f}...'.format(train_recall),
                  'Val Loss:{:.3f}...'.format(val_loss),
                  'Val Accuracy:{:.3f}...'.format(val_acc),
                  'Val F1 Score:{:.3f}'.format(val_f1_score),
                  'val SPE:{:.3f}...'.format(val_spe),
                  'Val SEN:{:.3f}...'.format(val_recall),
                  'Val AUC:{:.3f}...'.format(val_auc),
                  "val_precision:{:.3f}...".format(val_precision)
                  )
        if e % save_epoch_freq == 0:
            torch.save(model.state_dict(), expr_dir + '/{}_net.pth'.format(e))

    end = time.time()
    runing_time = end - start
    print('Training time is {:.0f}m {:.0f}s'.format(runing_time // 60, runing_time % 60))
