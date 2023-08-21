import time

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, precision_score
from tqdm import tqdm


def test_data(model, test_dataloaders, criterion):
    '''
    test process
    :param model: the corresponding model
    :param test_dataloaders: the smci/pmci test loader
    :param criterion: CE criterion
    :return:
    '''
    start = time.time()
    val_loss = 0.
    y_val_true = []
    y_val_pred = []
    val_prob_all = []
    with torch.no_grad():
        model.eval()
        for ii, (images, labels) in enumerate(tqdm(test_dataloaders)):
            images, labels = images.cuda(), labels.cuda()
            _, _, outputs = model(images)
            _, val_predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            y_val_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
            y_val_pred.extend(np.ravel(np.squeeze(val_predicted.cpu().detach().numpy())).tolist())
            val_prob_all.extend(outputs[:, 1].cpu().detach().numpy())

    # loss logging
    val_loss = val_loss / len(test_dataloaders)
    val_acc = accuracy_score(y_val_true, y_val_pred)
    val_f1_score = f1_score(y_val_true, y_val_pred, average='weighted')
    val_recall = recall_score(y_val_true, y_val_pred, average='weighted')
    val_spe = recall_score(y_val_true, y_val_pred, pos_label=0, average='binary')
    val_precision = precision_score(y_val_true, y_val_pred, average='weighted')
    val_auc = roc_auc_score(y_val_true, val_prob_all, average='weighted')

    print(
        'Val Loss:{:.3f}...'.format(val_loss),
        'Val Accuracy:{:.3f}...'.format(val_acc),
        'Val F1 Score:{:.3f}'.format(val_f1_score),
        'val SPE:{:.3f}...'.format(val_spe),
        'Val SEN:{:.3f}...'.format(val_recall),
        'Val AUC:{:.3f}...'.format(val_auc),
        "val_precision:{:.3f}...".format(val_precision)
    )

    end = time.time()
    runing_time = end - start
    print('Testing time is {:.0f}m {:.0f}s'.format(runing_time // 60, runing_time % 60))
