import torch
import torch.nn as nn


class BasicComputing(nn.Module):
    def __init__(self, class_num, gpu_ids, dim=512):
        super(BasicComputing, self).__init__()
        self.dim = dim
        self.class_num = class_num

    # compute current prototype
    def compute_mean(self, x):
        return torch.mean(x, dim=0, keepdim=True)

    # compute instance-to-class or the first term of class-to-class loss
    def compute_loss(self, x, mean):
        xg_bar = x - mean  # [None,d]
        if len(xg_bar.shape) == 1:
            xg_bar = xg_bar.unsqueeze(0)
        return torch.sum(
            torch.matmul(xg_bar.unsqueeze(dim=1), xg_bar.unsqueeze(dim=2))
            , dim=0).reshape(-1)

    def __call__(self, features, labels):
        compactness_losslist = []
        separation_losslist = []
        all_means = []
        class_index = 0
        # [0] basic definition multi-class
        for i in range(self.class_num):
            index = torch.nonzero(labels == i).squeeze()
            mu_k = features.index_select(0, index)
            if mu_k.shape[0] > 0:
                class_index += 1
                all_means.append(self.compute_mean(mu_k))
                compactness_losslist.append(self.compute_loss(mu_k, self.compute_mean(mu_k)))
                separation_losslist.append(self.compute_loss(self.compute_mean(mu_k),
                                                             self.compute_mean(features)) * mu_k.shape[0])

        compactness_loss = sum(compactness_losslist)
        separation_loss = sum(separation_losslist)
        return compactness_loss, separation_loss, torch.stack(all_means, dim=0)
