import torch
import torch.nn as nn
import torch.nn.functional as F


class RankLoss(nn.Module):
    def __init__(self, lambda_val):
        super(RankLoss, self).__init__()
        self.lambda_val = lambda_val

    def __call__(self, x, y):
        # [0] setup
        # features shape [N,D]             : x
        # targets shape  [N,1]             : y
        y = y.unsqueeze(1)

        # [1] similarity computation
        xxt = torch.matmul(F.normalize(x.view(x.size(0), -1)), F.normalize(x.view(x.size(0), -1)).permute(1, 0))

        # [2] compute ranking similarity loss
        loss = 0
        for i in range(len(y)):
            label_ranks = rank_normalised(-torch.abs(y[i] - y).transpose(0, 1))
            feature_ranks = TrueRanker.apply(xxt[i].unsqueeze(dim=0), self.lambda_val)
            loss += F.mse_loss(feature_ranks, label_ranks)
        return loss / y.shape[0]


def rank(seq):
    return torch.argsort(torch.argsort(seq).flip(1))


def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]


# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Code is based on the Blackbox Combinatorial Solvers (https://github.com/martius-lab/blackbox-backprop) implementation
# from https://github.com/martius-lab/blackbox-backprop by Marin Vlastelica et al.

class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None
