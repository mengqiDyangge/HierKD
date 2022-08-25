# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import softmax_focal_loss as _softmax_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss

def py_softmax_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None,
                          pos_inds=None,
                          negative_sample=None,
                          sample_per=None,
                          iters=None):
    # important to add reduction='none' to keep per-batch-item loss
    ce_loss = F.cross_entropy(pred.contiguous(), target, reduction='none') 
    pt = torch.exp(-ce_loss)
    loss = (alpha * (1 - pt) ** gamma * ce_loss)

    # if negative_sample and (iters // 5 <= 6736):
    if negative_sample:
        if len(pos_inds) > 0:
            # sample negative
            all_inds = torch.tensor([True] * len(target)).to(target.device)
            all_inds[pos_inds] = False
            nega_inds = torch.where(all_inds == True)[0]
            if sample_per == 0: # posi: nega = 1:1
                random_inds = torch.randperm(len(nega_inds))[:len(pos_inds)]
            else: # sample 0.x negatives
                random_inds = torch.randperm(len(nega_inds))[:int(sample_per * len(nega_inds))]
            sample_nega_inds = nega_inds[random_inds]
            loss_inds = torch.cat((pos_inds, sample_nega_inds), dim=0)
            loss = loss[loss_inds]
            weight = weight[loss_inds]

            if weight is not None:
                if weight.shape != loss.shape:
                    if weight.size(0) == loss.size(0):
                        weight = weight.view(-1, 1)
                    else:
                        assert weight.numel() == loss.numel()
                        weight = weight.view(loss.size(0), -1)
                assert weight.ndim == loss.ndim
            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
            return loss
        else:
            return pred.sum() * 0
    else:
        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    weight = weight.view(-1, 1)
                else:
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)
            assert weight.ndim == loss.ndim
            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss


def softmax_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _softmax_focal_loss(pred.contiguous(), target, gamma, alpha, None,
                               'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class SoftMaxFocalLoss(nn.Module):

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(SoftMaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.iter = 0

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                pos_inds=None,
                negative_sample=None,
                sample_per=None):
        self.iter += 1

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # # there is a bug in mmcv, so we adopt pytorch version
        # if torch.cuda.is_available() and pred.is_cuda:
        #     calculate_loss_func = softmax_focal_loss

        # loss_cls = self.loss_weight * calculate_loss_func(
        #     pred,
        #     target,
        #     weight,
        #     gamma=self.gamma,
        #     alpha=self.alpha,
        #     reduction=reduction,
        #     avg_factor=avg_factor)
        
        loss_cls = self.loss_weight * py_softmax_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor,
            pos_inds=pos_inds,
            negative_sample=negative_sample,
            sample_per=sample_per,
            iters=self.iter)

        return loss_cls
