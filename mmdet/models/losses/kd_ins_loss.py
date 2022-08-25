# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

from ..builder import LOSSES


@LOSSES.register_module()
class KD_L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(KD_L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        distance = torch.abs(pred - target).sum(dim=-1).mean()
        loss_dist = self.loss_weight * distance
        return loss_dist


@LOSSES.register_module()
class KD_L2Loss(nn.Module):
    """L2 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(KD_L2Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        distance = (pred - target).norm(dim=-1).mean()
        loss_dist = self.loss_weight * distance
        return loss_dist

@LOSSES.register_module()
class Symmetric_CLLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(Symmetric_CLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self,
                text_img_feats,
                text_feats,
                tem,
                mask,
                type='caption'):
        device = text_feats.device
        if type == 'caption':
            text_img_match = tem * torch.bmm(text_img_feats, text_feats.unsqueeze(2)).squeeze(-1)
            img_text_match = text_img_match.t()
            target = torch.LongTensor(range(len(text_img_feats))).to(device)
            loss_dist = self.loss_weight * (self.ce(img_text_match, target) + self.ce(text_img_match, target))
        elif type == 'noun_phrase':
            text_img_match = tem * torch.bmm(text_img_feats, text_feats.unsqueeze(2)).squeeze(-1)
            text_img_match = text_img_match.reshape(text_img_feats.size(1), -1, text_img_feats.size(1)).permute(0, 2, 1)
            text_img_match = F.log_softmax(text_img_match, dim=1)
            mask = mask.unsqueeze(1)
            text_img_match = mask * text_img_match
            num_phrase = torch.sum(mask, dim=-1, keepdim=True)
            text_img_match = text_img_match / (num_phrase + 1e-10)
            text_img_match = text_img_match.sum(dim=-1)
            img_text_match = text_img_match.t()
            loss_dist = -(text_img_match.diag().mean() + img_text_match.diag().mean())
        else:
            raise ValueError('Not implement')
        return loss_dist

@LOSSES.register_module()
class PLLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self,
                img_text_match,
                text_img_match,
                target,
                tem):
        positive_pair = -img_text_match[range(len(img_text_match)), range(len(img_text_match))] / tem
        loss_dist = self.loss_weight * positive_pair.mean()
        return loss_dist