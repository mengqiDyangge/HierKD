# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .anchor_head import AnchorHead
from .retina_head import RetinaHead


@HEADS.register_module()
class RetinaZSDHead(RetinaHead):
    r"""modified RetinaHead for Zero-shot Detection

    the last conv layer in class branch from 3 * 3 * 256 * K * A to 3 * 3 * 256 * d_f(dim of feature) * A
    which will used to calculate cosine simlarity with fixed word embedding initialize from CLIP text encoder

    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.visual_channels = kwargs.get('visual_channels', None)
        self.word_embedding = kwargs.get('word_embedding', None)
        self.base_classes = kwargs.get('base_classes', None)
        self.target_classes = kwargs.get('target_classes', None)
        # if align global
        self.dist_featuremap = kwargs.get('dist_featuremap', None)
        # if align instance
        self.dist_instance = kwargs.get('dist_instance', None)
        # if validation on target class
        self.zero_shot = kwargs.get('zero_shot', None)
        self.loss_cls_all = kwargs.get('loss_cls_all', None)
        
        del kwargs['visual_channels']
        del kwargs['word_embedding']
        del kwargs['base_classes']
        del kwargs['target_classes'] 
        del kwargs['dist_featuremap']
        del kwargs['dist_instance']
        del kwargs['zero_shot']
        del kwargs['loss_cls_all']

        super().__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        
        # self.loss_cls = FocalLoss()
        

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.visual_channels,
            3,
            padding=1)
        
        self.word_embedding = torch.load(self.word_embedding, 'cpu')
        self.base_embedding = self.word_embedding['base']
        self.target_embedding = self.word_embedding['target']

        self.base_linear = nn.Linear(512, self.base_classes, bias=False)
        self.base_linear.weight = self.base_embedding
        self.base_linear.weight.requires_grad = False

        self.target_linear = nn.Linear(512, self.target_classes, bias=False)
        self.target_linear.weight = self.target_embedding
        self.target_linear.weight.requires_grad = False

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_feat = self.retina_cls(cls_feat)

        # b * c * h * w -> b* h * w * c
        b ,_ , h, w = cls_feat.shape
        cls_feat = cls_feat.permute(0, 2, 3, 1)
        # b * h * w * c -> b* h * w * num_anchors * visual_channels
        cls_feat = cls_feat.view(b, h , w, self.num_anchors, -1).contiguous()

        # word embedding already normlizaed, visual feature also should normalized
        cls_feat = cls_feat / (cls_feat.norm(dim=-1, keepdim=True))
        
        if self.training:
            cls_score = self.base_linear(cls_feat)
        else:
            if self.zero_shot:
                cls_score = self.target_linear(cls_feat)
            else:
                cls_score = self.base_linear(cls_feat)
        logit_scale = self.logit_scale.exp()
        cls_score = logit_scale * cls_score
        cls_score = cls_score.view(b, h, w, -1).contiguous()
        cls_score = cls_score.permute(0, 3, 1, 2)

        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        if self.loss_cls_all:
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=cls_score.shape[0])
        else:
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox
