# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
from .fpn import FPN


@NECKS.register_module()
class FPNImg(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(in_channels,
                         out_channels,
                         num_outs,
                         start_level,
                         end_level,
                         add_extra_convs,
                         relu_before_extra_convs,
                         no_norm_on_lateral,
                         conv_cfg,
                         norm_cfg,
                         act_cfg,
                         upsample_cfg,
                         init_cfg)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        # if self.training:
        img = inputs[0]
        inputs = inputs[1:]
        outs = super().forward(inputs)
        outs = [img] + [item for item in outs]
        # else:
        #     outs = super().forward(inputs)
        return tuple(outs)
