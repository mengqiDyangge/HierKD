from .resnet import ResNet, BasicBlock, Bottleneck
from ..builder import BACKBONES

@BACKBONES.register_module()
class ResNetImg(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
                 depth,
                 in_channels,
                 stem_channels,
                 base_channels,
                 num_stages,
                 strides,
                 dilations,
                 out_indices,
                 style,
                 deep_stem,
                 avg_down,
                 frozen_stages,
                 conv_cfg,
                 norm_cfg,
                 norm_eval,
                 dcn,
                 stage_with_dcn,
                 plugins,
                 with_cp,
                 zero_init_residual,
                 pretrained,
                 init_cfg)
    
    def forward(self, x):
        """Forward function."""
        # if self.training:
        #     outs = [x]
        # else:
        #     outs = []
            
        outs = [x]
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class ResNetV1dImg(ResNetImg):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1dImg, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)