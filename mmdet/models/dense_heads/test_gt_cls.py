# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import torch.nn as nn
import numpy as np
from ..clip import clip
import torch.nn.functional as F


from PIL import Image
from mmcv.runner import force_fp32
from ..builder import HEADS, build_loss
from .atss_zsd_sfl_head import ATSSZSDSFLHead
from mmdet.core import (anchor_inside_flags, bbox, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)
from torchvision.transforms import Resize, CenterCrop, Compose, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _transform():
    return Compose([
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

@HEADS.register_module()
class TestGtHead(ATSSZSDSFLHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='atss_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):

        super().__init__(
                 num_classes,
                 in_channels,
                 stacked_convs=stacked_convs,
                 conv_cfg=conv_cfg,
                 norm_cfg=norm_cfg,
                 loss_centerness=loss_centerness,
                 init_cfg=init_cfg,
            **kwargs)

        self.instances_unseen = {}
        with open('/home/zyma/python_work/CLIP/data/COCO_cls/val_unseen.txt', 'r') as f:
            for line in f.readlines():
                line = line.split()
                filename = line[0].split('_')[0] + '.jpg'
                if filename in self.instances_unseen.keys():
                    self.instances_unseen[filename].append([line[0], line[1], line[2]])    
                else:
                    self.instances_unseen[filename] = []
            
    def simple_test(self, feats, img_metas, rescale=False, **kwargs):
        # gt_lables = kwargs.get('gt_labels', None)
        # gt_bboxes = kwargs.get('gt_bboxes', None)
        # kwargs = {'gt_bboxes': gt_bboxes, 'gt_labels': gt_lables}
        return self.simple_test_bboxes(feats, img_metas, **kwargs)

    def simple_test_bboxes(self, feats, img_metas, **kwargs):
        img = feats[0]
        feats = feats[1:]
        gt_lables = kwargs.get('gt_labels', None)
        gt_bboxes = kwargs.get('gt_bboxes', None)
        import ipdb
        ipdb.set_trace()
        results_list = self.get_bboxes(img_metas, img, gt_bboxes, gt_lables)
        return results_list
    
    def transforms(self, x, n_px=224):
        _, _, h, w = x.shape
        scale = min(n_px / h, n_px / w)
        nh, nw = int(h * scale), int(w * scale)
        if nh == 0:
            nh = 1
        if nw == 0:
            nw = 1
        x = F.interpolate(x, size=(nh, nw), mode='bicubic', align_corners=False)
        w_p, h_p = 224 - nw, 224 - nh
        pad = (w_p // 2 + 1 if w_p % 2 != 0 else w_p // 2, w_p // 2,
            h_p // 2 + 1 if h_p % 2 != 0 else h_p // 2, h_p // 2)
        x_pad = F.pad(x, pad, mode='constant', value=0)
        return x_pad.squeeze(0)
            
    @force_fp32()
    def get_bboxes(self,
                   img_metas,
                   img,
                   gt_bboxes,
                   gt_labels):
        # self.clip, _ = clip.load("ViT-B/32")
        for i in range(len(img)):
            if gt_bboxes[i].shape[0] > 0:
                gt_bboxes[i] = gt_bboxes[i].view(-1, 4)
                
                img_ori = Compose([ToTensor(),Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(Image.open(img_metas[i]['filename'])).to(img.device)
                gt_bbox_ori = gt_bboxes[i] / (torch.tensor(img_metas[i]['scale_factor']).to(img.device).view(-1, 4))
                extract_img_ori = []
                for bbox in gt_bbox_ori.int():
                    extract_img_ori.append(self.extract_bbox(img_ori, bbox).unsqueeze(0)) 
                extract_img_ori = torch.cat(extract_img_ori, dim=0)
                clip_ve_feats_ori = self.clip.encode_image(extract_img_ori).float()
                clip_ve_feats_ori = clip_ve_feats_ori / (clip_ve_feats_ori.norm(dim=-1, keepdim=True) + 1e-10)
                clip_scores_ori = self.clip.logit_scale.exp() * self.target_linear(clip_ve_feats_ori)
                clip_scores_ori = clip_scores_ori.softmax(dim=-1).unsqueeze(0)

                # to do: load instances from 'CLIP/data/COCO_cls/val_unseen.txt'
                instances = []
                filename = img_metas[i]['filename'].split('/')[-1]
                filepath = '/home/zyma/python_work/CLIP/data/COCO_cls/val2017'
                for instance in self.instances_unseen[filename]:
                    ins = Compose([ToTensor(),Normalize((0.48145466, 0.4578275, 0.40821073), 
                                        (0.26862954, 0.26130258, 0.27577711))])(Image.open(os.path.join(filepath, instance[0]))).to(img.device)
                    instances.append(self.transforms(ins.unsqueeze(0)).unsqueeze(0))
                instances = torch.cat(instances, dim=0)
                clip_ve_feats_ins = self.clip.encode_image(instances).float()
                clip_ve_feats_ins = clip_ve_feats_ins / (clip_ve_feats_ins.norm(dim=-1, keepdim=True) + 1e-10)
                clip_scores_ins = self.clip.logit_scale.exp() * self.target_linear(clip_ve_feats_ins)
                clip_scores_ins = clip_scores_ins.softmax(dim=-1).unsqueeze(0)  

                extract_img = []
                for bbox in gt_bboxes[i].int():
                    extract_img.append(self.extract_bbox(img[0][:], bbox).unsqueeze(0))
                extract_img = torch.cat(extract_img, dim=0)
                clip_ve_feats = self.clip.encode_image(extract_img).float()
                clip_ve_feats = clip_ve_feats / (clip_ve_feats.norm(dim=-1, keepdim=True) + 1e-10)
                clip_scores = self.clip.logit_scale.exp() * self.target_linear(clip_ve_feats)
                clip_scores = clip_scores.softmax(dim=-1).unsqueeze(0)

                import ipdb
                ipdb.set_trace()  

        return 