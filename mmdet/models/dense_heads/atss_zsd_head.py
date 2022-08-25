# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
import os
import json
from ..clip import clip
import torch.nn.functional as F

from torchvision.transforms import Resize, Compose, ToTensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import scale
BICUBIC = InterpolationMode.BICUBIC
from mmcv.runner import force_fp32
from ..builder import HEADS, build_loss
from .atss_head import ATSSHead
from mmdet.core import (anchor_inside_flags, bbox, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)

# def transforms(n_px=224):
#     # deformation resize
#     return Compose([
#         Resize(size=(n_px, n_px), interpolation=BICUBIC),
#         ToTensor()
#     ])

@HEADS.register_module()
class ATSSZSDHead(ATSSHead):
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
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.visual_channels = kwargs.get('visual_channels', None)
        self.word_embedding = kwargs.get('word_embedding', None)
        self.base_classes = kwargs.get('base_classes', None)
        self.target_classes = kwargs.get('target_classes', None)
        # if align global
        self.dist_featmap = kwargs.get('dist_featuremap', None)
        self.dist_featmap_loss = kwargs.get('dist_featuremap_loss', None)
        self.dist_feat_patch = kwargs.get('dist_featuremap_patch', None)
        self.dist_featmap_pool = kwargs.get('dist_featuremap_pool', None)
        self.dist_featmap_tem = kwargs.get('dist_featuremap_tem', None)
        self.dist_text_type = kwargs.get('text_type', None)
        self.max_noun_phrases = kwargs.get('max_noun_phrases', None)
        self.captions_path = kwargs.get('captions_path', None)
        self.noun_phrases_path = kwargs.get('noun_phrases_path', None)
        # if align instance
        self.dist_instance = kwargs.get('dist_instance', None)
        self.dist_instance_box = kwargs.get('dist_instance_box', None)
        self.dist_ins_type = kwargs.get('dist_instance_type')
        self.dist_ins_weight = kwargs.get('dist_instance_weight', None)
        self.dist_ins_size = kwargs.get('dist_instance_size', None)
        # if validation on target class
        self.zero_shot = kwargs.get('zero_shot', None)
        self.generalized_zero_shot = kwargs.get('generalized_zero_shot', None)
        # if use cls loss
        self.use_loss_cls = kwargs.get('use_loss_cls', None)
        self.cls_weight = kwargs.get('cls_weight', None)
        self.loss_cls_sample = kwargs.get('loss_cls_sample', None)
        self.loss_cls_sample_per = kwargs.get('loss_cls_sample_per', None)
        # initial temperature
        self.temperature = kwargs.get('temperature', None)
        # combine with clip ve for test
        self.test_with_clip_ve = kwargs.get('test_with_clip_ve', None)
        self.test_with_clip_scale = kwargs.get('test_with_clip_scale', None)
        self.test_with_clip_bg = kwargs.get('test_with_clip_bg', None)
        self.clip_lambda = kwargs.get('clip_lambda', None)
        # whether the background embedding learnable
        self.background_embedding_fix = kwargs.get('background_embedding_fix', None)
        # iou thresh to filter instance for align visual feature and CLIP VE
        self.ins_iou_thresh = kwargs.get('instance_iou_thresh', None)
        # centerness or iou branch
        self.score_prob = kwargs.get('score_prob', None)
        # sort classification score
        self.score_scort = kwargs.get('score_sort', None)

        del kwargs['visual_channels']
        del kwargs['word_embedding']
        del kwargs['base_classes']
        del kwargs['target_classes'] 
        del kwargs['dist_featuremap']
        del kwargs['dist_featuremap_loss']
        del kwargs['dist_featuremap_patch']
        del kwargs['dist_featuremap_pool']
        del kwargs['dist_featuremap_tem']
        del kwargs['text_type']
        del kwargs['max_noun_phrases']
        del kwargs['captions_path']
        del kwargs['noun_phrases_path']
        del kwargs['dist_instance']
        del kwargs['dist_instance_box']
        del kwargs['dist_instance_type']
        del kwargs['dist_instance_weight']
        del kwargs['dist_instance_size']
        del kwargs['zero_shot']
        del kwargs['generalized_zero_shot']
        del kwargs['loss_cls_sample']
        del kwargs['loss_cls_sample_per']
        del kwargs['use_loss_cls']
        del kwargs['cls_weight']
        del kwargs['temperature']
        del kwargs['test_with_clip_ve']
        del kwargs['test_with_clip_scale']
        del kwargs['test_with_clip_bg']
        del kwargs['clip_lambda']
        del kwargs['background_embedding_fix']
        del kwargs['instance_iou_thresh']
        del kwargs['score_prob']
        del kwargs['score_sort']

        super().__init__(
                 num_classes,
                 in_channels,
                 stacked_convs=stacked_convs,
                 conv_cfg=conv_cfg,
                 norm_cfg=norm_cfg,
                 loss_centerness=loss_centerness,
                 init_cfg=init_cfg,
            **kwargs)
        
        self.loss_dist_ins = build_loss(
                                        dict(
                                        type='L1Loss',
                                        loss_weight=5.0))
        # self.loss_dist_feat = build_loss()

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.visual_channels,
            3,
            padding=1)
        
        self.word_embedding = torch.load(self.word_embedding, 'cpu')
        self.base_embedding = self.word_embedding['base']
        self.target_embedding = self.word_embedding['target']
        # self.target_embedding = torch.load('/home/zyma/python_work/mmdetection/data/coco/fullwe.pth', 'cpu').float()
        # self.target_embedding = nn.Parameter(self.target_embedding)

        self.base_linear = nn.Linear(512, self.base_classes, bias=False)
        self.base_linear.weight = self.base_embedding
        self.base_linear.weight.requires_grad = False

        self.target_linear = nn.Linear(512, self.target_classes, bias=False)
        # self.target_linear = nn.Conv2d(512, self.target_classes, 1, 1, 0, bias=False)
        self.target_linear.weight = self.target_embedding
        self.target_linear.weight.requires_grad = False

        self.union_linear = nn.Linear(512, self.base_classes + self.target_classes, bias=False)
        self.union_linear.weight = self.word_embedding['all']
        self.union_linear.weight.requires_grad = False

        # load CLIP for vision and language alignment and fix parameters
        self.clip, _ = clip.load("ViT-B/32", 'cpu')
        for param in self.clip.parameters():
            param.requires_grad = False
        
        self.tao = nn.Parameter(torch.tensor(self.temperature))

        if self.use_loss_cls:
            self.cls_weight = nn.Parameter(torch.tensor(self.cls_weight))
        else:
            self.cls_weight = nn.Parameter(torch.tensor(0.0))

        if self.dist_featmap:
            # divide the featuremap to k * k patch
            if self.dist_featmap_pool == 'max':
                self.pool_featmap_patch = nn.AdaptiveMaxPool2d((self.dist_feat_patch, self.dist_feat_patch))
            elif self.dist_featmap_pool == 'mean':
                self.pool_featmap_patch = nn.AdaptiveAvgPool2d((self.dist_feat_patch, self.dist_feat_patch))
            else:
                raise ValueError("must be mean or max")
            
            self.dist_featmap_tem = nn.Parameter(torch.tensor(self.dist_featmap_tem))
            # load captions
            self.captions = {}
            for mode in ['train', 'val']:
                with open(os.path.join(self.captions_path, 'captions_{}2017.json'.format(mode))) as f:
                    captions = json.load(f)
                    self.captions[mode] = self.load_captions(captions)

            # load noun phrases
            self.noun_phrases = {}
            for mode in ['train']:
                with open(os.path.join(self.noun_phrases_path, 'noun_phrases_{}2017.json'.format(mode))) as f:
                    noun_phrases = json.load(f)
                    self.noun_phrases[mode] = self.load_noun_phrases(noun_phrases)

        if not self.background_embedding_fix:
            # random initialize background word embedding and make it learnable
            background_prompt = ['a photo of background.']
            with torch.no_grad():
                background_class = clip.tokenize(background_prompt)
                background_features = self.clip.encode_text(background_class)
                background_features = background_features / (background_features.norm(dim=-1, keepdim=True) + 1e-10)
                background_features = background_features.float()
            background_embedding = nn.Parameter(background_features)

            self.background_linear = nn.Linear(512, 1, bias=False)
            self.background_linear.weight = background_embedding
            self.background_linear.weight.requires_grad = True
    
    def load_captions(self, captions_dict):
        image_caption_dict = {}

        images = captions_dict['images']
        for image in images:
            image_id = "%06d" %image['id']
            image_caption_dict[image_id] = []
        
        captions = captions_dict['annotations']
        for caption in captions:
            image_caption_id = "%06d" %caption['image_id']
            sentence = caption['caption']
            image_caption_dict[image_caption_id].append(sentence)
        return image_caption_dict
    
    def padding_noun_phrases(self, noun_phrases):
        phrase_num = len(noun_phrases)
        if phrase_num < self.max_noun_phrases:
            noun_phrases.extend(['xxx' for i in range(self.max_noun_phrases - phrase_num)])
        else:
            noun_phrases = noun_phrases[:self.max_noun_phrases]
            phrase_num = self.max_noun_phrases
        return noun_phrases, phrase_num

    def load_noun_phrases(self, noun_phrases_dict):
        image_nounphrases_dict = {}

        for image, sentences in noun_phrases_dict.items():
            image_id = "%06d" %(int(image))
            image_nounphrases_dict[image_id] = []
            for sentence in sentences:
                if sentence['phrases']:
                    noun_phrases = [item['phrase'] for item in sentence['phrases']]
                    noun_phrases, phrases_num = self.padding_noun_phrases(noun_phrases)
                    image_nounphrases_dict[image_id].append((noun_phrases, phrases_num))
        return image_nounphrases_dict

    def freeze_layers(self):
        # fix bn in CLIP when model status transform from eval to train
        layers = [self.clip] if hasattr(self, 'clip') else []
        for layer in layers:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def transforms(self, x, n_px=224):
        # deformation resize
        # return F.interpolate(x, size=(n_px, n_px), mode='bicubic')

        # keep ratio resize: long side resize to n_px, short size padding
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
        return x_pad
    
    def scale_bbox(self, bbox, scale, width, height):
        # device = bbox.device
        left, top, right, down = bbox
        center_x, center_y = (left + right) // 2, (top + down) // 2
        delta_left, delta_right = center_x - left, right - center_x
        delta_top, delta_down = center_y - top, down - center_y
        scale_delta_left, scale_delta_right = scale * delta_left, scale * delta_right
        scale_delta_top, scale_delta_down = scale * delta_top, scale * delta_down
        scale_left, scale_right = max(0, int(center_x - scale_delta_left)), \
                                min(width, int(center_x + scale_delta_right))
        scale_top, scale_down = max(0, int(center_y - scale_delta_top)), \
                                min(height, int(center_y + scale_delta_down))
        return [scale_left, scale_top, scale_right, scale_down]

    def extract_bbox(self, img, bbox, scale=None):
        bbox[bbox < 0] = 0
        if scale is not None:
            bbox = self.scale_bbox(bbox, scale, img.shape[2], img.shape[1])
        left, top, right, bottom = bbox
        right, bottom = right + 1, bottom + 1
        region = img[:, top: bottom, left: right].unsqueeze(0)
        resize_region = self.transforms(region)
        return resize_region

    def forward_train(self,
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=None,
                    proposal_cfg=None,
                    **kwargs):
        img = x[0] # first element is the img
        x = x[1:]  # five FPN featuremaps
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas, img)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, img)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        img = feats[0]
        feats = feats[1:]
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, img=img)
        return results_list
    
    # def train(self, mode=True):
    #     self.training = mode
    #     self.cls_out_channels = self.base_classes
    #     for module in self.children():
    #         module.train(mode)
    #     self.freeze_layers()
    #     return self

    # def forward(self, feats):
    #     return multi_apply(self.forward_single, feats, self.scales)

    # def forward_single(self, x, scale):
    #     cls_feat = x
    #     reg_feat = x
    #     for cls_conv in self.cls_convs:
    #         cls_feat = cls_conv(cls_feat)
    #     for reg_conv in self.reg_convs:
    #         reg_feat = reg_conv(reg_feat)
    #     cls_feat = self.atss_cls(cls_feat)

    #     # b * c * h * w -> b * h * w * c
    #     b ,_ , h, w = cls_feat.shape
    #     cls_feat = cls_feat.permute(0, 2, 3, 1)
    #     # b * h * w * c -> b* h * w * num_anchors * visual_channels
    #     cls_feat = cls_feat.view(b, h , w, self.num_anchors, -1).contiguous()

    #     # word embedding already normlizaed, visual feature also should normalized
    #     cls_feat = cls_feat / (cls_feat.norm(dim=-1, keepdim=True))
    #     if self.training:
    #         cls_score = self.base_linear(cls_feat)
    #     else:
    #         if self.zero_shot:
    #             self.cls_out_channels = self.target_classes
    #             cls_score = self.target_linear(cls_feat)
    #         else:
    #             cls_score = self.base_linear(cls_feat)
    #     cls_score = self.tao * cls_score
    #     cls_score = cls_score.view(b, h, w, -1).contiguous()
    #     cls_score = cls_score.permute(0, 3, 1, 2)

    #     # we just follow atss, not apply exp in bbox_pred
    #     bbox_pred = scale(self.atss_reg(reg_feat)).float()
    #     centerness = self.atss_centerness(reg_feat)
        
    #     if self.training:
    #         return cls_score, bbox_pred, centerness, cls_feat.view(b, h , w, self.num_anchors * self.visual_channels).contiguous()
    #     else:
    #         return cls_score, bbox_pred, centerness

    # def distill_featmap(self, featmap, ):
    #     pass

    # # align instance level feature
    # def distill_instance(self, pos_cls_feat, pos_decode_bbox_pred, pos_label, img, pos_inds, single_anchor_num):
    #     # align visual feature
    #     pos_regions = []
    #     pos_bbox_pred = pos_decode_bbox_pred.int()
    #     img_inds = pos_inds // single_anchor_num
    #     for img_ind, bbox in zip(img_inds, pos_bbox_pred):
    #         # align text feature
    #         pos_regions.append(self.extract_bbox(img[img_ind][:], bbox))
    #     pos_regions = torch.cat(pos_regions, dim=0)
    #     clip_ve_feats = self.clip.encode_image(pos_regions)
    #     norm_clip_ve_feats = clip_ve_feats / (clip_ve_feats.norm(dim=-1, keepdim=True) + 1e-10)
    #     loss_dist_ins = self.loss_dist_ins(pos_cls_feat, norm_clip_ve_feats)
    #     return loss_dist_ins

    # def loss_single(self, anchors, cls_score, bbox_pred, centerness, cls_feat, labels,
    #                 label_weights, bbox_targets, num_total_samples, img):
    #     single_anchor_num = anchors.shape[1]
    #     anchors = anchors.reshape(-1, 4)
    #     cls_score = cls_score.permute(0, 2, 3, 1).reshape(
    #         -1, self.cls_out_channels).contiguous()
    #     bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    #     centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
    #     cls_feat = cls_feat.reshape(-1, self.visual_channels)
    #     bbox_targets = bbox_targets.reshape(-1, 4)
    #     labels = labels.reshape(-1)
    #     label_weights = label_weights.reshape(-1)

    #     # # avg factor
    #     # if self.loss_cls_all:
    #     #     num_total_samples = cls_score.shape[0]

    #     # classification loss
    #     if self.use_loss_cls:
    #         loss_cls = self.loss_cls(
    #             cls_score, labels, label_weights, avg_factor=num_total_samples)
    #         loss_cls = 1 / torch.exp(self.cls_weight) * loss_cls
    #     else:
    #         loss_cls = cls_score.sum() * 0

    #     # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    #     bg_class_ind = self.num_classes
    #     pos_inds = ((labels >= 0)
    #                 & (labels < bg_class_ind)).nonzero().squeeze(1)
        
    #     if len(pos_inds) > 0:
    #         pos_bbox_targets = bbox_targets[pos_inds]
    #         pos_bbox_pred = bbox_pred[pos_inds]
    #         pos_anchors = anchors[pos_inds]
    #         pos_centerness = centerness[pos_inds]

    #         centerness_targets = self.centerness_target(
    #             pos_anchors, pos_bbox_targets)
    #         pos_decode_bbox_pred = self.bbox_coder.decode(
    #             pos_anchors, pos_bbox_pred)
    #         pos_decode_bbox_targets = self.bbox_coder.decode(
    #             pos_anchors, pos_bbox_targets)

    #         # regression loss
    #         loss_bbox = self.loss_bbox(
    #             pos_decode_bbox_pred,
    #             pos_decode_bbox_targets,
    #             weight=centerness_targets,
    #             avg_factor=1.0)

    #         # centerness loss
    #         loss_centerness = self.loss_centerness(
    #             pos_centerness,
    #             centerness_targets,
    #             avg_factor=num_total_samples)

    #     else:
    #         loss_bbox = bbox_pred.sum() * 0
    #         loss_centerness = centerness.sum() * 0
    #         centerness_targets = bbox_targets.new_tensor(0.)
        
    #     # distill instance loss
    #     if self.dist_instance:
    #         if len(pos_inds) == 0:
    #             loss_dist_ins =  cls_feat.sum() * 0
    #         else:
    #             pos_cls_feat = cls_feat[pos_inds]
    #             pos_label = labels[pos_inds]
    #             loss_dist_ins = self.distill_instance(pos_cls_feat, pos_decode_bbox_pred, pos_label, img, pos_inds, single_anchor_num)
    #         return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum(), loss_dist_ins
    #     else:
    #         return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum(), cls_feat.sum() * 0

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'cls_feats'))
    # def loss(self,
    #          cls_scores,
    #          bbox_preds,
    #          centernesses,
    #          cls_feats,
    #          gt_bboxes,
    #          gt_labels,
    #          img_metas,
    #          img,
    #          gt_bboxes_ignore=None):

    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     assert len(featmap_sizes) == self.anchor_generator.num_levels

    #     device = cls_scores[0].device
    #     anchor_list, valid_flag_list = self.get_anchors(
    #         featmap_sizes, img_metas, device=device)
    #     label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
    #     cls_reg_targets = self.get_targets(
    #         anchor_list,
    #         valid_flag_list,
    #         gt_bboxes,
    #         img_metas,
    #         gt_bboxes_ignore_list=gt_bboxes_ignore,
    #         gt_labels_list=gt_labels,
    #         label_channels=label_channels)
    #     if cls_reg_targets is None:
    #         return None

    #     (anchor_list, labels_list, label_weights_list, bbox_targets_list,
    #      bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

    #     num_total_samples = reduce_mean(
    #         torch.tensor(num_total_pos, dtype=torch.float,
    #                      device=device)).item()
    #     num_total_samples = max(num_total_samples, 1.0)

    #     losses_cls, losses_bbox, loss_centerness,\
    #         bbox_avg_factor, losses_dist_ins = multi_apply(
    #             self.loss_single,
    #             anchor_list,
    #             cls_scores,
    #             bbox_preds,
    #             centernesses,
    #             cls_feats,
    #             labels_list,
    #             label_weights_list,
    #             bbox_targets_list,
    #             num_total_samples=num_total_samples,
    #             img=img)

    #     bbox_avg_factor = sum(bbox_avg_factor)
    #     bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
    #     losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
    #     return dict(
    #         loss_cls=losses_cls,
    #         loss_bbox=losses_bbox,
    #         loss_centerness=loss_centerness,
    #         loss_dist_ins=losses_dist_ins,
    #         cls_weight_loss=torch.abs(self.cls_weight))

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    # def get_bboxes(self,
    #                cls_scores,
    #                bbox_preds,
    #                centernesses,
    #                img_metas,
    #                cfg=None,
    #                rescale=False,
    #                with_nms=True,
    #                img=None):

    #     cfg = self.test_cfg if cfg is None else cfg
    #     assert len(cls_scores) == len(bbox_preds)
    #     num_levels = len(cls_scores)
    #     device = cls_scores[0].device
    #     featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    #     mlvl_anchors = self.anchor_generator.grid_anchors(
    #         featmap_sizes, device=device)

    #     cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
    #     bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
    #     centerness_pred_list = [
    #         centernesses[i].detach() for i in range(num_levels)
    #     ]
    #     img_shapes = [
    #         img_metas[i]['img_shape'] for i in range(cls_scores[0].shape[0])
    #     ]
    #     scale_factors = [
    #         img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
    #     ]
    #     result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
    #                                    centerness_pred_list, mlvl_anchors,
    #                                    img_shapes, scale_factors, cfg, rescale,
    #                                    with_nms, img)
    #     return result_list

    # def _get_bboxes(self,
    #                 cls_scores,
    #                 bbox_preds,
    #                 centernesses,
    #                 mlvl_anchors,
    #                 img_shapes,
    #                 scale_factors,
    #                 cfg,
    #                 rescale=False,
    #                 with_nms=True,
    #                 img=None):
    #     assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
    #     device = cls_scores[0].device
    #     batch_size = cls_scores[0].shape[0]
    #     # convert to tensor to keep tracing
    #     nms_pre_tensor = torch.tensor(
    #         cfg.get('nms_pre', -1), device=device, dtype=torch.long)
    #     mlvl_bboxes = []
    #     mlvl_scores = []
    #     mlvl_centerness = []
    #     mlvl_clip_scores = []
    #     for cls_score, bbox_pred, centerness, anchors in zip(
    #             cls_scores, bbox_preds, centernesses, mlvl_anchors):
    #         assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
    #         scores = cls_score.permute(0, 2, 3, 1).reshape(
    #             batch_size, -1, self.cls_out_channels).sigmoid()
    #         centerness = centerness.permute(0, 2, 3,
    #                                         1).reshape(batch_size,
    #                                                    -1).sigmoid()
    #         bbox_pred = bbox_pred.permute(0, 2, 3,
    #                                       1).reshape(batch_size, -1, 4)

    #         # Always keep topk op for dynamic input in onnx
    #         if nms_pre_tensor > 0 and (torch.onnx.is_in_onnx_export()
    #                                    or scores.shape[-2] > nms_pre_tensor):
    #             from torch import _shape_as_tensor
    #             # keep shape as tensor and get k
    #             num_anchor = _shape_as_tensor(scores)[-2].to(device)
    #             nms_pre = torch.where(nms_pre_tensor < num_anchor,
    #                                   nms_pre_tensor, num_anchor)

    #             max_scores, _ = (scores * centerness[..., None]).max(-1)
    #             _, topk_inds = max_scores.topk(nms_pre)
    #             anchors = anchors[topk_inds, :]
    #             batch_inds = torch.arange(batch_size).view(
    #                 -1, 1).expand_as(topk_inds).long()
    #             bbox_pred = bbox_pred[batch_inds, topk_inds, :]
    #             scores = scores[batch_inds, topk_inds, :]
    #             centerness = centerness[batch_inds, topk_inds]
    #         else:
    #             anchors = anchors.expand_as(bbox_pred)

    #         bboxes = self.bbox_coder.decode(
    #             anchors, bbox_pred, max_shape=img_shapes)

    #         # combine with CLIP visual encoder for prediction scores
    #         if self.test_with_clip_ve:
    #             if bboxes.shape[1] > 0:
    #                 extract_img = []
    #                 for bbox in bboxes.view(-1, 4).int():
    #                     extract_img.append(self.extract_bbox(img[0][:], bbox))
    #                 extract_img = torch.cat(extract_img, dim=0)
    #                 clip_ve_feats = self.clip.encode_image(extract_img).float()
    #                 clip_ve_feats = clip_ve_feats / (clip_ve_feats.norm(dim=-1, keepdim=True) + 1e-10)
    #                 if self.zero_shot:
    #                     clip_scores = self.clip.logit_scale.exp() * self.target_linear(clip_ve_feats)
    #                     clip_scores = clip_scores.softmax(dim=-1).unsqueeze(0)
    #                     # mlvl_clip_scores.append(clip_scores)
    #                     # scores = scores.pow(self.clip_lambda) * clip_scores.pow(1 - self.clip_lambda)
    #                     scores = clip_scores

    #         mlvl_bboxes.append(bboxes)
    #         mlvl_scores.append(scores)
    #         mlvl_centerness.append(centerness)

    #     # # debug to compare the difference between clip_scores and scores
    #     # mlvl_clip_score = torch.cat(mlvl_clip_scores, dim=1)
    #     # mlvl_score = torch.cat(mlvl_scores, dim=1)
    #     # import ipdb
    #     # ipdb.set_trace()

    #     batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    #     if rescale:
    #         batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
    #             scale_factors).unsqueeze(1)
    #     batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    #     batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

    #     # Set max number of box to be feed into nms in deployment
    #     deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
    #     if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
    #         batch_mlvl_scores, _ = (
    #             batch_mlvl_scores *
    #             batch_mlvl_centerness.unsqueeze(2).expand_as(batch_mlvl_scores)
    #         ).max(-1)
    #         _, topk_inds = batch_mlvl_scores.topk(deploy_nms_pre)
    #         batch_inds = torch.arange(batch_size).view(-1,
    #                                                    1).expand_as(topk_inds)
    #         batch_mlvl_scores = batch_mlvl_scores[batch_inds, topk_inds, :]
    #         batch_mlvl_bboxes = batch_mlvl_bboxes[batch_inds, topk_inds, :]
    #         batch_mlvl_centerness = batch_mlvl_centerness[batch_inds,
    #                                                       topk_inds]
    #     # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
    #     # BG cat_id: num_class
    #     padding = batch_mlvl_scores.new_zeros(batch_size,
    #                                           batch_mlvl_scores.shape[1], 1)
    #     batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

    #     if with_nms:
    #         det_results = []
    #         for (mlvl_bboxes, mlvl_scores,
    #              mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
    #                                      batch_mlvl_centerness):
    #             det_bbox, det_label = multiclass_nms(
    #                 mlvl_bboxes,
    #                 mlvl_scores,
    #                 cfg.score_thr,
    #                 cfg.nms,
    #                 cfg.max_per_img,
    #                 score_factors=mlvl_centerness)
    #             det_results.append(tuple([det_bbox, det_label]))
    #     else:
    #         det_results = [
    #             tuple(mlvl_bs)
    #             for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
    #                                batch_mlvl_centerness)
    #         ]
    #     return det_results