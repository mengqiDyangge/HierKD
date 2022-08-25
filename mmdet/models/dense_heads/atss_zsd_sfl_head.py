# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import torch
import torch.nn as nn
import numpy as np
import random
import time
from ..clip import clip
import torch.nn.functional as F

from torchvision.transforms import Resize, Compose, ToTensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import scale
BICUBIC = InterpolationMode.BICUBIC
from mmcv.runner import force_fp32
from ..builder import HEADS, build_loss
from .atss_zsd_head import ATSSZSDHead
from mmdet.core import (anchor_inside_flags, bbox, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)


@HEADS.register_module()
class ATSSZSDSFLHead(ATSSZSDHead):
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
        if kwargs['loss_cls'].get('use_sigmoid', None):
            del kwargs['loss_cls']['use_sigmoid']

        super().__init__(
                 num_classes,
                 in_channels,
                 stacked_convs=stacked_convs,
                 conv_cfg=conv_cfg,
                 norm_cfg=norm_cfg,
                 loss_centerness=loss_centerness,
                 init_cfg=init_cfg,
            **kwargs)
              
        self.loss_dist_ins = build_loss(dict(
                                        type=self.dist_ins_type,
                                        loss_weight=self.dist_ins_weight))
        self.loss_dist_feat = build_loss(dict(
                                    type=self.dist_featmap_loss))

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()

    def train(self, mode=True):
        self.training = mode
        self.cls_out_channels = self.base_classes + 1
        self.num_classes = self.base_classes
        for module in self.children():
            module.train(mode)
        self.freeze_layers()
        return self

    def forward(self, feats):
        # with torch.no_grad():
        #     weight_clone = self.background_linear.weight.data.clone() 
        #     weight_clone = weight_clone / (weight_clone.norm(dim=-1, keepdim=True) + 1e-10)
        #     weight_clone = nn.Parameter(weight_clone)
        # normalized background word embedding before use
        
        # print(self.background_linear.weight.norm())
        # weight_norm(self.background_linear, 'weight', dim=-1)
        # print(self.background_linear.weight.norm())
        if not self.background_embedding_fix:
            with torch.no_grad():
                self.background_linear.weight.data = self.background_linear.weight.data / \
                                                    (self.background_linear.weight.data.norm(dim=-1, keepdim=True) + 1e-10)
        # print(self.background_linear.weight.norm())
        return multi_apply(self.forward_single, feats, self.scales)
        # ts = time.time()
        # result = multi_apply(self.forward_single, feats, self.scales)
        # te = time.time()
        # print(te - ts)
        # return result

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_feat = self.atss_cls(cls_feat)

        # b * c * h * w -> b * h * w * c
        b ,_ , h, w = cls_feat.shape
        cls_feat = cls_feat.permute(0, 2, 3, 1)
        # b * h * w * c -> b* h * w * num_anchors * visual_channels
        cls_feat = cls_feat.view(b, h , w, self.num_anchors, -1).contiguous()

        # word embedding already normlizaed, visual feature also should normalized
        cls_feat = cls_feat / (cls_feat.norm(dim=-1, keepdim=True) + 1e-10)

        # detach visual feature for training background embedding        
        # if self.background_embedding_fix:
        #     background_score = self.background_linear(cls_feat)
        # else:
        #     # if background embedding is trainable, detach visual feature for grad decent
        #     detach_cls_feat = cls_feat.detach()
        #     background_score = self.background_linear(detach_cls_feat)

        if not self.background_embedding_fix:
            assert (self.background_linear.weight.norm() - 1.0) < 0.01
            bg_score = self.background_linear(cls_feat)
        else:
            bg_score = 0.2 * torch.ones_like(cls_feat[:, :, :, :, :1]).to(cls_feat.device)

        if self.training:
            fore_score = self.base_linear(cls_feat)
        else:
            if self.generalized_zero_shot:
                self.cls_out_channels = self.base_classes + self.target_classes + 1
                self.num_classes = self.base_classes + self.target_classes
                fore_score = self.union_linear(cls_feat)
            elif self.zero_shot:
                self.cls_out_channels = self.target_classes + 1
                self.num_classes = self.target_classes
                fore_score = self.target_linear(cls_feat)
            else:
                fore_score = self.base_linear(cls_feat)

        cls_score = torch.cat((fore_score, bg_score), dim=-1)
        cls_score = self.tao * cls_score
        cls_score = cls_score.view(b, h, w, -1).contiguous()
        cls_score = cls_score.permute(0, 3, 1, 2)

        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        
        if self.training:
            return cls_score, bbox_pred, centerness, cls_feat.view(b, h , w, self.num_anchors * self.visual_channels).contiguous()
        else:
            return cls_score, bbox_pred, centerness
    
    def distill_featmap(self, text_feats, featmap, ori_shape):
        b, h, w = ori_shape 
        # featmap: bhw*c -> b*h*w*c -> b*c*h*w
        featmap = featmap.reshape((b, h, w, -1)).permute(0, 3, 1, 2)
        # adaptive average pool2d to k*k patch: b*c*h*w -> b*c*k*k -> b*k*k*c
        feat_patch = self.pool_featmap_patch(featmap).permute(0, 2, 3, 1)
        feat_patch = feat_patch / (feat_patch.norm(dim=-1, keepdim=True) + 1e-10)
        # b*k*k*c -> bkk*c
        feat_patch = feat_patch.reshape(-1, self.visual_channels)
        # response: b*bkk  
        response = text_feats @ feat_patch.t()
        # response: b*b*kk
        # response = response.reshape(b, b, -1)
        return feat_patch.view(b, -1, self.visual_channels).contiguous(), response

    # align instance level feature
    def distill_instance(self, pos_cls_feat, pos_bbox_pred, pos_label, pos_bbox_target, img, pos_inds, single_anchor_num):
        # align visual feature

        # IOU filter
        iou = self.assigner.iou_calculator(pos_bbox_pred, pos_bbox_target)
        iou = torch.diagonal(iou, dim1=-2, dim2=-1)
        reserve_ind = torch.where(iou >= self.ins_iou_thresh)[0]
        
        if len(reserve_ind) == 0:
            loss_dist_ins =  pos_cls_feat.sum() * 0
        else:
            pos_cls_feat = pos_cls_feat[reserve_ind]
            pos_bbox_pred = pos_bbox_pred[reserve_ind]
            pos_label = pos_label[reserve_ind]
            pos_bbox_target = pos_bbox_target[reserve_ind]
            pos_inds = pos_inds[reserve_ind]

            if self.dist_instance_box == 'gt':
                distill_bbox = pos_bbox_target
            elif self.dist_instance_box == 'pred':
                distill_bbox = pos_bbox_pred

            pos_regions = []
            distill_bbox = distill_bbox.int()
            img_inds = pos_inds // single_anchor_num
            for img_ind, bbox in zip(img_inds, distill_bbox):
                # align text feature
                if self.dist_ins_size != 1.0:
                    pos_regions.append(self.extract_bbox(img[img_ind][:], bbox, self.dist_ins_size))
                else:
                    pos_regions.append(self.extract_bbox(img[img_ind][:], bbox))
            pos_regions = torch.cat(pos_regions, dim=0)
            clip_ve_feats = self.clip.encode_image(pos_regions)
            norm_clip_ve_feats = clip_ve_feats / (clip_ve_feats.norm(dim=-1, keepdim=True) + 1e-10)
            
            # # test cls accuracy of the crop instance
            # probs = 100 * self.base_linear(norm_clip_ve_feats)
            # probs = probs.softmax(dim=-1)
            # probs, indi = probs.topk(5)
            # equal = sum(indi == pos_label.unsqueeze(1))
            # top1 = equal[0] / len(pos_label)
            # top5 = sum(equal) / len(pos_label)
            # import ipdb
            # ipdb.set_trace()

            loss_dist_ins = self.loss_dist_ins(pos_cls_feat, norm_clip_ve_feats)
        return loss_dist_ins
    
    def text_mask(self, seq, max_seq_len):
        mask = torch.zeros(len(seq), max_seq_len)
        for i, seq_len in enumerate(seq):
            mask[i, :seq_len] = 1
        return mask

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, cls_feat, labels,
                    label_weights, bbox_targets, num_total_samples, img, text_feats):
        single_anchor_num = anchors.shape[1]
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        b, h, w, _ = cls_feat.shape
        cls_feat = cls_feat.reshape(-1, self.visual_channels)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        # classification loss
        if self.use_loss_cls:
            loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples, 
                                    pos_inds=pos_inds, negative_sample=self.loss_cls_sample,
                                    sample_per=self.loss_cls_sample_per)
            loss_cls = 1 / torch.exp(self.cls_weight) * loss_cls
        else:
            loss_cls = cls_score.sum() * 0
        
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)
            
            if self.score_prob == 'centerness':
                centerness_targets = self.centerness_target(
                                        pos_anchors, pos_bbox_targets)
            elif self.score_prob == 'iou':
                centerness_targets = self.assigner.iou_calculator(pos_decode_bbox_pred.detach(), 
                                                                  pos_decode_bbox_targets)
                centerness_targets = torch.diagonal(centerness_targets, dim1=-2, dim2=-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)
        
        # distill instance loss
        if self.dist_instance:
            if len(pos_inds) == 0:
                loss_dist_ins =  cls_feat.sum() * 0
            else:
                pos_cls_feat = cls_feat[pos_inds]
                pos_label = labels[pos_inds]
                loss_dist_ins = self.distill_instance(pos_cls_feat,
                                                      pos_decode_bbox_pred,
                                                      pos_label,
                                                      pos_decode_bbox_targets,
                                                      img,
                                                      pos_inds,
                                                      single_anchor_num)
        else:
            loss_dist_ins =  cls_feat.sum() * 0
        
        # align featuremap with caption
        if self.dist_featmap:
            feat_and_response = self.distill_featmap(text_feats, cls_feat, (b, h, w))
        else:
            feat_and_response = None
        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum(), loss_dist_ins, feat_and_response


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'cls_feats'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             cls_feats,
             gt_bboxes,
             gt_labels,
             img_metas,
             img,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        if self.dist_featmap:
            if self.dist_text_type == 'caption':
                img_captions = []
                for img_meta in img_metas:
                    img_id = img_meta['ori_filename'].split('.')[0][-6:]
                    num_caps = len(self.captions['train'][img_id])
                    random_select_ind = random.choice(range(num_caps))
                    img_captions.append(self.captions['train'][img_id][random_select_ind])
                img_captions = clip.tokenize(img_captions).to(device)
                text_feats = self.clip.encode_text(img_captions)
                text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-10)
                text_mask = self.text_mask([1] * len(img_metas), 1).to(device)
            elif self.dist_text_type == 'noun_phrase':
                noun_phrases = []
                phrase_num = []
                for img_meta in img_metas:
                    img_id = img_meta['ori_filename'].split('.')[0][-6:]
                    num_sents = len(self.noun_phrases['train'][img_id])
                    random_select_ind = random.choice(range(num_sents))
                    noun_phrases.extend(self.noun_phrases['train'][img_id][random_select_ind][0])
                    phrase_num.append(self.noun_phrases['train'][img_id][random_select_ind][1])
                # padding the noun phrases in each sentence and record mask
                noun_phrases = clip.tokenize(noun_phrases).to(device)
                text_feats = self.clip.encode_text(noun_phrases)
                text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-10)
                # text_feats = text_feats.reshape(len(img_metas), self.max_noun_phrases, -1)
                text_mask = self.text_mask(phrase_num, self.max_noun_phrases).to(device)
            else:
                raise ValueError('Not implement')
        else:
            text_feats = None

        losses_cls, losses_bbox, loss_centerness,\
            bbox_avg_factor, losses_dist_ins, feats_and_responses = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                cls_feats,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                num_total_samples=num_total_samples,
                img=img,
                text_feats=text_feats)
        
        if self.dist_featmap:
            feat_patchs = []
            responses = []
            for item in feats_and_responses:
                feat_patchs.append(item[0])
                responses.append(item[1].reshape(-1, len(img_metas), self.dist_feat_patch ** 2))
            feat_patchs = torch.cat(feat_patchs, dim=1)
            attentions = torch.cat(responses, dim=-1).softmax(dim=-1)
            text_img_feats = torch.matmul(attentions.unsqueeze(2), feat_patchs).squeeze(2)
            text_img_feats = text_img_feats / (text_img_feats.norm(dim=-1, keepdim=True) + 1e-10)
            # text_img_match = self.dist_featmap_tem * torch.bmm(text_img_feats, text_feats.unsqueeze(2)).squeeze(-1)
            # img_text_match = text_img_match.t()
            # target = torch.LongTensor(range(len(text_img_feats))).to(device)
            loss_dist_featmap = self.loss_dist_feat(text_img_feats, text_feats, self.dist_featmap_tem, text_mask, self.dist_text_type)
        else:
            loss_dist_featmap = cls_feats[0].sum() * 0

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        losses_dist_ins = list(map(lambda x: x / len(losses_dist_ins), losses_dist_ins))

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness,
            loss_dist_ins=losses_dist_ins,
            loss_dist_featmap=loss_dist_featmap,
            cls_weight_loss=torch.abs(self.cls_weight))

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   img=None):

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        img_shapes = [
            img_metas[i]['img_shape'] for i in range(cls_scores[0].shape[0])
        ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, mlvl_anchors,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms, img)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True,
                    img=None):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_bg_scores = []
        for cls_score, bbox_pred, centerness, anchors in zip(
                cls_scores, bbox_preds, centernesses, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).softmax(dim=-1)
            bg_scores = scores[:, :, -1]
            scores = scores[:, :, :-1]
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)

            # Always keep topk op for dynamic input in onnx
            if nms_pre_tensor > 0 and (torch.onnx.is_in_onnx_export()
                                       or scores.shape[-2] > nms_pre_tensor):
                from torch import _shape_as_tensor
                # keep shape as tensor and get k
                num_anchor = _shape_as_tensor(scores)[-2].to(device)
                nms_pre = torch.where(nms_pre_tensor < num_anchor,
                                      nms_pre_tensor, num_anchor)

                # score sort for anchors
                if self.score_scort == 'prob+cls':
                    max_scores, _ = (scores * centerness[..., None]).max(-1)
                elif self.score_scort == 'prob':
                    max_scores = centerness
                elif self.score_scort == 'cls':
                    max_scores, _ = scores.max(-1)
                else:
                    raise ValueError('Not implement')

                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
                centerness = centerness[batch_inds, topk_inds]
                bg_scores = bg_scores[batch_inds, topk_inds]
            else:
                anchors = anchors.expand_as(bbox_pred)

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shapes)

            # combine with CLIP visual encoder for prediction scores
            if self.test_with_clip_ve:
                if bboxes.shape[1] > 0:
                    extract_img = []
                    for bbox in bboxes.view(-1, 4).int():
                        extract_img.append(self.extract_bbox(img[0][:], bbox, self.test_with_clip_scale))
                    extract_img = torch.cat(extract_img, dim=0)
                    clip_ve_feats = self.clip.encode_image(extract_img).float()
                    clip_ve_feats = clip_ve_feats / (clip_ve_feats.norm(dim=-1, keepdim=True) + 1e-10)

                    if self.zero_shot:
                        clip_scores = self.clip.logit_scale.exp() * self.target_linear(clip_ve_feats)
                    else:
                        clip_scores = self.clip.logit_scale.exp() * self.base_linear(clip_ve_feats)                                            
                    
                    if self.test_with_clip_bg:
                        clip_bg_scores = self.clip.logit_scale.exp() * self.background_linear(clip_ve_feats)
                        clip_scores = torch.cat((clip_scores, clip_bg_scores), dim=1)
                        clip_scores = clip_scores.softmax(dim=-1).unsqueeze(0)
                        clip_scores = clip_scores[:, :, :-1]
                    else:
                        clip_scores = clip_scores.softmax(dim=-1).unsqueeze(0)
                    # mlvl_clip_scores.append(clip_scores)
                    # scores = scores.pow(self.clip_lambda) * clip_scores.pow(1 - self.clip_lambda)
                    scores = clip_scores

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_bg_scores.append(bg_scores)

        # # debug to compare the difference between clip_scores and scores
        # mlvl_clip_score = torch.cat(mlvl_clip_scores, dim=1)
        # mlvl_score = torch.cat(mlvl_scores, dim=1)
        # import ipdb
        # ipdb.set_trace()

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
        batch_mlvl_bg_scores = torch.cat(mlvl_bg_scores, dim=1)

        # Set max number of box to be feed into nms in deployment
        deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
        if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
            batch_mlvl_scores, _ = (
                batch_mlvl_scores *
                batch_mlvl_centerness.unsqueeze(2).expand_as(batch_mlvl_scores)
            ).max(-1)
            _, topk_inds = batch_mlvl_scores.topk(deploy_nms_pre)
            batch_inds = torch.arange(batch_size).view(-1,
                                                       1).expand_as(topk_inds)
            batch_mlvl_scores = batch_mlvl_scores[batch_inds, topk_inds, :]
            batch_mlvl_bboxes = batch_mlvl_bboxes[batch_inds, topk_inds, :]
            batch_mlvl_centerness = batch_mlvl_centerness[batch_inds,
                                                          topk_inds]
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        # concat bg score from softmax output
        # batch_mlvl_scores = torch.cat([batch_mlvl_scores, batch_mlvl_bg_scores.unsqueeze(2)], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                if self.score_scort == 'prob+cls':
                        score_factors = mlvl_centerness
                elif self.score_scort == 'prob':
                        mlvl_scores = mlvl_centerness.unsqueeze(1).expand(mlvl_scores.size(0),
                                                                          mlvl_scores.size(1) - 1)
                        mlvl_scores = torch.cat([mlvl_scores, batch_mlvl_scores.new_zeros(batch_mlvl_scores.shape[1], 1)], dim=-1)
                        score_factors = None
                elif self.score_scort == 'cls':
                        score_factors = None
                else:
                    raise ValueError('Not implement')
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=score_factors)
                det_results.append(tuple([det_bbox, det_label]))
                # scores = [cls_score.permute(0, 2, 3, 1).softmax(dim=-1) for cls_score in cls_scores]
                # det_results.append(tuple([det_bbox, det_label, scores]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results