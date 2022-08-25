_base_ = [
    '../../../_base_/models/atss_r50_fpn_zsd.py',
    '../../../_base_/datasets/coco_zero_shot_detection.py',
    '../../../_base_/schedules/schedule_1x_zsd.py', 
    '../../../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

# distill setting
model = dict(
    bbox_head=dict(
            use_loss_cls = True,
            test_with_clip_ve = True,
            dist_featuremap = False,
            dist_instance = True,
        )
    # test_cfg=dict(
    #     nms_pre=100,
    #     min_bbox_size=0,
    #     score_thr=0.3,
    #     nms=dict(type='nms', iou_threshold=0.1),
    #     max_per_img=10)
    )
