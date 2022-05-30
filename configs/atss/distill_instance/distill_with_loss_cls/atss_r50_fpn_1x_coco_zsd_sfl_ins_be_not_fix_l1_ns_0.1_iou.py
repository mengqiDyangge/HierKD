_base_ = [
    '../../../_base_/models/atss_r50_fpn_zsd.py',
    '../../../_base_/datasets/coco_zero_shot_detection.py',
    '../../../_base_/schedules/schedule_1x_zsd.py', 
    '../../../_base_/default_runtime.py'
]

# log
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

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
        type='ATSSZSDSFLHead',
        zero_shot=True,
        background_embedding_fix=False,
        use_loss_cls=True,
        temperature=100.0,
        cls_weight=1.0,
        test_with_clip_ve=False,
        test_with_clip_scale=1.0,
        test_with_clip_bg=False,
        dist_featuremap=False,
        dist_instance=True,
        dist_instance_type='KD_L1Loss',
        dist_instance_weight=1.0,
        loss_cls=dict(
            type='SoftMaxFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        score_prob='iou',    
        loss_cls_sample=True,
        loss_cls_sample_per=0.1,
        ),
    test_cfg=dict(
        score_thr=0.0,
        nms=dict(type='nms', iou_threshold=0.4))
    )

data_root = 'data/coco/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val=dict(
        ann_file=data_root + 'annotations/instances_val_unseen.json'
    ),
    test=dict(
        ann_file=data_root + 'annotations/instances_val_unseen.json',
    )
)