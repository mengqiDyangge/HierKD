_base_ = [
    '../../_base_/models/atss_r50_fpn_zsd.py',
    '../../_base_/datasets/coco_zero_shot_detection.py',
    '../../_base_/schedules/schedule_1x_zsd.py', 
    '../../_base_/default_runtime.py'
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
        use_loss_cls = True,
        test_with_clip_ve = False,
        dist_featuremap = False,
        dist_instance = False,
        loss_cls=dict(
            type='SoftMaxFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        )
    )

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
