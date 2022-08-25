_base_ = [
    '../../_base_/models/atss_r50_fpn_zsd.py',
    '../../_base_/datasets/coco_zero_shot_detection.py',
    '../../_base_/schedules/schedule_1x.py', 
    '../../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# distill setting
model = dict(
    bbox_head=dict(
            zero_shot=True,
            test_with_clip_ve=True,
            dist_featuremap = False,
            dist_instance = False,
        )
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
