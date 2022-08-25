_base_ = [
    '../../_base_/models/retinanet_r50_fpn_zsd.py',
    '../../_base_/datasets/coco_zero_shot_detection.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# distill setting
model = dict(
    bbox_head=dict(
            dist_featuremap = True,
            dist_instance = True,
        )
    )