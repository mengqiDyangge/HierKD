_base_ = [
    '../../../_base_/models/atss_r50_fpn_zsd.py',
    '../../../_base_/datasets/coco_zero_shot_detection.py',
    '../../../_base_/schedules/schedule_1x_zsd.py', 
    '../../../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# distill setting
model = dict(
    bbox_head=dict(
            use_loss_cls = False,
            dist_featuremap = False,
            dist_instance = True,
        )
    )
