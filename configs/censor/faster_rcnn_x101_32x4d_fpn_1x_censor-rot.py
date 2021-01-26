_base_ = './faster_rcnn_r50_fpn_1x_censor.py'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Rotate',
        prob=0.5,
        level=10,
        scale=30,
        random_negative_prob=0.5),
    dict(type='RandomFlip', flip_ratio=0.5,direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5,direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
       #pretrained='open-mmlab://resnext101_32x4d',
    
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))
load_from='./pretrained/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-ebbc5c81.pth'