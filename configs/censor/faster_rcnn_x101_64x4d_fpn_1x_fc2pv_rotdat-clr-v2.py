_base_ = './faster_rcnn_r50_fpn_1x_censor.py'
dataset_type = 'CensorDataset'
data_root = 'data/censor/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(
         type='PhotoMetricDistortion',
         brightness_delta=16,
         contrast_range=(0.8, 1.2),
         saturation_range=(0.8, 1.2),
         hue_delta=9),
    dict(type='RandomFlip', flip_ratio=0.5,direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5,direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        filter_empty_gt = False,
        ann_file=data_root + 'train/random12_pornpic2k_hm3k_abby_javporn_fc2pv_jointRot_bgrd.json',
        #ann_file=data_root + 'train/random12_pornpic2k_hm3k_abby_javporn_flt.json',
       img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        filter_empty_gt = False,
        ann_file=data_root + 'val/val_jointRot_bgrd.json',
        #ann_file=data_root + 'val/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        filter_empty_gt = False,
        ann_file=data_root + 'val/val_jointRot_bgrd.json',
        #ann_file=data_root + 'val/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))

model = dict(
#    pretrained='open-mmlab://resnext101_32x4d',
       #pretrained='open-mmlab://resnext101_32x4d',
    pretrained=None,
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 32, 48])
total_epochs = 54
fp16 = dict(loss_scale=512.)
#load_from='./pretrained/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-ebbc5c81.pth'
load_from='./pretrained/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-43d9edfe.pth'
#load_from='./pretrained/censor/x101_32x4d_wnRot_wBgrd-epoch27.pth'
#load_from='./pretrained/censor/x101_32x4d_wnRot_wBgrd-epoch27.pth'
#load_from='./pretrained/censor/faster_rcnn_x101_32x4d_fpn_1x_rotdat-epoch10.pth'
log_config = dict(interval=200)
evaluation = dict(start=1)
