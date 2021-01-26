# _base_ = [
#     '../_base_/models/mask_rcnn_r50_fpn.py',
#     '../_base_/datasets/censor_detection.py',
#     '../_base_/schedules/schedule_01x.py', '../_base_/default_runtime.py'
# ]
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_01x.py','../_base_/default_runtime.py',
    '../_base_/datasets/censor_detection.py',
]
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1203), mask_head=dict(num_classes=1203)))

test_cfg = dict(
    rcnn=dict(
        score_thr=0.0001,
        # LVIS allows up to 300
        max_per_img=300))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(
#         type='Resize',
#         img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
#                    (1333, 768), (1333, 800)],
#         multiscale_mode='value',
#         keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]
#data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))
load_from='./pretrained/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth'
