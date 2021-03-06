_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    #pretrained='./pretrained/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-ebbc5c81.pth',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))
