'''
Author: bin.zhu
Date: 2022-06-14 11:29:44
LastEditors: bin.zhu
LastEditTime: 2022-06-16 13:26:02
Description: file content
'''

_base_ = './yolox_tiny_8x8_300e_coco.py'
dataset_type = 'BookDataset'
data_root = '/albin/coco/'
# model settings
model = dict(
    backbone=dict(deepen_factor=0.33, widen_factor=0.25, use_depthwise=True),
    neck=dict(
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        use_depthwise=True),
    bbox_head=dict(
        num_classes=9, in_channels=64, feat_channels=64, use_depthwise=True))

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=[
        dict(type='Mosaic', img_scale=(1024, 1024), pad_val=114.0),
        dict(
            type='RandomAffine',
            scaling_ratio_range=(0.5, 1.5),
            border=(-320, -320)),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]),
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=4,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
load_from="/albin/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
