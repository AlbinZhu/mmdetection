'''
Author: bin.zhu
Date: 2022-06-14 11:29:44
LastEditors: bin.zhu
LastEditTime: 2022-06-14 15:23:52
Description: file content
'''

_base_ = './yolox_tiny_8x8_300e_coco.py'
dataset_type = 'BookDataset'
data_root = '/home/albin/Documents/projects/data_process/coco/'
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
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ))
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/'))
