_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_sgd_1200e.py',
    '../../_base_/det_models/dbnet_r18_fpnc.py',
    '../../_base_/det_datasets/icdar2015.py',
    '../../_base_/det_pipelines/dbnet_pipeline.py'
]


train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_r18 = {{_base_.train_pipeline_r18}}
test_pipeline_1333_736 = {{_base_.test_pipeline_1333_736}}

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_r18),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_1333_736),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_1333_736))

evaluation = dict(interval=100, metric='hmean-iou')

# Overwrite the model config
model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.FCRNBNRP',
        n=8,
        _delete_=True
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # frozen_stages=-1,
        # norm_cfg=dict(type='BN', requires_grad=True),
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        # norm_eval=False,
        # style='caffe'),
        ),
    neck=dict(
        type='FPNC', in_channels=[8, 16, 32, 64], lateral_channels=256),
    bbox_head=dict(
        type='DBHead',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='quad')),
    train_cfg=None,
    test_cfg=None)
