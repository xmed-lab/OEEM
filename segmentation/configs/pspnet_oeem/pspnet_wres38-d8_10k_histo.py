_base_ = './pspnet_r50-d8_10k_histo.py'
model = dict(
    pretrained='models/res38d.pth',
    backbone=dict(
        type='WideRes38'),
    decode_head=dict(
        in_channels=4096,
        loss_decode=dict(custom_str='oeem')),
    auxiliary_head=dict(
        in_channels=1024,
        loss_decode=dict(custom_str='oeem'))
)
test_cfg = dict(mode='slide', crop_size=(320, 320), stride=(256, 256), crf=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 256), ratio_range=(0.75, 3.)),
    dict(type='RandomCrop', crop_size=(256, 256)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data_root = 'glas/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
        data_root=data_root,
        img_dir='img_train_256_192',
        ann_dir='gt_train_ours_256_192',
        split='train.lst',
        ),
    val=dict(
        data_root=data_root,
        img_dir='img_val_256_192',
        ann_dir='gt_val_256_192',
        split='val.lst',
        ),
    test=dict(
        data_root=data_root,
        img_dir='img_val_256_192',
        ann_dir='gt_val_256_192',
        split='val.lst',
        ))
