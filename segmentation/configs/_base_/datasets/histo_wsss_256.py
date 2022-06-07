dataset_type = 'HistoDataset'
data_root = 'glas/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 256), ratio_range=(0.75, 3.)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 256),
        img_ratios=[0.75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3.],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_root=data_root,
        img_dir='img_train_256_192',
        ann_dir='gt_train_ours_256_192',
        split='train.lst',
        ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        img_dir='img_val_256_192',
        ann_dir='gt_val_256_192',
        split='val.lst',
        ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        img_dir='img_val_256_192',
        ann_dir='gt_val_256_192',
        split='val.lst',
        ))
