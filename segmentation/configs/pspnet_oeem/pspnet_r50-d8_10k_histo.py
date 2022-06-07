_base_ = [
    '../_base_/models/pspnet_r50_wsss.py',
    '../_base_/datasets/histo_wsss_256.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_10k.py'
]
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
