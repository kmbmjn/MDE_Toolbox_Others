_base_ = [
    ### '../_base_/models/bts.py', '../_base_/datasets/kitti.py',
    ### '../_base_/models/bts.py', '../_base_/datasets/ddad.py',
    '../_base_/models/bts.py', '../_base_/datasets/argo.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    decode_head=dict(
        final_norm=False,
        min_depth=1e-3,
        ### max_depth=80,
        max_depth=200,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    )
