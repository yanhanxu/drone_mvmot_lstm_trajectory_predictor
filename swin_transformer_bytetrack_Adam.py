

_base_ = [
    '../../_base_/models/new_faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/test_challenge.py', '../../_base_/default_runtime.py'
]


img_scale = (1333, 800)
samples_per_gpu = 4

model = dict(
    type='ByteTrack',
    detector=dict(
        neck=dict(
            type='FPN_CARAFE',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            start_level=0,
            end_level=-1,
            norm_cfg=None,
            act_cfg=None,
            order=('conv', 'norm', 'act'),
            upsample_cfg=dict(
                type='carafe',
                up_kernel=5,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64)
                ),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                num_classes=3,
                reg_class_agnostic=False,
                )
                ),

        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
           'D:\Multi-Drone-Multi-Object-Detection-and-Tracking-main\demo\checkpoint/faster_rcnn_r50_fpn_carafe_1x_full_mdmt/btorbm_epoch_48.pth'  # noqa: E501
        )

        ),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.7, tentative=0.5),
        num_frames_retain=30),
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64),
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
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))


optimizer = dict(lr=0.002, paramwise_cfg=dict(norm_decay_mult=0.))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1.0 / 5000,
    step=[40, 55])
total_epochs = 60
runner = dict(type='EpochBasedRunner', max_epochs=60)

checkpoint_config = dict(interval=1)
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

fp16 = dict(loss_scale=dict(init_scale=512.))



