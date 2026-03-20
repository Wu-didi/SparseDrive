_base_ = './sparsedrive_small_stage2_exp26.py'

# exp28: 从 exp27 的最佳 planning checkpoint 继续做 masked robustness 微调
# 目标：保持 planning 优势，同时重新放开补全模块，减少 masked eval 下的退化

work_dir = './work_dirs/sparsedrive_small_stage2_exp28'

cam_dropout_cfg = dict(
    p_missing=0.6,
    n_min=1,
    n_max=2,
    sticky_fault=True,
    sticky_min_frames=2,
    sticky_max_frames=6,
    curriculum_steps=1000,
    p_missing_start=0.35,
    n_max_start=1,
)

pv_recon_cfg = dict(
    type='flow',
    ch_per_scale=[256, 256, 256, 256],
    hidden_channels=128,
    time_embed_dim=128,
    lambda_flow=0.003,
    num_integration_steps=4,
    num_cameras=6,
    use_camera_embed=True,
    detach_inputs=True,
)

model = dict(
    test_cam_missing=True,
    cam_dropout_cfg=cam_dropout_cfg,
    pv_recon_cfg=pv_recon_cfg,
    frozen_modules=[
        'img_backbone',
        'img_neck',
        'depth_branch',
        'world_model',
        'head.det_head',
        'head.map_head',
    ],
)

optimizer = dict(
    lr=2e-6,
)

runner = dict(
    max_iters=28130,
)

checkpoint_config = dict(
    interval=7032,
)

evaluation = dict(
    interval=7032,
)

load_from = 'work_dirs/sparsedrive_small_stage2_exp27/iter_42195.pth'
