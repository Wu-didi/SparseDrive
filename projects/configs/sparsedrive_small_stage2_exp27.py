_base_ = './sparsedrive_small_stage2_exp26.py'

# exp27: 从 exp26 的最佳 planning checkpoint 继续微调
# 目标：冻结感知/检测/建图与 flow 重建主干，只优化 planning 相关模块

work_dir = './work_dirs/sparsedrive_small_stage2_exp27'

model = dict(
    test_cam_missing=True,
    frozen_modules=[
        'img_backbone',
        'img_neck',
        'depth_branch',
        'pv_recon',
        'world_model',
        'temporal_completion',
        'head.det_head',
        'head.map_head',
    ],
)

optimizer = dict(
    lr=3e-6,
)

runner = dict(
    max_iters=70325,
)

checkpoint_config = dict(
    interval=14065,
)

evaluation = dict(
    interval=14065,
)

load_from = 'work_dirs/sparsedrive_small_stage2_exp26/iter_70325.pth'
