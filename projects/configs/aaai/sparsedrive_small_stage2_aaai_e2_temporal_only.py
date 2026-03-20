_base_ = './sparsedrive_small_stage2_aaai_masked_base.py'

# E2: only unfreeze temporal completion on top of the planning-only route.

work_dir = './work_dirs/sparsedrive_small_stage2_aaai_e2_temporal_only'

model = dict(
    frozen_modules=[
        'img_backbone',
        'img_neck',
        'depth_branch',
        'pv_recon',
        'world_model',
        'head.det_head',
        'head.map_head',
    ],
)
