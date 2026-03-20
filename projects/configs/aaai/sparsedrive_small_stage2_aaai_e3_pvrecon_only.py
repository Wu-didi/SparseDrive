_base_ = './sparsedrive_small_stage2_aaai_masked_base.py'

# E3: only unfreeze PV reconstruction on top of the planning-only route.

work_dir = './work_dirs/sparsedrive_small_stage2_aaai_e3_pvrecon_only'

model = dict(
    frozen_modules=[
        'img_backbone',
        'img_neck',
        'depth_branch',
        'world_model',
        'temporal_completion',
        'head.det_head',
        'head.map_head',
    ],
)
