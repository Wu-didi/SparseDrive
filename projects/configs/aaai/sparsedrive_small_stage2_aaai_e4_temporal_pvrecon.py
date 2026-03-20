_base_ = './sparsedrive_small_stage2_aaai_masked_base.py'

# E4: unfreeze both temporal completion and PV reconstruction.
# Use only if E2/E3 show clear masked-L2 gains.

work_dir = './work_dirs/sparsedrive_small_stage2_aaai_e4_temporal_pvrecon'

model = dict(
    frozen_modules=[
        'img_backbone',
        'img_neck',
        'depth_branch',
        'world_model',
        'head.det_head',
        'head.map_head',
    ],
)
