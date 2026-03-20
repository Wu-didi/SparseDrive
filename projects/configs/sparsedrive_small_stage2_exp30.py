_base_ = './sparsedrive_small_stage2_exp29.py'

# exp30: paper-minimal masked-robustness setup.
# Keep temporal completion and PV reconstruction as the main method,
# while disabling auxiliary world-model and planning-guided completion
# to get a cleaner, easier-to-explain contribution.

work_dir = './work_dirs/sparsedrive_small_stage2_exp30'

model = dict(
    world_model_cfg=dict(
        enabled=False,
    ),
    planning_guided_completion_cfg=dict(
        enable=False,
        trajectory_source='none',
    ),
)
