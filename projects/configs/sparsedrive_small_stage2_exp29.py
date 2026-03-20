_base_ = './sparsedrive_small_stage2_exp28.py'

# exp29: keep the masked-robustness fine-tuning recipe from exp28, but
# switch planning-guided completion to a stable GT-trajectory default and
# propagate temporal completion residuals to all scales.

work_dir = './work_dirs/sparsedrive_small_stage2_exp29'

model = dict(
    temporal_completion_cfg=dict(
        cross_scale_residual=0.5,
    ),
    planning_guided_completion_cfg=dict(
        trajectory_source='gt',
    ),
)
