_base_ = ["./sparsedrive_small_stage2_exp32.py"]

work_dir = "./work_dirs/sparsedrive_small_stage2_exp35_flow"

# Flow-only ablation: keep the exp32 training recipe and feature-space
# PV reconstruction, but remove temporal completion.
temporal_completion_cfg = dict(enable=False)

model = dict(
    temporal_completion_cfg=temporal_completion_cfg,
)
