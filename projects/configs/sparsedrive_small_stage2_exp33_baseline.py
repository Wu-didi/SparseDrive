_base_ = ["./sparsedrive_small_stage2_exp32.py"]

work_dir = "./work_dirs/sparsedrive_small_stage2_exp33_baseline"

# Baseline ablation: keep the exp32 training recipe, but remove
# temporal completion and feature-space PV reconstruction.
use_pv_recon = False
pv_recon_cfg = dict(type="identity")

temporal_completion_cfg = dict(enable=False)

model = dict(
    pv_recon_cfg=pv_recon_cfg,
    temporal_completion_cfg=temporal_completion_cfg,
)
