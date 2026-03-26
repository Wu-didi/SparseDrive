_base_ = ["./sparsedrive_small_stage2_exp32.py"]

work_dir = "./work_dirs/sparsedrive_small_stage2_exp34_temporal"

# Temporal-only ablation: keep the exp32 training recipe and temporal
# completion, but remove feature-space PV reconstruction.
use_pv_recon = False
pv_recon_cfg = dict(type="identity")

model = dict(
    pv_recon_cfg=pv_recon_cfg,
)
