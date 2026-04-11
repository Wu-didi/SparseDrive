_base_ = ["./sparsedrive_small_stage2_exp32.py"]

work_dir = "./work_dirs/sparsedrive_small_stage2_exp36_mask_only"

# Mask-only ablation: keep cam_dropout from exp32, but disable ALL
# reconstruction modules AND the SSL dual-branch.
# This isolates the effect of camera dropout as data augmentation alone.
use_pv_recon = False
pv_recon_cfg = dict(type="identity")

temporal_completion_cfg = dict(enable=False)

model = dict(
    pv_recon_cfg=pv_recon_cfg,
    temporal_completion_cfg=temporal_completion_cfg,
    ssl_weight=0.0,  # disable SSL dual-branch (skip 2nd backbone pass)
)
