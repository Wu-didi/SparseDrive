_base_ = ["./sparsedrive_small_stage2_exp32.py"]

work_dir = "./work_dirs/sparsedrive_small_stage2_exp37_no_ssl"

# No-SSL ablation: keep the exp32 training recipe (cam_dropout + temporal
# completion + flow PV reconstruction), but disable the self-supervised
# dual-branch consistency loss.
# This isolates the contribution of the SSL teacher-student signal.

model = dict(
    ssl_weight=0.0,  # disable SSL dual-branch loss
)
