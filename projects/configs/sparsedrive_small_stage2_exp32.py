_base_ = ["./sparsedrive_small_stage2_exp30.py"]

work_dir = "./work_dirs/sparsedrive_small_stage2_exp32"

load_from = "ckpt/sparsedrive_stage2.pth"

checkpoint_config = dict(interval=14065)

evaluation = dict(interval=14065)

cam_dropout_cfg = dict(
    p_missing=0.6,
    n_min=1,
    n_max=2,
    sticky_fault=True,
    sticky_min_frames=2,
    sticky_max_frames=6,
    curriculum_steps=70325,
    p_missing_start=0.15,
    n_max_start=1,
)

model = dict(cam_dropout_cfg=cam_dropout_cfg)

optimizer = dict(
    type="AdamW",
    lr=2e-6,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.05),
            "img_neck": dict(lr_mult=0.1),
            "depth_branch": dict(lr_mult=0.1),
            "head": dict(lr_mult=0.1),
        }
    ),
)
