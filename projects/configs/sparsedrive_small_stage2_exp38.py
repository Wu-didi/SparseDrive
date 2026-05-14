_base_ = ["./sparsedrive_small_stage2_exp32.py"]

work_dir = "./work_dirs/sparsedrive_small_stage2_exp38"

# exp38: fix sticky_fault for consecutive-frame camera dropout
# scene_token is now passed via meta_keys so RandCamMask can
# cache dropout masks across frames within the same scene.
# sticky_max_frames reduced to 3 (1.5s) to stay within queue_length.

cam_dropout_cfg = dict(
    p_missing=0.6,
    n_min=1,
    n_max=2,
    sticky_fault=True,
    sticky_min_frames=2,
    sticky_max_frames=3,
    curriculum_steps=70325,
    p_missing_start=0.15,
    n_max_start=1,
)

model = dict(cam_dropout_cfg=cam_dropout_cfg)
