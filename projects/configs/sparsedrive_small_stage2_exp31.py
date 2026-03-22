_base_ = ["./sparsedrive_small_stage2_exp30.py"]

work_dir = "./work_dirs/sparsedrive_small_stage2_exp31"

# Continue from the best exp30 checkpoint, but restart with a gentler LR.
load_from = "./work_dirs/sparsedrive_small_stage2_exp30/iter_70325.pth"

optimizer = dict(lr=1e-6)
