# ================ 消融实验：禁用时序补全 ==================
# 用于验证运动补偿时序补全模块的贡献

_base_ = '../sparsedrive_small_stage2.py'

# 禁用时序补全模块
temporal_completion_cfg = dict(
    enable=False,  # 关闭时序补全
)

model = dict(
    temporal_completion_cfg=temporal_completion_cfg,
)

# 工作目录
work_dir = './work_dirs/ablation_no_temporal'
