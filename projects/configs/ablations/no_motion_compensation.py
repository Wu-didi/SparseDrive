# ================ 消融实验：禁用运动补偿 ==================
# 用于验证运动补偿 warp 的贡献
# 时序补全仍启用，但不使用 T_temp2cur 进行几何对齐

_base_ = '../sparsedrive_small_stage2.py'

# 注意：当前架构下，运动补偿是时序补全的核心部分
# 要禁用运动补偿，需要使用简单的历史特征平均
# 或者直接禁用时序补全，改用其他补全策略

# 建议：使用规划引导补全但禁用时序补全
temporal_completion_cfg = dict(
    enable=False,  # 禁用包含运动补偿的时序补全
)

model = dict(
    temporal_completion_cfg=temporal_completion_cfg,
)

# 工作目录
work_dir = './work_dirs/ablation_no_motion_comp'
