# ================ 消融实验：Baseline（全部禁用）==================
# 用于对比所有补全模块的总体贡献
# 禁用：时序补全 + 规划引导补全

_base_ = '../sparsedrive_small_stage2.py'

# 禁用时序补全
temporal_completion_cfg = dict(
    enable=False,
)

# 禁用规划引导补全
planning_guided_completion_cfg = dict(
    enable=False,
)

model = dict(
    temporal_completion_cfg=temporal_completion_cfg,
    planning_guided_completion=planning_guided_completion_cfg,
)

# 工作目录
work_dir = './work_dirs/ablation_baseline'

# 注意：
# - 此配置下仅使用 VAE 进行基础补全
# - 可用于评估新增模块的总体提升
