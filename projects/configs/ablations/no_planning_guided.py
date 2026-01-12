# ================ 消融实验：禁用规划引导补全 ==================
# 用于验证 PlanningGuidedCompletion 模块的贡献

_base_ = '../sparsedrive_small_stage2.py'

# 时序补全保持启用，但禁用规划引导补全
# 注意：需要在 SparseDrive 模型中支持此配置

# 当前实现中，规划引导补全在 forward_train 和 simple_test 中调用
# 可以通过设置 enable=False 来禁用

planning_guided_completion_cfg = dict(
    enable=False,  # 禁用规划引导补全
)

model = dict(
    planning_guided_completion=planning_guided_completion_cfg,
)

# 工作目录
work_dir = './work_dirs/ablation_no_planning_guided'

# 注意：如果模型中未实现 planning_guided_completion_cfg 参数，
# 需要修改 sparsedrive.py 的初始化代码
