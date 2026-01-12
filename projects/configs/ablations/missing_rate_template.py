# ================ 实验：不同相机缺失率测试 ==================
# 用于测试在不同缺失率下的性能

_base_ = '../sparsedrive_small_stage2.py'

# 配置不同的相机缺失率
# 修改 RandCamMask 参数

# 缺失率 16.7% (1/6 相机)
cam_dropout_cfg = dict(
    p_missing=0.6,     # 每个样本触发缺失的概率
    n_min=1,           # 最少缺失 1 个相机
    n_max=1,           # 最多缺失 1 个相机
    train_only=False,  # 训练和测试都启用
    seed=42,
)

# 对于 33.3% (2/6)，设置 n_min=2, n_max=2
# 对于 50% (3/6)，设置 n_min=3, n_max=3

model = dict(
    # 注意：需要在 SparseDrive.__init__ 中支持此配置
    # cam_dropout_cfg=cam_dropout_cfg,
)

# 工作目录
work_dir = './work_dirs/missing_rate_1cam'

# 使用方法：
# 1. 复制此文件为 missing_rate_2cam.py, missing_rate_3cam.py
# 2. 修改 n_min, n_max 参数
# 3. 修改 work_dir
