# 快速测试eval流程的配置
# 从stage2配置继承，只修改关键参数
_base_ = ['./sparsedrive_small_stage2.py']

# 使用trainval数据集
version = 'trainval'

# 训练相关：只训练1个iteration
total_batch_size = 1  # 减小batch size加速
num_gpus = 1
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = 1  # 强制设置为1
num_epochs = 1

# 重要：在第1个iteration后就进行evaluation
evaluation = dict(
    interval=1,  # 每1个iteration就eval一次
    eval_mode=dict(
        with_bbox=True,
        with_map=True,
        with_motion=True,
        with_planning=True,
        tracking_threshold=0.2,
        motion_threshhold=0.2,
    )
)

# checkpoint配置
checkpoint_config = dict(interval=1)  # 每1个iteration保存一次

# runner配置：总共只运行2个iteration（1个训练+1个eval）
runner = dict(type='IterBasedRunner', max_iters=2)

# 日志配置
log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)

# 从已有checkpoint恢复（这样不需要从头训练）
resume_from = 'work_dirs/sparsedrive_small_stage2/iter_70320.pth'
load_from = None

# 工作目录
work_dir = './work_dirs/quickeval_test'

# 数据集配置：使用trainval版本，但减少workers加速启动
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,  # 减少worker加速启动
)

print("=" * 80)
print("快速测试配置已加载")
print(f"- 只运行 2 个iteration（1个训练 + 1个eval）")
print(f"- Eval interval: 1")
print(f"- 使用完整trainval数据集")
print(f"- Resume from: {resume_from}")
print("- 注意：eval会遍历完整验证集（6019个样本），需要等待几分钟")
print("=" * 80)
