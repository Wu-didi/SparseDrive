# 消融实验配置说明

本目录包含用于 RA-L 论文消融实验的配置文件。

## 配置文件列表

| 配置文件 | 说明 | 用途 |
|---------|------|------|
| `baseline_no_completion.py` | 禁用所有补全模块 | 对比基准 |
| `no_temporal_completion.py` | 禁用时序补全 | 验证时序补全贡献 |
| `no_motion_compensation.py` | 禁用运动补偿 | 验证运动补偿贡献 |
| `no_planning_guided.py` | 禁用规划引导补全 | 验证规划引导贡献 |
| `missing_rate_template.py` | 不同缺失率测试模板 | 测试不同缺失率 |

## 使用方法

### 1. 训练消融实验

```bash
# Baseline（全部禁用）
bash ./tools/dist_train.sh \
    projects/configs/ablations/baseline_no_completion.py \
    8 --deterministic

# 禁用时序补全
bash ./tools/dist_train.sh \
    projects/configs/ablations/no_temporal_completion.py \
    8 --deterministic

# 禁用运动补偿
bash ./tools/dist_train.sh \
    projects/configs/ablations/no_motion_compensation.py \
    8 --deterministic

# 禁用规划引导
bash ./tools/dist_train.sh \
    projects/configs/ablations/no_planning_guided.py \
    8 --deterministic
```

### 2. 测试不同缺失率

```bash
# 1个相机缺失 (16.7%)
bash ./tools/dist_test.sh \
    projects/configs/ablations/missing_rate_1cam.py \
    /path/to/checkpoint.pth \
    8 --deterministic --eval bbox

# 2个相机缺失 (33.3%)
# 修改配置中的 n_min=2, n_max=2

# 3个相机缺失 (50%)
# 修改配置中的 n_min=3, n_max=3
```

## 预期结果表格

### 消融实验（相机缺失率 33%）

| Method | mAP | NDS | Planning L2 |
|--------|-----|-----|-------------|
| Baseline (VAE only) | X.XX | X.XX | X.XX |
| + Temporal Completion | X.XX | X.XX | X.XX |
| + Motion Compensation | X.XX | X.XX | X.XX |
| + Planning Guided (Full) | X.XX | X.XX | X.XX |

### 不同缺失率对比

| Missing Rate | Baseline | Ours | Improvement |
|-------------|----------|------|-------------|
| 0% (Full) | X.XX | X.XX | - |
| 16.7% (1/6) | X.XX | X.XX | +X.X% |
| 33.3% (2/6) | X.XX | X.XX | +X.X% |
| 50.0% (3/6) | X.XX | X.XX | +X.X% |

## 注意事项

1. **训练时间**：每个配置约需 2-3 天（8×A100）
2. **显存占用**：Full 版本约 40GB/GPU，Baseline 约 25GB/GPU
3. **数据一致性**：确保使用相同的随机种子（seed=42）
4. **Checkpoint**：建议从 stage1 checkpoint 开始微调

## 配置修改建议

如果需要调整参数，可以在继承的基础上覆盖：

```python
_base_ = '../sparsedrive_small_stage2.py'

# 修改时序补全参数
temporal_completion_cfg = dict(
    enable=True,
    queue_length=3,  # 增加历史帧数
    kv_downsample=2,  # 减少下采样（增加精度）
)

model = dict(
    temporal_completion_cfg=temporal_completion_cfg,
)
```
