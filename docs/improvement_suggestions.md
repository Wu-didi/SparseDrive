# SparseDrive v2.0 改进建议

## 当前状态

已完成功能：
- ✅ 运动补偿时序补全模块 (Motion-Compensated Temporal Completion)
- ✅ 规划引导补全模块 (Planning-Guided Completion)
- ✅ 相机鲁棒性增强
- ✅ 显存优化版本
- ✅ eval阶段no_grad修复

---

## 发现的问题和改进建议

### 1. 场景切换时历史队列未重置 ⚠️

**问题描述**：
```python
# FeatureQueue 定义了 max_time_interval 但未使用
def __init__(self, queue_length: int = 2, max_time_interval: float = 2.0):
    self.max_time_interval = max_time_interval  # ❌ 未使用
```

在 nuScenes 数据集中，不同场景之间的历史特征应该被清空，否则会用错误的场景历史进行补全。

**改进方案**：

```python
def push(self, feat: torch.Tensor, metas: Dict):
    # 获取时间戳
    timestamp = metas.get('timestamp', 0.0)

    # 检查时间间隔（场景切换检测）
    if len(self.timestamp_queue) > 0:
        time_diff = abs(timestamp - self.timestamp_queue[-1])
        if time_diff > self.max_time_interval:
            # 时间间隔过大，可能是场景切换，清空队列
            self.reset()

    # 添加到队列
    self.feature_queue.append(feat.detach())
    ...
```

**优先级**：⭐⭐⭐ 高（影响准确性）

---

### 2. 配置灵活性不足 🔧

**问题描述**：

`temporal_completion` 的参数在代码中硬编码，不便于实验调整。

**当前代码**：
```python
# sparsedrive.py:1222
self.temporal_completion = MotionCompensatedTemporalCompletion(
    ch_per_scale=temporal_completion_cfg.get('ch_per_scale', [256, 256, 256, 256]),
    embed_dims=temporal_completion_cfg.get('embed_dims', 256),
    queue_length=temporal_completion_cfg.get('queue_length', 2),  # 硬编码
    reference_depths=temporal_completion_cfg.get('reference_depths', [10, 30]),
    ...
)
```

**改进方案**：

在 `sparsedrive_small_stage2.py` 中添加：

```python
# 时序补全配置（可根据显存调整）
temporal_completion_cfg = dict(
    enable=True,
    queue_length=2,           # 历史帧数
    reference_depths=[10, 30],  # 深度假设
    kv_downsample=4,          # Key/Value下采样
    embed_dims=256,
    num_heads=8,
)

model = dict(
    type="SparseDrive",
    ...
    temporal_completion_cfg=temporal_completion_cfg,
)
```

**优先级**：⭐⭐ 中（便于实验）

---

### 3. 文档过时 📄

**问题描述**：

`CAMERA_ROBUSTNESS.md` 描述的是旧版 GRU-based 时序补全，与当前的运动补偿版本不符。

**改进方案**：

更新 `CAMERA_ROBUSTNESS.md` 为当前架构：

```markdown
## 运动补偿时序补全模块

### 核心思想
利用自车运动信息（T_temp2cur）对历史帧特征进行几何对齐，
然后通过跨相机时序注意力进行特征补全。

### 技术实现
- 多深度假设 warp (10m, 30m)
- 可学习的偏移refinement
- 跨相机时序注意力
- Key/Value 4x下采样（显存优化）

### 流程图
历史特征 [B, V, T, C, H, W]
  -> 运动补偿warp (基于T_temp2cur)
  -> 跨相机时序注意力
  -> 空间解码
  -> 门控融合
  -> 补全特征 [B, C, H, W]
```

**优先级**：⭐⭐ 中（文档一致性）

---

### 4. 性能监控缺失 📊

**问题描述**：

没有方便的工具监控训练/推理时的显存和FPS。

**改进方案**：

添加性能统计工具：

```python
# tools/profile_memory.py
import torch
from mmcv import Config

def profile_model(config_path, checkpoint_path=None):
    """
    分析模型的显存占用和推理速度
    """
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model)

    # 显存分析
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # FPS测试
    with torch.no_grad():
        for i in range(100):
            # 推理
            ...

    print(f"显存峰值: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    print(f"FPS: {fps:.2f}")
```

**优先级**：⭐ 低（便利性）

---

### 5. 消融实验配置 🧪

**问题描述**：

为了RA-L投稿，需要快速切换不同的消融实验配置。

**改进方案**：

创建消融实验配置文件：

```python
# projects/configs/ablations/
# - no_temporal.py          # 禁用时序补全
# - no_motion_comp.py       # 禁用运动补偿
# - no_planning_guided.py   # 禁用规划引导
# - baseline.py             # 全部禁用

# 示例: no_temporal.py
_base_ = '../sparsedrive_small_stage2.py'

temporal_completion_cfg = dict(enable=False)

model = dict(
    temporal_completion_cfg=temporal_completion_cfg,
)
```

**优先级**：⭐⭐⭐ 高（论文需要）

---

### 6. 训练稳定性 🔥

**问题描述**：

如果 warp 后的特征有异常值（投影到图像外），可能导致训练不稳定。

**改进方案**：

在 `ImageLevelMotionWarp` 中添加安全检查：

```python
def forward(self, feat, T_temp2cur, lidar2img, img_shape):
    ...
    # Warp
    warped = F.grid_sample(feat, grid, ...)

    # 检查异常值
    if torch.isnan(warped).any() or torch.isinf(warped).any():
        # 返回零填充而不是传播NaN
        return torch.zeros_like(feat)

    # Clamp 到合理范围
    warped = warped.clamp(-10, 10)
    return warped
```

**优先级**：⭐⭐ 中（训练稳定性）

---

## 推荐的实施顺序

### 短期（1-2天）- 为论文实验准备

1. ⭐⭐⭐ **创建消融实验配置** - 方便快速切换实验
2. ⭐⭐⭐ **场景切换检测** - 保证实验准确性

### 中期（1周）- 代码质量提升

3. ⭐⭐ **配置灵活性** - 便于参数调优
4. ⭐⭐ **文档更新** - 保持一致性

### 长期（可选）

5. ⭐ **性能监控工具** - 便利性工具
6. ⭐⭐ **训练稳定性检查** - 防御性编程

---

## 代码质量评估

| 方面 | 评分 | 备注 |
|-----|------|------|
| 功能完整性 | ⭐⭐⭐⭐⭐ | 所有核心功能已实现 |
| 代码规范 | ⭐⭐⭐⭐ | 注释清晰，结构合理 |
| 错误处理 | ⭐⭐⭐ | 有基础检查，可加强 |
| 配置灵活性 | ⭐⭐⭐ | 部分硬编码 |
| 文档完整性 | ⭐⭐⭐ | 有文档但部分过时 |
| 测试覆盖 | ⭐⭐ | 缺少单元测试 |

---

## 总结

当前代码已经很完善，主要改进空间在：
1. **实验便利性**（消融实验配置）
2. **正确性保证**（场景切换检测）
3. **文档更新**（保持与代码一致）

建议优先实施前两项，以支持 RA-L 论文的实验工作。
