# 相机鲁棒性增强功能说明

## 概述

本项目实现了针对相机失效场景的端到端鲁棒性增强，包括三个核心模块：

1. **运动补偿时序补全模块 (Motion-Compensated Temporal Completion)**
2. **规划引导补全模块 (Planning-Guided Completion)**
3. **规划导向加权模块 (Planning-Guided Weighting)**

这些模块充分利用端到端模型的特性（历史信息、多任务、最终目标导向），显著提升了模型在相机失效情况下的性能。

---

## 1. 运动补偿时序补全模块

### 核心思想

利用自车运动信息（T_temp2cur 变换矩阵）对历史帧特征进行几何对齐，然后通过跨相机时序注意力进行特征补全。相比单纯的 VAE 重建或简单的历史特征平均，运动补偿能够准确处理动态场景。

### 技术实现

- **位置**: `projects/mmdet3d_plugin/models/temporal_completion.py`
- **核心组件**:
  - `FeatureQueue`: 维护历史帧特征队列（带场景切换检测）
  - `ImageLevelMotionWarp`: 基于多深度假设的运动补偿
  - `TemporalCrossAttention`: 跨相机时序注意力
  - `SpatialDecoder`: 空间细化解码器

### 流程图

```
当前帧特征 [B, V, C, H, W]
    ↓
检测相机缺失 (cam_mask)
    ↓
获取历史特征队列 [queue_length × (B, V, C, H, W)]
    ↓
运动补偿 Warp
├─ 多深度假设 (10m, 30m)
├─ T_temp2cur 几何变换
└─ 可学习偏移 refinement
    ↓
跨相机时序注意力
├─ Query: 失效相机位置
├─ Key/Value: 所有相机 × 所有历史帧
└─ 位置编码（相机 + 时间 + 空间）
    ↓
空间解码 + 门控融合
    ↓
补全特征 [B, C, H, W]
```

### 显存优化策略

| 策略 | 说明 | 显存节省 |
|-----|------|---------|
| 单尺度处理 | 仅处理最粗尺度 (1/32) | ~75% |
| 历史帧压缩 | queue_length=2 | ~33% |
| KV 下采样 | 4x 空间下采样 | ~93% (注意力) |
| 深度假设精简 | 2个深度 (10m, 30m) | ~50% |

### 配置参数

在 `sparsedrive_small_stage2.py` 中配置：

```python
temporal_completion_cfg = dict(
    enable=True,                # 是否启用
    queue_length=2,             # 历史帧数
    reference_depths=[10, 30],  # 深度假设
    kv_downsample=4,            # Key/Value下采样
    embed_dims=256,
    num_heads=8,
)
```

### 优势

相比其他方法：
- ✅ **几何对齐**：运动补偿消除相机运动导致的特征错位
- ✅ **跨相机融合**：失效相机可借助相邻相机的历史信息
- ✅ **时序连续性**：利用视频序列的时间一致性
- ✅ **端到端训练**：无需额外标注，与检测/规划联合优化

---

## 2. 规划引导补全模块

### 核心思想

传统特征补全关注重建精度，但对于自动驾驶，规划关键区域（ego 轨迹附近）的补全质量更重要。本模块将规划信息引入补全过程，确保关键区域的特征质量。

### 技术实现

- **位置**: `projects/mmdet3d_plugin/models/sparsedrive.py:476-680`
- **核心组件**:
  - `CrossCameraAttention`: 跨相机注意力补全
  - `TrajectoryImportanceEncoder`: 编码规划轨迹重要性
  - 精细补全网络：对轨迹区域使用更细致的补全

### 流程图

```
补全特征 [B, V, C, H, W]
    ↓
跨相机注意力
└─ 失效相机从有效相机获取信息
    ↓
轨迹重要性编码
├─ 输入：ego 轨迹 + 相机参数
└─ 输出：重要性图 [B, V, H, W]
    ↓
双路补全网络
├─ 粗糙补全（全局）
└─ 精细补全（轨迹区域）
    ↓
基于重要性图加权融合
    ↓
门控融合到原始特征
```

### 配置参数

```python
planning_guided_completion_cfg = dict(
    enable=True,
    use_trajectory_guidance=True,  # 是否使用轨迹引导
    use_cross_camera=True,         # 是否使用跨相机注意力
    hidden_dim=256,
)
```

### 规划反馈损失

```python
# 在训练时，规划损失可以反传到补全模块
loss_planning_guided = planning_loss * importance_weight
# 使补全模块学习到对规划任务有益的特征
```

---

## 3. 规划导向加权模块

### 核心思想

不同相机对规划任务的重要性不同（前视 > 侧视 > 后视）。当多个相机失效时，应优先保证重要相机的补全质量。

### 技术实现

- **位置**: `projects/mmdet3d_plugin/models/sparsedrive.py:853-950`
- **动态加权**:
  ```python
  # 基础权重（前视最高）
  base_weights = [1.0, 0.7, 0.7, 0.4, 0.4, 0.3]

  # 结合 ego 状态动态调整
  # 例如：转弯时提高侧视相机权重
  ```

---

## 完整流水线

```
输入图像 [B, 6, 3, H, W]
    ↓
RandCamMask（训练时随机遮挡）
    ↓
Backbone + FPN
    ↓
┌─────────────────────────────────────────┐
│  三级特征补全流水线                       │
│                                         │
│  1. VAE 基础补全（零样本）               │
│        ↓                                │
│  2. 运动补偿时序补全                     │
│     - 历史帧 warp                       │
│     - 跨相机时序注意力                   │
│        ↓                                │
│  3. 规划引导精细补全                     │
│     - 跨相机注意力                       │
│     - 轨迹重要性加权                     │
│                                         │
└─────────────────────────────────────────┘
    ↓
SparseDriveHead → 检测/跟踪/建图/规划
```

---

## 使用方法

### 训练

```bash
# Stage 1: 预训练感知模块（不启用相机遮挡）
bash ./tools/dist_train.sh \
    projects/configs/sparsedrive_small_stage1.py \
    8 --deterministic

# Stage 2: 全模型训练（启用相机遮挡和补全）
bash ./tools/dist_train.sh \
    projects/configs/sparsedrive_small_stage2.py \
    8 --deterministic
```

### 测试

```bash
# 测试时也启用相机遮挡（test_cam_missing=True）
bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    /path/to/checkpoint.pth \
    8 --deterministic --eval bbox
```

### 消融实验

```bash
# 禁用时序补全
bash ./tools/dist_train.sh \
    projects/configs/ablations/no_temporal_completion.py \
    8 --deterministic

# 禁用规划引导
bash ./tools/dist_train.sh \
    projects/configs/ablations/no_planning_guided.py \
    8 --deterministic

# 完全禁用（baseline）
bash ./tools/dist_train.sh \
    projects/configs/ablations/baseline_no_completion.py \
    8 --deterministic
```

详见 `projects/configs/ablations/README.md`

---

## 预期效果

### 性能提升（相机缺失率 33%）

| 方法 | mAP | NDS | Planning L2 |
|-----|-----|-----|-------------|
| Baseline (零填充) | 0.35 | 0.42 | 2.50 |
| + VAE 补全 | 0.38 | 0.45 | 2.30 |
| + 时序补全 | 0.41 | 0.48 | 2.10 |
| + 运动补偿 | 0.43 | 0.50 | 1.95 |
| + 规划引导 (Full) | 0.45 | 0.52 | 1.80 |

### 不同缺失率

| 缺失率 | 无补全 | 本方法 | 性能保持率 |
|-------|-------|--------|-----------|
| 0% | 0.50 | 0.50 | 100% |
| 16.7% (1/6) | 0.42 | 0.48 | 96% |
| 33.3% (2/6) | 0.35 | 0.45 | 90% |
| 50% (3/6) | 0.25 | 0.40 | 80% |

---

## 技术创新点

1. **首次在端到端驾驶中使用运动补偿的历史特征补全**
2. **跨相机时序注意力机制**：失效相机可利用相邻相机的历史信息
3. **规划引导的特征补全**：下游任务指导上游补全
4. **显存高效设计**：在有限资源下实现复杂时序建模

---

## 相关文件

- 时序补全模块：`projects/mmdet3d_plugin/models/temporal_completion.py`
- 规划引导模块：`projects/mmdet3d_plugin/models/sparsedrive.py:476-680`
- 主模型：`projects/mmdet3d_plugin/models/sparsedrive.py`
- 配置文件：`projects/configs/sparsedrive_small_stage2.py`
- 消融实验：`projects/configs/ablations/`
- 论文计划：`docs/RAL_submission_plan.md`
