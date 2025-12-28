# Planning-Guided Completion: 规划引导的特征补全

## 1. 研究背景

### 1.1 问题描述
在自动驾驶场景中，相机可能因为硬件故障、遮挡、污染等原因失效。现有方法通常采用简单的特征补全策略（如 VAE 重建、时序预测），但这些方法的优化目标是**重建原始特征**，而非**提升下游任务性能**。

### 1.2 核心洞察
SparseDrive 是一个端到端的自动驾驶模型，包含感知、预测、规划等模块。我们可以利用这一特性：**让规划任务的性能来指导特征补全的学习**。

## 2. 创新点

### 2.1 端到端规划反馈 (End-to-End Planning Feedback)

**核心思想**：补全模块的优化目标不仅是重建特征，更重要的是产生对规划任务有帮助的特征。

```
传统方法:
  补全特征 → MSE Loss(补全特征, GT特征) → 优化补全模块

我们的方法:
  补全特征 → Head → 规划预测 → Planning Loss → 梯度反传 → 优化补全模块
```

**实现方式**：
- 补全后的特征不使用 `detach()`，保持梯度连接
- 规划损失可以直接反传到补全模块的参数
- 补全模块学习的是"对规划有用"的特征，而非简单的特征重建

### 2.2 轨迹引导的区域性补全 (Trajectory-Guided Regional Completion)

**核心思想**：规划轨迹经过的区域对于规划决策更重要，应该获得更精确的补全。

**实现方式**：
- 根据 GT 轨迹方向计算各相机的重要性
- 前进 → 前视相机重要；左转 → 左侧相机重要
- 轨迹区域使用精细补全网络，其他区域使用粗糙补全网络

```python
# 相机重要性分配
前进: FRONT(2.0), FRONT_LEFT(1.0), FRONT_RIGHT(1.0)
左转: FRONT_LEFT(+1.5), BACK_LEFT(+1.0)
右转: FRONT_RIGHT(+1.5), BACK_RIGHT(+1.0)
```

### 2.3 跨相机注意力 (Cross-Camera Attention)

**核心思想**：利用有效相机的信息来补全缺失相机，因为相邻相机存在视野重叠。

**实现方式**：
- 缺失相机作为 Query，有效相机作为 Key/Value
- 多头注意力机制 + 相机位置编码
- 残差连接，缩放因子 0.1 保证训练稳定

```
CrossCameraAttention:
  Query: 缺失相机特征 + 位置编码
  Key/Value: 有效相机特征 + 位置编码
  Output: 注意力加权的全局信息
```

### 2.4 门控融合机制 (Gated Fusion)

**核心思想**：让模型自己学习原始特征和补全特征的最优混合比例。

```python
gate = sigmoid(Conv([原始特征, 补全特征]))  # 学习的门控值
output = 原始特征 * (1 - gate) + 补全特征 * gate
```

**初始化策略**：
- 门控 bias 初始化为 -2，使 sigmoid(-2) ≈ 0.1
- 初期：90% 原始特征 + 10% 补全特征
- 训练后：模型自动学习最优比例

## 3. 技术细节

### 3.1 模块架构

```
PlanningGuidedCompletion
├── completion_nets (粗糙补全，每个尺度独立)
├── fine_completion_nets (精细补全，用于轨迹区域)
├── gate_nets (门控网络)
├── cross_cam_attention (跨相机注意力)
└── compute_trajectory_importance (轨迹重要性计算)
```

### 3.2 损失函数

```python
PlanningFeedbackLoss:
├── SmoothL1 Loss (兼顾 L1 的鲁棒性和 L2 的平滑性)
├── log1p 软缩放 (代替硬裁剪，保持梯度连续)
└── 重要性加权 (轨迹区域权重更高)
```

### 3.3 训练稳定性保护

| 措施 | 说明 |
|------|------|
| SmoothL1 Loss | 小误差用 L2，大误差用 L1 |
| log1p 软缩放 | 大损失值被压缩但梯度不消失 |
| 小值初始化 | 补全网络最后一层 std=0.01 |
| 门控负偏置 | 初始 gate ≈ 0.1 |
| 残差缩放 | CrossCameraAttention 输出 × 0.1 |
| 梯度裁剪 | max_norm=25 (配置文件) |

## 4. 与现有方法对比

| 方法 | 优化目标 | 是否利用下游任务 | 区域性补全 |
|------|----------|------------------|------------|
| VAE 重建 | 重建损失 | ✗ | ✗ |
| 时序预测 | 预测损失 | ✗ | ✗ |
| **Ours** | **规划损失** | **✓** | **✓** |

## 5. 代码位置

```
projects/mmdet3d_plugin/models/sparsedrive.py
├── PlanningGuidedCompletion (line 475-650)
├── CrossCameraAttention (line 732-820)
├── TrajectoryImportanceEncoder (line 823-851)
├── PlanningFeedbackLoss (line 879-944)
└── SparseDrive.forward_train (集成，line 1441-1495)
```

## 6. 使用方式

训练时自动启用，观察以下损失项：
- `loss_completion_recon`: 补全重建损失

## 7. 未来改进方向

1. **可学习的轨迹重要性**：用神经网络替代启发式规则
2. **多模态轨迹**：考虑多条可能的规划轨迹
3. **不确定性建模**：对补全结果的置信度进行估计
4. **跨帧注意力**：结合时序信息和跨相机信息
