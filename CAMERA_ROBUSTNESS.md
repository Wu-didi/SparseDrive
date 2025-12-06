# 相机鲁棒性增强功能说明

## 概述

本项目实现了针对相机失效场景的端到端鲁棒性增强，包括两个核心模块：

1. **时序补全模块 (Temporal Feature Completion)**
2. **规划导向加权模块 (Planning-Guided Weighting)**

这些模块充分利用端到端模型的特性（历史信息、多任务、最终目标导向），显著提升了模型在相机失效情况下的性能。

---

## 1. 时序补全模块

### 核心思想

端到端模型维护了历史帧信息（`queue_length=4`），当某个相机失效时，可以利用该相机的历史轨迹预测当前帧的特征，而不是仅依赖单帧的VAE重建。

### 技术实现

- **位置**: `projects/mmdet3d_plugin/models/sparsedrive.py:234-354`
- **架构**:
  - 为每个特征尺度配备独立的GRU预测器
  - 输入：历史T帧的相机特征序列
  - 输出：当前帧该相机的预测特征

```python
# 时序补全流程
历史特征 [T, B, C, H, W]
  -> 全局池化 [T, B, C]
  -> GRU [B, T, C] -> [B, H]
  -> 解码器 [B, H] -> [B, C]
  -> 扩展 [B, C, H, W]
```

### 优势

相比单帧VAE：
- ✅ 利用时间连续性，预测更准确
- ✅ 符合真实场景（车辆运动连续）
- ✅ 无需额外标注，端到端训练

### 使用方式

**默认启用**，无需额外配置。如需禁用：

```python
# 在 sparsedrive.py 的初始化中
self.temporal_completion = TemporalFeatureCompletion(
    ch_per_scale=[256, 256, 256, 256],
    hidden_dim=128,
    num_layers=2,
    enable=False,  # 设为False禁用
)
```

---

## 2. 规划导向加权模块

### 核心思想

不是所有相机对规划同等重要：
- 前视相机 (CAM_FRONT)：最重要（规划主要看前方）
- 前左右相机：中等重要（变道、转弯）
- 后视相机：较低重要（主要用于盲区监控）

对重要相机的特征补全应该有更高的精度要求。

### 技术实现

- **位置**: `projects/mmdet3d_plugin/models/sparsedrive.py:357-458`
- **权重设计**:

| 相机位置 | 基础权重 | 说明 |
|---------|---------|------|
| CAM_FRONT | 2.0 | 前视，最重要 |
| CAM_FRONT_LEFT/RIGHT | 1.5 | 前左右，中等 |
| CAM_BACK_LEFT/RIGHT | 1.0 | 后左右，一般 |
| CAM_BACK | 0.5 | 后视，最低 |

- **动态调整**: 根据自车状态（速度、加速度）动态调整权重
  - 速度越快，前方权重越高
  - 倒车时，后视权重提升

### 使用方式

**默认启用**，在VAE重建损失中自动应用权重。

自定义权重：
```python
self.planning_weighting = PlanningGuidedWeighting(
    num_cameras=6,
    use_ego_state=True,
    base_weights=[3.0, 2.0, 2.0, 1.0, 1.0, 0.3],  # 自定义权重
)
```

---

## 训练配置

### 当前设置

```python
# sparsedrive_small_stage2.py

# 1. 相机随机遮挡
model.cam_dropout:
  p_missing=0.6      # 60%概率触发相机失效
  n_min=1, n_max=2   # 每次遮挡1-2个相机

# 2. 测试时相机遮挡（与训练一致）
model.test_cam_missing=True

# 3. 时序补全（默认启用）
model.temporal_completion.enable=True

# 4. 规划导向加权（默认启用）
model.planning_weighting.use_ego_state=True
```

### 训练流程

```
训练时：
  1. 随机遮挡1-2个相机
  2. 提取masked特征
  3. 时序补全（利用历史3帧）
  4. VAE补全（加规划导向权重）
  5. 计算任务loss（检测、地图、规划等）

测试时：
  1. 同样的相机遮挡策略
  2. 同样的时序补全
  3. 同样的VAE补全
  -> 训练测试一致！
```

---

## 效果验证

### 评估指标

建议关注以下指标在相机失效下的变化：

1. **检测**: NDS, mAP
2. **跟踪**: AMOTA, AMOTP
3. **地图**: mAP
4. **运动预测**: minADE, minFDE
5. **规划**: L2 误差, 碰撞率 (最关键！)

### 对比实验

| 配置 | 训练遮挡 | 测试遮挡 | 时序补全 | 规划加权 | 说明 |
|-----|---------|---------|---------|---------|------|
| Baseline | ❌ | ❌ | ❌ | ❌ | 无鲁棒性训练 |
| VAE-only | ✅ | ✅ | ❌ | ❌ | 仅VAE补全 |
| +Temporal | ✅ | ✅ | ✅ | ❌ | 加时序补全 |
| Full (当前) | ✅ | ✅ | ✅ | ✅ | 完整方案 |

### 测试脚本

```bash
# 1. 无遮挡baseline（理想情况）
bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    work_dirs/sparsedrive_small_stage2/latest.pth \
    1 --eval bbox \
    --cfg-options model.test_cam_missing=False

# 2. 1个相机失效
bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    work_dirs/sparsedrive_small_stage2/latest.pth \
    1 --eval bbox \
    --cfg-options \
    model.test_cam_missing=True \
    model.cam_dropout.p_missing=1.0 \
    model.cam_dropout.n_min=1 \
    model.cam_dropout.n_max=1

# 3. 2个相机失效
bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    work_dirs/sparsedrive_small_stage2/latest.pth \
    1 --eval bbox \
    --cfg-options \
    model.test_cam_missing=True \
    model.cam_dropout.p_missing=1.0 \
    model.cam_dropout.n_min=2 \
    model.cam_dropout.n_max=2
```

---

## 进一步改进方向

### 1. BEV空间补全

当前在图像特征空间补全，可以改为在BEV空间：
- BEV是以自车为中心，更符合规划需求
- 相机失效在BEV中是局部缺失，更容易补全

### 2. 任务级互补

利用多任务信息互补：
- 用地图信息推断车辆可能位置
- 用跟踪历史预测当前检测
- 用运动预测辅助规划

### 3. 不确定性估计

输出规划的不确定性：
- 相机失效越多，不确定性越高
- 高不确定性时采用更保守的规划

### 4. 对抗式训练

不是随机遮挡，而是找对规划影响最大的相机组合进行训练。

---

## 代码结构

```
projects/mmdet3d_plugin/models/sparsedrive.py
├── RandCamMask (33-101)              # 随机相机遮挡
├── PVReconVAE (160-228)              # VAE特征补全
├── TemporalFeatureCompletion (234-354)  # 时序补全 ⭐新增
├── PlanningGuidedWeighting (357-458)    # 规划导向加权 ⭐新增
├── LightDreamerRSSM (461-636)        # 世界模型
└── SparseDrive (640-1062)            # 主模型
    ├── __init__: 初始化所有模块
    ├── forward_train: 训练流程（集成时序补全+加权）
    └── simple_test: 测试流程（集成时序补全）
```

---

## 常见问题

**Q: 时序补全需要多少历史帧？**

A: 默认3帧（`max_history_length=3`），可以调整。历史越长，预测越准，但计算开销越大。

**Q: 规划加权会影响其他任务吗？**

A: 不会。只影响VAE重建损失，检测、跟踪等任务的loss不受影响。

**Q: 测试时必须开启相机遮挡吗？**

A: 建议开启（`test_cam_missing=True`），保持训练测试一致。如果想测试理想情况，可以关闭。

**Q: 如何确认新功能生效？**

A: 查看训练日志，应该看到时序补全和规划加权模块的参数被更新。可以打印 `importance_weights` 查看权重分布。

---

## 参考

- 原始SparseDrive论文: https://arxiv.org/abs/2405.19620
- Dreamer (世界模型): https://arxiv.org/abs/1912.01603
- VAD (端到端AD): https://github.com/hustvl/VAD
