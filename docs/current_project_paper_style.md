# 面向相机失效鲁棒性的 SparseDrive 增强项目介绍

## 摘要

本项目以 **SparseDrive** 为基础，围绕“多相机端到端自动驾驶在真实部署中容易遭遇相机失效、遮挡、污染和掉线”这一问题，构建了一条面向 **相机缺失鲁棒性** 的增强路线。与原始 SparseDrive 假设多视角输入完整不同，当前项目在图像 Backbone 与多任务 Head 之间引入了随机相机失效模拟、运动补偿时序补全、规划导向加权、VAE 视角补全、规划引导精细补全，以及 teacher-student 特征一致性和轻量世界模型监督等模块，目标是在相机缺失条件下尽量保持检测、建图、运动预测与规划能力。整体上，这个项目已经从“想法”推进到“可训练、可评测、可做消融”的研究原型阶段，但从现有可追溯实验结果看，性能和训练稳定性仍未达到原始 SparseDrive 的发布水平。

---

## 1. 研究动机

端到端自动驾驶模型通常建立在多相机输入完整可用的前提下，但这一假设在真实车端并不稳固。相机可能因为雨雪泥污、曝光异常、局部遮挡、硬件故障或通信异常而短时或持续失效。一旦视角缺失，基于 BEV 或稀疏场景表示的多任务系统会同时受到影响：检测召回下降、在线地图断裂、运动预测不稳定、规划安全性变差。

原始 SparseDrive 已经证明，基于稀疏场景表示可以高效统一检测、跟踪、建图、运动预测与规划；但它并未系统解决“传感器不完整输入”问题。因此，当前项目的核心问题可以概括为：

> 如何在尽量保留 SparseDrive 原有统一框架和效率优势的前提下，使模型在相机缺失场景中仍保持可用的端到端驾驶能力？

---

## 2. 项目做了什么

当前项目并不是重新设计一套新的端到端自动驾驶框架，而是在 **SparseDrive 主干不变** 的条件下，新增了一条“相机鲁棒性增强链路”。其核心改动位于 `projects/mmdet3d_plugin/models/sparsedrive.py` 和 `projects/mmdet3d_plugin/models/temporal_completion.py`，主要包括以下五部分：

1. **随机相机失效建模**  
   训练时通过 `RandCamMask` 对 6 路相机进行随机 blackout，支持：
   - 缺失概率控制；
   - 一次失效 1 到 2 个相机；
   - sticky fault 连续掉线；
   - curriculum 式从轻扰动到重扰动的训练策略。

2. **运动补偿时序补全**  
   利用历史帧特征和自车位姿变换，对缺失相机当前特征做几何对齐与时序补全。

3. **规划导向的 VAE 视角补全**  
   对缺失视角特征进行重建，但不是平均地看待所有视角，而是按规划相关性对相机赋权。

4. **规划引导精细补全**  
   使用 ego 未来轨迹引导补全过程，优先修复对规划决策更关键的区域和视角。

5. **辅助监督与端到端反馈**  
   通过 teacher-student 特征一致性、自监督世界模型和规划反馈损失，让补全模块不只是“看起来像原特征”，而是“真正对下游驾驶任务有用”。

---

## 3. 论文式创新点概括

### 创新点 1：将相机鲁棒性问题嵌入端到端多任务框架

项目没有把缺失恢复当作一个独立的预处理任务，而是把补全模块直接插入 SparseDrive 的感知到规划链路中，使其与检测、建图、运动预测和规划共同训练。这使得补全目标从“重建像素/特征”转向“维持驾驶性能”。

### 创新点 2：基于几何对齐的运动补偿时序补全

相比简单地用历史特征平均或直接拷贝当前其他相机特征，当前项目显式使用：

- 历史特征队列；
- `T_global` 构造的时序位姿变换；
- 多深度假设的图像级 motion warp；
- 跨相机时序注意力；
- 空间细化与门控融合。

这样做的出发点是：缺失相机当前视野的信息，往往能够在历史时刻或相邻视角中找到，但必须先解决几何错位问题。

### 创新点 3：让规划任务反向指导补全

传统补全模块主要优化重建误差，而本项目进一步引入规划相关信号：

- 根据 ego 状态动态给不同相机分配权重；
- 使用预测轨迹或 GT 轨迹构建重要性图；
- 对轨迹关键区域采用更精细的补全网络；
- 保持补全特征与下游规划 Head 的梯度连通。

这意味着补全模块学习的不只是“像不像完整特征”，而是“能不能帮助模型做出更稳的规划”。

### 创新点 4：把鲁棒性做成一条完整训练管线

当前系统不是一个单点模块，而是一条从故障模拟、特征恢复、辅助监督、任务反馈到测试时缺失评测的完整研究原型，已经具备进一步做系统消融和论文实验的条件。

---

## 4. 如何做的

### 4.1 基础模型

项目继承了 SparseDrive 的主干结构：

- **图像编码**：`ResNet-50 + FPN`
- **稀疏感知头**：`SparseDriveHead`
- **检测分支**：`det_head`
- **在线建图分支**：`map_head`
- **运动预测与规划分支**：`motion_plan_head`

原始 SparseDrive 的核心思想是以稀疏实例表示统一多任务；当前项目保留这一主框架，只在 FPN 输出与 Head 输入之间增加鲁棒性模块。

### 4.2 真实执行的训练流程

按照当前代码实现，训练时的主流程是：

```text
完整图像 img
  ├─ full branch: 提取完整特征，作为 teacher
  └─ masked branch: 先做 RandCamMask，再提取 student 特征
           ↓
      SSL 特征一致性监督
           ↓
      轻量世界模型监督（可选）
           ↓
      运动补偿时序补全
           ↓
      规划导向加权
           ↓
      VAE 视角补全
           ↓
      规划引导精细补全
           ↓
      SparseDriveHead
           ↓
      检测 / 建图 / 运动预测 / 规划联合损失
```

这里需要强调一个关键点：**当前代码中的真实顺序是“时序补全 → VAE 补全 → 规划引导细化”**，而不是部分旧文档中描述的其他顺序。

### 4.3 损失设计

当前训练目标可以概括为：

```math
\mathcal{L} =
\mathcal{L}_{task}
 + \lambda_{ssl}\mathcal{L}_{ssl}
 + \lambda_{vae}\mathcal{L}_{vae}
 + \lambda_{world}\mathcal{L}_{world}
 + \lambda_{comp}\mathcal{L}_{completion}
```

其中：

- `L_task`：SparseDrive 原有的检测、建图、运动预测、规划等任务损失；
- `L_ssl`：full/masked 分支在缺失视角上的特征一致性损失；
- `L_vae`：VAE 重建损失与 KL 散度；
- `L_world`：轻量 Dreamer/RSSM 风格的世界模型监督；
- `L_completion`：规划引导补全的重建损失。

由于补全后的特征不做 `detach`，规划损失会沿着计算图自然反传到补全模块，形成真正的端到端反馈。

---

## 5. 模型结构是什么

从结构上看，当前项目可以分为“基线骨架”和“鲁棒性增强层”两部分。

### 5.1 基线骨架

```text
Multi-view Images
    ↓
ResNet-50
    ↓
FPN Multi-scale Features
    ↓
SparseDriveHead
    ├─ det_head
    ├─ map_head
    └─ motion_plan_head
```

### 5.2 新增鲁棒性增强层

```text
Masked Multi-view Features
    ↓
MotionCompensatedTemporalCompletion
    ├─ FeatureQueue
    ├─ ImageLevelMotionWarp
    ├─ TemporalCrossAttention
    └─ Gated Fusion
    ↓
PlanningGuidedWeighting
    ↓
PVReconVAE
    ↓
PlanningGuidedCompletion
    ├─ CrossCameraAttention
    ├─ Conditional Completion
    ├─ Trajectory Importance
    ├─ Coarse / Fine Completion
    └─ Gated Fusion
    ↓
SparseDriveHead
```

### 5.3 各模块作用

| 模块 | 作用 |
| --- | --- |
| `RandCamMask` | 在训练和可选测试阶段模拟真实相机缺失 |
| `MotionCompensatedTemporalCompletion` | 借助历史帧和几何对齐恢复缺失视角的粗语义特征 |
| `PlanningGuidedWeighting` | 根据相机位置和 ego 状态，调整不同视角的重要性 |
| `PVReconVAE` | 提供多尺度视角级重建能力，作为零历史时的补全兜底 |
| `PlanningGuidedCompletion` | 根据轨迹关键区域做进一步细化补全 |
| `LightDreamerRSSM` | 用潜在时序建模约束缺失视角的特征分布 |

---

## 6. 当前项目的阶段性实验形态

从配置文件可以看出，当前项目已经进入“围绕规划指标继续微调”的阶段，而不是只停留在基础模块接入：

- `sparsedrive_small_stage2.py` 作为当前主配置，已经默认接入相机缺失模拟、时序补全和规划引导补全；
- 默认从 `exp20` checkpoint 继续训练，并冻结了大量感知与补全模块，主要微调 `motion_plan_head`；
- `exp23 / exp24 / exp25` 继续对运动/规划损失权重、重排序策略和学习率做小步搜索，目标更偏向提升规划 L2。

这说明当前项目的研究重点已经从“模块能否接上”转向“在现有补全框架下，如何把规划性能调到更好”。

需要额外说明的是：

- 仓库中已经存在 `temporal_completion_v2.py` 和 `sparsedrive_small_stage2_v2.py`；
- 但当前主模型构造函数默认仍接入 V1 的 `MotionCompensatedTemporalCompletion`；
- 因此，**V2 更适合被描述为已实现的候选优化版本，而不是默认主线结果**。

---

## 7. 现阶段效果

### 7.1 原始 SparseDrive 发布结果

根据仓库 `README.md` 中给出的 released checkpoint 结果，原始 Stage2 SparseDrive 在 nuScenes 上达到：

| 指标 | 数值 |
| --- | --- |
| Detection NDS | 0.5257 |
| Mapping mAP | 0.5656 |
| Tracking AMOTA | 0.372 |
| Motion EPA (car) | 0.492 |
| Planning Collision Rate | 0.097% |
| Planning L2 | 0.61 |

这组结果可以视为当前项目需要追平或超越的**基线天花板**。

### 7.2 当前项目可追溯的本地实验结果

截至 **2025 年 12 月 11 日**，仓库内最完整、可追溯的一份本地验证日志是 `20251208_162017.log`。这份实验已经能够完成训练和端到端评测，得到如下结果：

| 指标 | 数值 |
| --- | --- |
| Detection mAP | 0.2794 |
| Detection NDS | 0.4105 |
| Tracking AMOTA | 0.1865 |
| Mapping mAP_normal | 0.2532 |
| Motion EPA (car / pedestrian) | 0.3875 / 0.3288 |
| Planning L2 | 0.7163 |
| Planning obj_box_col | 0.115% |

需要说明的是，这组结果来自本地鲁棒性实验配置，和 `README.md` 中的 released checkpoint 并非完全同一训练/评测设定，因此更适合作为**当前阶段性参考**，而不是严格的一一对标结论。

### 7.3 对当前效果的客观判断

从现有结果看，可以得出三个结论：

1. **系统已经跑通**  
   相机缺失模拟、补全链路、联合训练与端到端评测都已经接入完成，不再是纯设计阶段。

2. **性能尚未达到原始 SparseDrive 基线**  
   当前可追溯实验结果仍明显低于发布版 SparseDrive，说明鲁棒性增强模块虽然已经成型，但超参数和训练策略还未收敛到理想状态。

3. **训练稳定性仍是主要瓶颈**  
   从日志可以看到，`loss_pv_vae_rec`、`loss_pv_ssl` 和 `loss_world_rec` 在训练后期偏大，且出现了 `grad_norm: nan`。这说明辅助补全损失与主任务损失之间的平衡仍需继续调参。

因此，更准确的表述不是“当前项目已经显著提升了最终指标”，而是：

> 当前项目已经完成了面向相机失效鲁棒性的核心方法实现，并验证了其训练和评测可行性；但在最终性能上仍处于持续调参与稳定化阶段。

---

## 8. 一句话总结

如果用论文摘要式的一句话概括当前项目，那么它做的事情是：

> **在 SparseDrive 的统一稀疏端到端自动驾驶框架上，引入面向相机失效场景的时序补全与规划引导补全机制，试图在传感器不完整输入条件下维持感知到规划的整体性能。**
