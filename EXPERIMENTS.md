# SparseDrive 实验记录

完整的 `work_dirs` 实验汇总见 [WORKDIR_EXPERIMENTS.md](/home/wudi/code/mySparseDrive/SparseDrive/WORKDIR_EXPERIMENTS.md)。

## 2026-03 规划/轨迹分支冻结微调记录

### 实验目标

基于 `exp20` 的权重继续微调，仅更新 `motion_plan_head`，冻结感知和建图相关模块，验证是否可以在保持整体指标基本稳定的前提下进一步降低规划 `L2` 误差。

### 实验配置

- 配置文件：`projects/configs/sparsedrive_small_stage2.py`
- 冻结策略：冻结 `img_backbone`、`img_neck`、`depth_branch`、`pv_recon`、`world_model`、`temporal_completion`、`planning_weighting`、`planning_guided_completion`、`planning_feedback_loss`、`head.det_head`、`head.map_head`
- 训练分支：`motion_plan_head`
- 注意：当前 `motion_plan_head` 仍同时优化 `motion loss` 和 `planning loss`，并不是纯 `planning-only` 微调

### 实验目录

- `exp20` 基线：`work_dirs/sparsedrive_small_stage2_exp20/`
- `exp22-0`：误加载权重的无效实验，目录为 `work_dirs/sparsedrive_small_stage2_exp22-0/`
- `exp22`：修正后从 `exp20` 最终权重启动的有效实验，目录为 `work_dirs/sparsedrive_small_stage2_exp22/`
- `exp23`：从 `exp22/latest.pth` 继续做更强的 L2-focused 微调
- `exp24`：从 `exp23/iter_210975.pth` 继续做稳定性修正
- `exp25`：这次实验，从 `exp22/iter_140650.pth` 重启，走更稳的低学习率方案

### 关键结果对比

| 指标 | exp20 基线 | exp22-0（误加载） | exp22（修正后，iter_140650） |
|------|------------|-------------------|------------------------------|
| mAP | 0.4131 | 0.4145 | 0.4126 |
| NDS | 0.5258 | 0.5246 | 0.5269 |
| AMOTA | 0.3792 | 0.3699 | 0.3796 |
| car EPA | 0.4977 | 0.4856 | 0.4961 |
| pedestrian EPA | 0.4109 | 0.4115 | 0.4149 |
| car minADE | 0.6359 | 0.6197 | 0.6435 |
| car minFDE | 1.0053 | 0.9785 | 1.0283 |
| pedestrian minADE | 0.7195 | 0.7031 | 0.7070 |
| pedestrian minFDE | 1.0558 | 1.0303 | 1.0328 |
| obj_box_col | 0.103% | 0.246% | 0.112% |
| L2 | 0.6429 | 0.7900 | 0.6311 |

### 结果分析

- `exp22-0` 不能用于评估“基于 `exp20` 微调是否有效”，因为它实际从 `ckpt/sparsedrive_stage2.pth` 启动，而不是从 `exp20` 启动
- `exp22-0` 的主要问题是规划明显退化，`L2` 从 `0.6429` 升到 `0.7900`，同时 `obj_box_col` 也显著升高
- `exp22` 修正为从 `work_dirs/sparsedrive_small_stage2_exp20/iter_281300.pth` 启动后，`L2` 降到 `0.6311`，优于 `exp20`
- 修正后检测和跟踪基本稳定，`NDS` 和 `AMOTA` 没有明显恶化
- 修正后 motion 指标有小幅波动，说明当前微调更偏向改善 planning，不一定会同时优化所有 motion 指标

### 后续微调结果表格

下表把 `exp22` 之后的三次微调一起整理出来，便于横向看“这次实验”在整个冻结微调序列中的位置。

| 实验 | 初始化权重 | 主要设置 | 最新 checkpoint | mAP | NDS | AMOTA | car EPA | ped EPA | obj_box_col | L2 | 备注 |
|------|------------|----------|-----------------|-----|-----|-------|---------|---------|-------------|----|------|
| exp22 | `exp20/iter_281300.pth` | 冻结大部分模块，仅训 `motion_plan_head`，`lr=1.5e-5` | `iter_140650` | 0.4126 | 0.5269 | 0.3796 | 0.4961 | 0.4149 | 0.112% | 0.6311 | 当前最优 L2 和 NDS |
| exp23 | `exp22/latest.pth` | `lr=5e-6`，`motion loss=0`，`plan cls/status=0` | `iter_281300` | 0.4114 | 0.5249 | 0.3772 | 0.4944 | 0.4156 | 0.110% | 0.6657 | 过度压缩 loss，L2 回升 |
| exp24 | `exp23/iter_210975.pth` | `lr=5e-6`，`motion loss=0`，`plan cls=0.2`，`status=0` | `iter_140650` | 0.4121 | 0.5262 | 0.3772 | 0.4946 | 0.4155 | 0.209% | 0.7116 | 规划和碰撞同时恶化 |
| exp25 | `exp22/iter_140650.pth` | `lr=3e-6`，`motion loss=0.1`，`plan status=0.2` | `iter_281300` | 0.4133 | 0.5264 | 0.3808 | 0.4986 | 0.4158 | 0.104% | 0.6464 | 这次实验，次优平衡点 |

### 后续微调分析

- `exp23` 说明把 `motion loss` 和 `plan cls/status` 一次性压到接近 0 会让训练目标过窄，`L2` 反而从 `0.6311` 回升到 `0.6657`
- `exp24` 虽然尝试恢复部分 `plan cls`，但 `obj_box_col` 升到 `0.209%`，说明稳定性没有真正修回来
- `exp25` 改为从 `exp22` 中期 checkpoint 重启，并把学习率降到 `3e-6`，结果把 `AMOTA` 拉回到 `0.3808`，`obj_box_col` 压到 `0.104%`
- 就这次实验本身而言，`exp25` 比 `exp23/24` 明显更稳，但 `L2 = 0.6464` 仍未超过 `exp22` 的 `0.6311`
- 如果目标是追求最优 `L2`，当前仍应以 `exp22` 为主；如果目标是兼顾 tracking 和碰撞率，`exp25` 是目前更均衡的后续版本

### 排查结论

- 这次性能判断偏差的根因是 `load_from` 配置错误
- 错误实验的结论应视为无效，只能作为“错误初始化会导致规划退化”的反例
- 当前有效配置：`work_dir = "./work_dirs/sparsedrive_small_stage2_exp22"`
- 当前有效配置：`load_from = "work_dirs/sparsedrive_small_stage2_exp20/iter_281300.pth"`

### 后续建议

- 优先将 `exp22` 继续训练到最终 checkpoint，再与 `exp20` 做完整对比
- 如果更关注整体均衡性，可以从 `exp25` 继续小步微调，而不是从 `exp24` 延续
- 如果目标是进一步压低 `L2`，建议补一个真正的 `planning-only` 消融实验，将 `motion_loss_cls` 和 `motion_loss_reg` 置零
- 后续实验记录统一写入本文件，避免只看 `work_dir` 名字导致混淆

## 2026-03 exp26: flow 版 PVRecon 实验

### 实验目标

验证把 `pv_recon` 从原有 VAE 路线切换为 `flow` 路线后，整体 e2e 指标是否还能保持稳定，以及它对 tracking、motion 和 planning 的影响。

### 实验配置

- 配置文件：`projects/configs/sparsedrive_small_stage2_exp26.py`
- 初始化权重：`ckpt/sparsedrive_stage2.pth`
- 关键改动：`pv_recon_type='flow'`
- 轨迹来源：`trajectory_source='pred'`
- 学习率：默认主线设置，未采用 `exp22-25` 的冻结微调方案
- 这一实验不属于 `exp22-25` 的“冻结 `motion_plan_head` 微调线”，而是一条独立的结构改动实验

### 全部评测结果

`exp26` 一共做了 4 次完整评测，分别对应：

- `iter_70325`
- `iter_140650`
- `iter_210975`
- `iter_281300`

结果来源：

- 中间 3 次评测来自 `work_dirs/sparsedrive_small_stage2_exp26/20260313_205923.log`
- 最终一次评测同时也写入了 `work_dirs/sparsedrive_small_stage2_exp26/e2e_metrics.json`

#### checkpoint 级结果总表

| checkpoint | mAP | NDS | AMOTA | mAP_normal | car EPA | ped EPA | obj_box_col | L2 | 备注 |
|------------|-----|-----|-------|------------|---------|---------|-------------|----|------|
| iter_70325 | 0.4007 | 0.5154 | 0.3674 | 0.5400 | 0.4815 | 0.4031 | 0.166% | **0.6329** | 第一次评测，planning 最好 |
| iter_140650 | 0.4050 | 0.5177 | 0.3666 | 0.5476 | 0.4903 | 0.4012 | 0.084% | 0.6655 | 碰撞更低，但 L2 回升 |
| iter_210975 | 0.4077 | **0.5262** | 0.3767 | 0.5442 | 0.4953 | 0.4005 | **0.075%** | 0.6467 | 最均衡 checkpoint |
| iter_281300 | **0.4123** | 0.5258 | **0.3810** | **0.5511** | **0.5009** | **0.4114** | 0.155% | 0.6603 | 最终 checkpoint，tracking/motion 最强 |

#### 详细指标表

| checkpoint | car ADE / FDE / MR | pedestrian ADE / FDE / MR | AMOTP / Recall / MOTA / MOTP |
|------------|---------------------|----------------------------|-------------------------------|
| iter_70325 | 0.6447 / 1.0073 / 0.1414 | 0.7302 / 1.0663 / 0.1492 | 1.2655 / 0.4679 / 0.3397 / 0.6295 |
| iter_140650 | 0.6360 / 0.9954 / 0.1377 | 0.7594 / 1.1198 / 0.1591 | 1.2503 / 0.5343 / 0.3331 / 0.6656 |
| iter_210975 | 0.6323 / 1.0008 / 0.1283 | 0.7405 / 1.0835 / 0.1525 | 1.2455 / 0.4831 / 0.3448 / 0.6293 |
| iter_281300 | 0.6235 / 0.9929 / 0.1287 | 0.7258 / 1.0655 / 0.1449 | 1.2493 / 0.4842 / 0.3461 / 0.6212 |

### 结果分析

- `exp26` 的 4 次评测不是单调变好，而是出现了明显的任务 trade-off
- 如果只看 planning，第一次 `iter_70325` 最好，`L2 = 0.6329`，已经非常接近 `exp22 = 0.6311`
- 如果看碰撞安全性，`iter_210975` 最好，`obj_box_col = 0.075%`
- 如果看 detection / tracking / motion，最终 `iter_281300` 最强，`mAP = 0.4123`、`AMOTA = 0.3810`、`car EPA = 0.5009`
- 也就是说，`exp26` 越往后训练，感知、跟踪和运动预测在持续提升，但 planning `L2` 没有同步受益，反而从 `0.6329` 回升到 `0.6603`
- 这说明 `flow` 版 `pv_recon` 对感知和时序建模是有效的，但当前训练目标还没有把这种收益稳定传递到规划头

### 当前结论

- 如果目标是追求 **最佳 planning L2**，`exp26` 应该优先取 `iter_70325`，而不是最终 `iter_281300`
- 如果目标是追求 **最均衡 checkpoint**，`iter_210975` 更合适：`NDS` 高、`obj_box_col` 最低、`L2` 也优于最终 checkpoint
- 如果目标是追求 **最强 tracking / motion**，最终 `iter_281300` 仍然最合适
- 更合理的下一步不是只保留一个“最终结果”，而是同时记住：
  - `iter_70325`：最佳 planning
  - `iter_210975`：最佳均衡
  - `iter_281300`：最佳 tracking / motion
- 后续如果继续做 `exp26` 族实验，更建议在 `iter_70325` 或 `iter_210975` 基础上接规划微调，而不是默认从最终 checkpoint 开始

## 2026-03 exp27: 基于 exp26 最佳 planning checkpoint 的低学习率微调

### 实验目标

验证从 `exp26` 的最佳 planning checkpoint `iter_70325` 出发，冻结感知、检测、建图和 `flow pv_recon` 主干，仅保留 planning 相关模块继续低学习率微调，是否能把 `flow` 路线的 planning 优势稳定保留下来。

### 实验配置

- 配置文件：`projects/configs/sparsedrive_small_stage2_exp27.py`
- 初始化权重：`work_dirs/sparsedrive_small_stage2_exp26/iter_70325.pth`
- 继承结构：沿用 `exp26` 的 `pv_recon_type='flow'`
- 冻结模块：`img_backbone`、`img_neck`、`depth_branch`、`pv_recon`、`world_model`、`temporal_completion`、`head.det_head`、`head.map_head`
- 学习率：`lr=3e-6`
- 训练长度：`70325 iter`
- 评测频率：每 `14065 iter` 做一次完整评测

### 全部评测结果

`exp27` 一共做了 5 次完整评测，分别对应：

- `iter_14065`
- `iter_28130`
- `iter_42195`
- `iter_56260`
- `iter_70325`

结果来源：

- 前 4 次和最终一次评测都记录在 `work_dirs/sparsedrive_small_stage2_exp27/20260317_221159.log`
- 最终一次评测同时也写入了 `work_dirs/sparsedrive_small_stage2_exp27/e2e_metrics.json`

#### checkpoint 级结果总表

| checkpoint | mAP | NDS | AMOTA | mAP_normal | car EPA | ped EPA | obj_box_col | L2 | 备注 |
|------------|-----|-----|-------|------------|---------|---------|-------------|----|------|
| iter_14065 | 0.4003 | 0.5144 | 0.3659 | 0.5402 | 0.4878 | 0.3999 | 0.120% | 0.6268 | 第一次评测，已优于 exp26 起点 |
| iter_28130 | 0.4002 | 0.5147 | 0.3638 | 0.5402 | 0.4860 | 0.4024 | **0.105%** | 0.6306 | 碰撞最低 |
| iter_42195 | 0.4004 | 0.5137 | **0.3671** | 0.5402 | 0.4881 | 0.4023 | 0.118% | **0.6222** | planning 最好，也是 tracking 最强 |
| iter_56260 | 0.4002 | 0.5148 | 0.3650 | 0.5402 | **0.4882** | 0.4035 | 0.123% | 0.6420 | 中后期回退明显 |
| iter_70325 | **0.4015** | **0.5157** | 0.3656 | 0.5401 | 0.4869 | **0.4042** | 0.113% | 0.6268 | 最终 checkpoint，family 内 mAP/NDS 最高 |

#### 详细指标表

| checkpoint | car ADE / FDE / MR | pedestrian ADE / FDE / MR | AMOTP / Recall / MOTA / MOTP |
|------------|---------------------|----------------------------|-------------------------------|
| iter_14065 | 0.6331 / 0.9819 / 0.1324 | 0.7349 / 1.0776 / 0.1523 | 1.2617 / 0.5080 / 0.3302 / 0.6512 |
| iter_28130 | 0.6297 / 0.9771 / 0.1327 | 0.7229 / 1.0580 / 0.1459 | 1.2718 / 0.4587 / 0.3316 / 0.6299 |
| iter_42195 | 0.6296 / 0.9770 / 0.1317 | 0.7264 / 1.0627 / 0.1458 | 1.2622 / 0.4991 / 0.3314 / 0.6555 |
| iter_56260 | 0.6320 / 0.9816 / 0.1308 | 0.7284 / 1.0667 / 0.1472 | 1.2670 / 0.4811 / 0.3338 / 0.6335 |
| iter_70325 | 0.6319 / 0.9821 / 0.1320 | 0.7253 / 1.0588 / 0.1449 | 1.2651 / 0.4793 / 0.3341 / 0.6215 |

### 结果分析

- `exp27` 证明“`exp26` 中期 checkpoint + 低学习率 planning 微调”这条路线是有效的
- 与初始化点 `exp26/iter_70325 (L2 = 0.6329)` 相比，`exp27` 的最佳 checkpoint `iter_42195` 把 `L2` 进一步压到 `0.6222`
- 即使只看最终 checkpoint，`exp27/iter_70325` 的 `L2 = 0.6268` 也已经优于当前 `exp22 = 0.6311`
- `exp27` 的 detection / tracking 基本维持在 `mAP ≈ 0.400`、`NDS ≈ 0.514-0.516` 的窄区间，没有出现明显崩坏，但整体上仍弱于 `exp22/25` 的感知主线
- 这说明从 `exp26` 早期 planning 较好的点继续做低学习率微调，确实能把 `flow` 路线的 planning 优势进一步兑现出来
- 但与此同时，`exp27` 并没有把 `flow` 路线的 detection / NDS 一起拉高，说明当前这条微调更偏向“保留已有感知 + 修 planning”，而不是整体指标同步优化

### 当前结论

- 如果只看 **planning L2**，`exp27` 已经是当前最强的一条有效实验线
- `exp27` 家族内最佳 planning checkpoint 是 `iter_42195`，`L2 = 0.6222`
- `exp27` 家族内最安全 checkpoint 是 `iter_28130`，`obj_box_col = 0.105%`
- `exp27` 家族内最终 checkpoint `iter_70325` 更像“默认可复现结果”：`L2 = 0.6268`、`NDS = 0.5157`
- 这也说明 `exp26 -> exp27` 比直接沿 `exp26` 训到最终点更合理；真正值得保留的不是 `exp26` 最终 checkpoint，而是它的早期 planning 点和后续微调结果

### 补充：exp27 latest 的 masked eval

为了验证 `exp27` 在测试时真正遇到相机缺失时的鲁棒性，又补做了一次 masked eval。

结果来源：

- `exp27_mask_eval/e2e_metrics.json`
- 对比基线仍使用 `work_dirs/sparsedrive_small_stage2_exp27/e2e_metrics.json`

#### 标准 eval vs masked eval

| 评测口径 | mAP | NDS | AMOTA | mAP_normal | car EPA | ped EPA | obj_box_col | L2 |
|----------|-----|-----|-------|------------|---------|---------|-------------|----|
| 标准 eval（无 mask） | 0.4015 | 0.5157 | 0.3656 | 0.5401 | 0.4869 | 0.4042 | 0.113% | **0.6268** |
| masked eval（有 mask） | 0.3896 | 0.5076 | 0.3517 | 0.5128 | 0.4777 | 0.3919 | 0.125% | 0.6344 |

#### masked eval 详细指标

| 评测口径 | car ADE / FDE / MR | pedestrian ADE / FDE / MR | AMOTP / Recall / MOTA / MOTP |
|----------|---------------------|----------------------------|-------------------------------|
| 标准 eval（无 mask） | 0.6319 / 0.9821 / 0.1320 | 0.7253 / 1.0588 / 0.1449 | 1.2651 / 0.4793 / 0.3341 / 0.6215 |
| masked eval（有 mask） | 0.6448 / 0.9964 / 0.1337 | 0.7328 / 1.0657 / 0.1486 | 1.2742 / 0.5009 / 0.3167 / 0.6589 |

#### masked eval 分析

- 启用测试时 mask 后，`exp27` 的各项指标出现了预期中的退化，没有出现“开 mask 反而更高”的异常情况
- 相比标准 eval，`mAP` 下降 `0.0119`，`NDS` 下降 `0.0081`，`AMOTA` 下降 `0.0139`
- planning 方面，`L2` 从 `0.6268` 回升到 `0.6344`，`obj_box_col` 从 `0.113%` 升到 `0.125%`
- 这个掉幅不算离谱，说明 `exp27` 对相机缺失有一定鲁棒性，但它的优化目标仍然主要是标准评测下的 planning 微调，不是专门为 masked eval 做的鲁棒性优化
- 结合当前配置看，这个结果也符合预期：`exp27` 冻结了 `pv_recon` 和 `temporal_completion`，所以后半程并没有继续针对“缺失相机补全能力”做强化

## V2 版本实验（局部注意力优化）

### 实验目标
验证局部注意力优化（V2）的有效性，对比 V1 vs V2 的性能、显存、速度。

### 实验配置

#### V1 基线版本
- **配置文件**：`projects/configs/sparsedrive_small_stage2.py`
- **模块**：`MotionCompensatedTemporalCompletion`
- **特点**：全局跨相机注意力（6个相机）

#### V2 优化版本
- **配置文件**：`projects/configs/sparsedrive_small_stage2_v2.py`
- **模块**：`MotionCompensatedTemporalCompletionV2`
- **特点**：局部跨相机注意力（3个相邻相机，基于拓扑）

### 实验计划

#### 阶段 1: 单卡验证（快速验证）
```bash
# 训练命令
bash ./tools/dist_train.sh projects/configs/sparsedrive_small_stage2_v2.py 1 --deterministic

# 预期训练时间：约 2-3 天（单卡 A100）
# 监控指标：显存占用、训练速度（iter/s）、损失曲线
```

**关注指标**：
- [ ] 显存占用（预期：V2 比 V1 减少 ~30-50%）
- [ ] 训练速度（预期：V2 比 V1 快 ~20-40%）
- [ ] 损失收敛情况

#### 阶段 2: 多卡完整训练
```bash
# V1 基线
bash ./tools/dist_train.sh projects/configs/sparsedrive_small_stage2.py 8 --deterministic

# V2 优化
bash ./tools/dist_train.sh projects/configs/sparsedrive_small_stage2_v2.py 8 --deterministic
```

**关注指标**：
- [ ] Detection mAP（预期：V2 ≈ V1，差异 < 1%）
- [ ] Tracking AMOTA（预期：V2 ≈ V1）
- [ ] Mapping AP（预期：V2 ≈ V1）
- [ ] Motion/Planning 指标（预期：V2 ≈ V1）

#### 阶段 3: 消融实验
- [ ] 关闭时序补全：验证时序补全的贡献
- [ ] V1 vs V2：直接对比全局 vs 局部注意力
- [ ] 不同相机拓扑：测试拓扑设计的合理性

### 实验结果记录

#### 实验 1: [日期] - 单卡快速验证

**硬件配置**：
- GPU:
- 显存:
- Batch size:

**V1 基线**：
- 显存占用:
- 训练速度:
- 损失曲线:

**V2 优化**：
- 显存占用:
- 训练速度:
- 损失曲线:

**初步结论**：


---

#### 实验 2: [日期] - 完整训练评估

**训练配置**：
- Epochs:
- Learning rate:
- 其他超参数:

**V1 基线结果**：
```
Detection:
  - mAP:
  - NDS:
Tracking:
  - AMOTA:
  - AMOTP:
Mapping:
  - AP:
Motion:
  - minADE:
  - minFDE:
Planning:
  - L2 error:
  - Collision rate:
```

**V2 优化结果**：
```
Detection:
  - mAP:
  - NDS:
Tracking:
  - AMOTA:
  - AMOTP:
Mapping:
  - AP:
Motion:
  - minADE:
  - minFDE:
Planning:
  - L2 error:
  - Collision rate:
```

**性能对比**：
- mAP 差异:
- 显存减少:
- 速度提升:

**结论**：


---

### 注意事项

1. **Checkpoint 管理**：
   - V1 checkpoints 保存在：`work_dirs/sparsedrive_small_stage2/`
   - V2 checkpoints 保存在：`work_dirs/sparsedrive_v2_local_attention/`

2. **日志查看**：
   ```bash
   # Tensorboard
   tensorboard --logdir work_dirs/

   # 对比两个版本
   tensorboard --logdir_spec v1:work_dirs/sparsedrive_small_stage2,v2:work_dirs/sparsedrive_v2_local_attention
   ```

3. **显存监控**：
   ```bash
   # 实时监控显存
   watch -n 1 nvidia-smi
   ```

4. **评估命令**：
   ```bash
   # V1
   bash ./tools/dist_test.sh projects/configs/sparsedrive_small_stage2.py \
       work_dirs/sparsedrive_small_stage2/latest.pth 8 --eval bbox

   # V2
   bash ./tools/dist_test.sh projects/configs/sparsedrive_small_stage2_v2.py \
       work_dirs/sparsedrive_v2_local_attention/latest.pth 8 --eval bbox
   ```

---

## 其他实验

（后续其他改进的实验记录可以添加在这里）

### 深度引导运动补偿（计划中）

### 多路并行补全（计划中）

### 时序一致性约束（计划中）
