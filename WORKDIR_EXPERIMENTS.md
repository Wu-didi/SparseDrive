# Work_dirs 实验总览

更新时间：2026-03-20

数据来源：

- `work_dirs/*/metrics_summary.json`
- `work_dirs/*/e2e_metrics.json`
- `work_dirs/*/*.log`
- `work_dirs/*/*.py` 配置快照
- `./exp27_mask_eval/e2e_metrics.json`（补充记录 `exp27` 的 masked eval）

整理原则：

- 有 `e2e_metrics.json` 的实验归入“完整 e2e 实验”
- 只有 `metrics_summary.json` 的实验归入“历史 tracking 实验”
- 没有完整评估文件的目录不纳入横向比较

## 论文主表

说明：

- `train commit` 来自对应实验目录下最新 `*.log.json` 的环境记录；早期实验日志未保留 commit 信息时记为 `unknown`
- `branch` 表示当前仓库中仍保留、与该实验最接近的分支名；精确训练代码以 `train commit` 为准
- `Tracking` 列格式为 `AMOTA / AMOTP / Recall / MOTA / MOTP`
- `Motion` 列格式为 `car EPA / ADE / FDE / MR ; ped EPA / ADE / FDE / MR`
- `Planning` 列格式为 `obj_col / obj_box_col / L2`

| 实验 | 类型 | 配置快照 | train commit | branch | 初始化权重 | 改动内容 | Detection | Tracking | Map | Motion | Planning | 备注 |
|------|------|----------|--------------|--------|------------|----------|-----------|----------|-----|--------|----------|------|
| stage1_before | stage1 | `sparsedrive_small_stage1.py` | `unknown` | `未保留` | `unknown` | 早期 stage1 感知预训练结果 | `-` | `0.0014 / 1.7761 / 0.4125 / 0.0070 / 0.8879` | `-` | `-` | `-` | 与 stage2 e2e 不可直接比较 |
| exp8 | tracking-only | `sparsedrive_small_stage2.py` | `unknown` | `未保留` | `unknown` | 早期 stage2 基线 | `-` | `0.3866 / 1.2526 / 0.4831 / 0.3539 / 0.6268` | `-` | `-` | `-` | 早期较强 tracking 基线 |
| exp9 | tracking-only | `sparsedrive_small_stage2.py` | `unknown` | `未保留` | `unknown` | 早期 stage2 变体 | `-` | `0.3289 / 1.2932 / 0.4537 / 0.3114 / 0.6733` | `-` | `-` | `-` | 相比 exp8 退化 |
| exp13 | tracking-only | `sparsedrive_small_stage2.py` | `unknown` | `未保留` | `ckpt/sparsedrive_stage2.pth` | 高学习率尝试，`lr=3.5e-5` | `-` | `0.1865 / 1.5158 / 0.2837 / 0.1933 / 0.8917` | `-` | `-` | `-` | 失败实验 |
| exp14 | tracking-only | `sparsedrive_small_stage2.py` | `unknown` | `未保留` | `ckpt/sparsedrive_stage2.pth` | 高学习率重试，`lr=3.5e-5` | `-` | `0.3749 / 1.2624 / 0.5087 / 0.3368 / 0.6538` | `-` | `-` | `-` | 恢复正常 |
| exp15 | tracking-only | `sparsedrive_small_stage2.py` | `unknown` | `未保留` | `ckpt/sparsedrive_stage2.pth` | exp14 同族稳定复跑，`lr=3.5e-5` | `-` | `0.3702 / 1.2695 / 0.4812 / 0.3406 / 0.6536` | `-` | `-` | `-` | 与 exp14 接近 |
| exp16 | tracking-only | `sparsedrive_small_stage2.py` | `unknown` | `未保留` | `exp15/iter_210960.pth` | 在 exp15 上继续训练，`lr=3.5e-5` | `-` | `0.3730 / 1.2587 / 0.4913 / 0.3418 / 0.6408` | `-` | `-` | `-` | 小幅提升 |
| exp17 | tracking-only | `sparsedrive_small_stage2.py` | `unknown` | `未保留` | `ckpt/sparsedrive_stage2.pth` | 学习率降到 `1.5e-5` | `-` | `0.3715 / 1.2616 / 0.4850 / 0.3370 / 0.6373` | `-` | `-` | `-` | 进入稳定区间 |
| exp18 | tracking-only | `sparsedrive_small_stage2.py` | `unknown` | `未保留` | `ckpt/sparsedrive_stage2.pth` | 启用规划引导补全相关配置 | `-` | `0.3696 / 1.2691 / 0.4915 / 0.3380 / 0.6517` | `-` | `-` | `-` | exp19 前置版本 |
| exp19 | e2e | `sparsedrive_small_stage2.py` | `a2ed2cb` | `exp19` | `ckpt/sparsedrive_stage2.pth` | `trajectory_source='none'` | `0.4137 / 0.5244` | `0.3905 / 1.2494 / 0.4907 / 0.3590 / 0.6287` | `0.5496` | `0.5018 / 0.6265 / 0.9889 / 0.1278 ; 0.4111 / 0.7198 / 1.0557 / 0.1473` | `0.6702% / 0.1769% / 0.6448` | tracking 最强 |
| exp20 | e2e | `sparsedrive_small_stage2.py` | `2a5dbeb` | `exp20` | `ckpt/sparsedrive_stage2.pth` | `trajectory_source='pred'` | `0.4131 / 0.5258` | `0.3792 / 1.2490 / 0.5319 / 0.3479 / 0.6475` | `0.5515` | `0.4977 / 0.6359 / 1.0053 / 0.1305 ; 0.4109 / 0.7195 / 1.0558 / 0.1487` | `0.6702% / 0.1031% / 0.6429` | 规划基线 |
| exp21 | e2e | `sparsedrive_small_stage2.py` | `049f490` | `exp21` | `ckpt/sparsedrive_stage2.pth` | `trajectory_source='gt'` | `0.4095 / 0.5229` | `0.3710 / 1.2487 / 0.5262 / 0.3440 / 0.6473` | `0.5483` | `0.4917 / 0.6427 / 0.9919 / 0.1322 ; 0.4005 / 0.7521 / 1.0994 / 0.1538` | `0.6702% / 0.0922% / 0.7314` | GT 引导导致规划退化 |
| exp22-0 | e2e | `sparsedrive_small_stage2.py` | `003bd6a` | `exp22` 家族 | `ckpt/sparsedrive_stage2.pth` | 冻结大部分模块，仅训 `motion_plan_head`，但误加载初始权重 | `0.4145 / 0.5246` | `0.3699 / 1.2543 / 0.4999 / 0.3436 / 0.6250` | `0.5656` | `0.4856 / 0.6197 / 0.9785 / 0.1400 ; 0.4115 / 0.7031 / 1.0303 / 0.1393` | `0.6702% / 0.2458% / 0.7900` | 无效对照 |
| exp22 | e2e | `sparsedrive_small_stage2.py` | `003bd6a` | `exp22` | `exp20/iter_281300.pth` | 冻结大部分模块，仅训 `motion_plan_head`，`lr=1.5e-5` | `0.4126 / 0.5269` | `0.3796 / 1.2495 / 0.4958 / 0.3470 / 0.6213` | `0.5515` | `0.4961 / 0.6435 / 1.0283 / 0.1327 ; 0.4149 / 0.7070 / 1.0328 / 0.1428` | `0.6702% / 0.1118% / 0.6311` | 当前最优 L2 与 NDS |
| exp23 | e2e | `sparsedrive_small_stage2_exp23.py` | `a4ee13a` | `exp23` | `exp22/latest.pth` | L2-focused 微调，`lr=5e-6`，`motion loss=0`，`plan cls/status=0` | `0.4114 / 0.5249` | `0.3772 / 1.2523 / 0.4915 / 0.3418 / 0.6162` | `0.5515` | `0.4944 / 0.6485 / 1.0382 / 0.1349 ; 0.4156 / 0.7061 / 1.0318 / 0.1404` | `0.6702% / 0.1096% / 0.6657` | 过度压缩 loss，L2 回升 |
| exp24 | e2e | `sparsedrive_small_stage2_exp24.py` | `a4ee13a` | `exp24` | `exp23/iter_210975.pth` | 稳定性修正，`lr=5e-6`，`motion loss=0`，`plan cls=0.2`，`status=0` | `0.4121 / 0.5262` | `0.3772 / 1.2584 / 0.5325 / 0.3433 / 0.6399` | `0.5516` | `0.4946 / 0.6516 / 1.0472 / 0.1353 ; 0.4155 / 0.7102 / 1.0409 / 0.1426` | `0.6702% / 0.2089% / 0.7116` | 碰撞和 L2 同时恶化 |
| exp25 | e2e | `sparsedrive_small_stage2_exp25.py` | `7d80546` | `exp25` | `exp22/iter_140650.pth` | 稳健版重启，`lr=3e-6`，`motion loss=0.1`，`plan status=0.2` | `0.4133 / 0.5264` | `0.3808 / 1.2508 / 0.5028 / 0.3493 / 0.6299` | `0.5514` | `0.4986 / 0.6466 / 1.0385 / 0.1305 ; 0.4158 / 0.7079 / 1.0340 / 0.1449` | `0.6702% / 0.1042% / 0.6464` | 次优平衡点 |
| exp26 | e2e | `sparsedrive_small_stage2_exp26.py` | `7d80546` | `exp26` | `ckpt/sparsedrive_stage2.pth` | flow 版 `pv_recon`，`pv_recon_type='flow'`，`trajectory_source='pred'` | `0.4123 / 0.5258` | `0.3810 / 1.2493 / 0.4842 / 0.3461 / 0.6212` | `0.5511` | `0.5009 / 0.6235 / 0.9929 / 0.1287 ; 0.4114 / 0.7258 / 1.0655 / 0.1449` | `0.6702% / 0.1552% / 0.6603` | 主表记最终 ckpt；家族内最佳 planning 是 `iter_70325, L2=0.6329` |
| exp27 | e2e | `sparsedrive_small_stage2_exp27.py` | `e975c8d` | `exp27` | `exp26/iter_70325.pth` | 从 `exp26` 最佳 planning 点继续，冻结感知/检测/建图与 `pv_recon`，`lr=3e-6` | `0.4015 / 0.5157` | `0.3656 / 1.2651 / 0.4793 / 0.3341 / 0.6215` | `0.5401` | `0.4869 / 0.6319 / 0.9821 / 0.1320 ; 0.4042 / 0.7253 / 1.0588 / 0.1449` | `0.6702% / 0.1129% / 0.6268` | 主表记标准 eval 最终 ckpt；家族内最佳 planning 是 `iter_42195, L2=0.6222`；latest 的 masked eval 为 `NDS=0.5076, L2=0.6344` |
| exp28 | e2e | `sparsedrive_small_stage2_exp28.py` | `6e6df2a` | `exp27` | `exp27/iter_42195.pth` | masked robustness 微调，放开 `pv_recon/temporal_completion`，短 curriculum，`lambda_flow=0.003` | `0.3883 / 0.5042` | `0.3511 / 1.2856 / 0.4956 / 0.3177 / 0.6713` | `0.5124` | `0.4780 / 0.6433 / 0.9957 / 0.1326 ; 0.3952 / 0.7262 / 1.0534 / 0.1460` | `0.6702% / 0.1053% / 0.6394` | masked eval 口径；family 内 best NDS 是 `iter_14064=0.5062`，best L2 是 `iter_28130=0.6394`；未超过 `exp27` masked |
| 官方权重无mask | official | `N/A` | `N/A` | `官方` | `官方 checkpoint` | 标准评估，无相机缺失 | `-` | `0.3706 / 1.2550 / 0.5014 / 0.3486 / 0.6270` | `-` | `-` | `-` | 官方参考 |
| 官方权重mask后 | official | `N/A` | `N/A` | `官方` | `官方 checkpoint` | mask 鲁棒性评估 | `-` | `0.3543 / 1.2833 / 0.5075 / 0.3298 / 0.6545` | `-` | `-` | `-` | 相机缺失降低 tracking |

## 结论摘要

- 当前 `work_dirs` 中规划 `L2` 最好的有效实验是 `exp27`，最终 checkpoint `L2 = 0.6268`
- 如果按家族内最佳 checkpoint 看，当前 `work_dirs` 中最好的 planning 结果也是 `exp27`，`iter_42195` 的 `L2 = 0.6222`
- 当前 `work_dirs` 中 `NDS` 最好的有效实验也是 `exp22`，`NDS = 0.5269`
- 当前 `work_dirs` 中 `AMOTA` 最好的完整 e2e 实验是 `exp19`，`AMOTA = 0.3905`
- `exp22-0` 的 `mAP` 虽然最高，但这是一次误加载权重的无效对照，不能作为“基于 exp20 微调”的有效结论
- 从当前结果看，`exp22` 仍是最均衡的一次规划微调；但如果只追求 planning，`exp27` 已经超过 `exp22`
- `exp26` 证明 flow 版 `pv_recon` 训练是可行的；`exp27` 则进一步证明，从 `exp26` 的早期 planning checkpoint 出发做低学习率微调是有效的
- `exp27` 的 latest checkpoint 在 masked eval 下会退化到 `mAP=0.3896`、`NDS=0.5076`、`AMOTA=0.3517`、`L2=0.6344`，说明它具备一定鲁棒性，但不是专门为 masked eval 优化的最强模型
- `exp28` 继续针对 masked robustness 微调后，最终得到 `mAP=0.3883`、`NDS=0.5042`、`AMOTA=0.3511`、`obj_box_col=0.105%`、`L2=0.6394`；碰撞更低，但 `L2/NDS` 没有超过 `exp27` 的 masked baseline

## 一、完整 E2E 实验

这些目录包含 `e2e_metrics.json`，可以同时比较 detection、tracking、motion、planning。

| 实验 | 权重来源 | 主要设置 | 最新 checkpoint | mAP | NDS | AMOTA | car EPA | ped EPA | obj_box_col | L2 | 备注 |
|------|----------|----------|-----------------|-----|-----|-------|---------|---------|-------------|----|------|
| exp19 | `ckpt/sparsedrive_stage2.pth` | `trajectory_source='none'` | `iter_281300` | 0.4137 | 0.5244 | 0.3905 | 0.5018 | 0.4111 | 0.177% | 0.6448 | 完整训练，tracking 最强 |
| exp20 | `ckpt/sparsedrive_stage2.pth` | `trajectory_source='pred'` | `iter_281300` | 0.4131 | 0.5258 | 0.3792 | 0.4977 | 0.4109 | 0.103% | 0.6429 | 规划基线，后续微调起点 |
| exp21 | `ckpt/sparsedrive_stage2.pth` | `trajectory_source='gt'` | `iter_140650` | 0.4095 | 0.5229 | 0.3710 | 0.4917 | 0.4005 | 0.092% | 0.7314 | 轨迹引导改为 GT，规划退化明显 |
| exp22-0 | `ckpt/sparsedrive_stage2.pth` | 冻结大部分模块，仅训 `motion_plan_head` | `iter_70325` | 0.4145 | 0.5246 | 0.3699 | 0.4856 | 0.4115 | 0.246% | 0.7900 | 误加载权重，无效对照 |
| exp22 | `exp20/iter_281300.pth` | 冻结大部分模块，仅训 `motion_plan_head`，`lr=1.5e-5` | `iter_140650` | 0.4126 | 0.5269 | 0.3796 | 0.4961 | 0.4149 | 0.112% | 0.6311 | 当前最优 L2 和 NDS |
| exp23 | `exp22/latest.pth` | 延续冻结微调，`lr=5e-6` | `iter_281300` | 0.4114 | 0.5249 | 0.3772 | 0.4944 | 0.4156 | 0.110% | 0.6657 | 继续训练后 L2 回升 |
| exp24 | `exp23/iter_210975.pth` | 延续冻结微调，`lr=5e-6` | `iter_140650` | 0.4121 | 0.5262 | 0.3772 | 0.4946 | 0.4155 | 0.209% | 0.7116 | 规划与碰撞同时退化 |
| exp25 | `exp22/iter_140650.pth` | 从 exp22 中期重启，`lr=3e-6` | `iter_281300` | 0.4133 | 0.5264 | 0.3808 | 0.4986 | 0.4158 | 0.104% | 0.6464 | 次优平衡点，tracking 稳 |
| exp26 | `ckpt/sparsedrive_stage2.pth` | `pv_recon_type='flow'`，`trajectory_source='pred'` | `iter_281300` | 0.4123 | 0.5258 | 0.3810 | 0.5009 | 0.4114 | 0.155% | 0.6603 | flow 补全版本，tracking/motion 强于 exp25，planning 一般 |
| exp27 | `exp26/iter_70325.pth` | 冻结感知/检测/建图与 `pv_recon`，仅保留 planning 相关模块继续训练，`lr=3e-6` | `iter_70325` | 0.4015 | 0.5157 | 0.3656 | 0.4869 | 0.4042 | 0.113% | 0.6268 | 最终 ckpt 在 planning `L2` 上已优于 exp22；家族内最佳 planning 为 `iter_42195, L2=0.6222` |
| exp28 | `exp27/iter_42195.pth` | 放开 `pv_recon/temporal_completion`，短 curriculum，`lambda_flow=0.003`，masked eval 口径 | `iter_28130` | 0.3883 | 0.5042 | 0.3511 | 0.4780 | 0.3952 | 0.105% | 0.6394 | 这是 masked eval 实验，不纳入无 mask 主排名；family 内 best NDS 为 `iter_14064, 0.5062` |

### 关键观察

- `exp19 -> exp20 -> exp21` 主要是在比较 `planning_guided_completion` 中的 `trajectory_source`
- 从结果看，`pred` 比 `gt` 更稳，`gt` 明显拉高了 `L2`
- `exp22` 开始引入冻结微调，目标是基于 `exp20` 压低规划误差
- `exp22-0` 因为仍从 `ckpt/sparsedrive_stage2.pth` 启动，所以虽然目录名是 `exp22-0`，但不能代表“基于 exp20 微调”的效果
- `exp22` 修正初始化后，`L2` 从 `0.6429` 降到 `0.6311`，说明这条微调路线是有效的
- `exp23` 和 `exp24` 继续沿这条路线训练，整体没有超过 `exp22`
- `exp25` 选择从 `exp22` 的中期 checkpoint 重启，并降低 `lr` 到 `3e-6`，结果比 `exp23/24` 稳，但仍略逊于 `exp22`
- `exp26` 是另一条实验线，不属于 `exp22-25` 的冻结微调家族；它通过 `flow` 版 `pv_recon` 保住了较强的 tracking/motion
- `exp26` 如果只看最终 checkpoint，规划 `L2=0.6603` 不如 `exp22` 和 `exp25`
- 但如果看整个 `exp26` 家族，第一次评测 `iter_70325` 的 `L2=0.6329` 实际已经非常接近 `exp22`
- `exp27` 正是在 `exp26/iter_70325` 基础上继续做低学习率规划微调，结果把最终 `L2` 压到 `0.6268`
- 如果看 `exp27` 家族内最佳 checkpoint，`iter_42195` 的 `L2=0.6222` 已经是当前所有有效实验中最好的 planning 结果
- 但如果把 `exp27` 的 latest checkpoint 切到 masked eval，`L2` 会从 `0.6268` 回升到 `0.6344`，`NDS` 会从 `0.5157` 降到 `0.5076`
- 这说明 `exp27` 更像“标准评测下的 planning 微调最优点”，而不是“mask 鲁棒性最优点”
- `exp28` 试图继续优化 mask robustness，但最终 `L2=0.6394` 仍然高于 `exp27 masked = 0.6344`
- 不过 `exp28` 把 `obj_box_col` 压到了 `0.105%`，说明它在 masked 口径下更安全，只是这种改进没有同步转化成更低的 `L2`
- 因为 `exp28` 是 masked eval 口径，下面的 `L2` 排名仍只统计标准 eval 的实验

### 排名参考

按 `L2` 从优到劣排序：

1. `exp27`：0.6268
2. `exp22`：0.6311
3. `exp20`：0.6429
4. `exp19`：0.6448
5. `exp25`：0.6464
6. `exp26`：0.6603
7. `exp23`：0.6657
8. `exp24`：0.7116
9. `exp21`：0.7314
10. `exp22-0`：0.7900

## 二、历史 Tracking 实验

这些目录只有 `metrics_summary.json`，可用于 tracking 历史对比，但不适合和完整 e2e 实验直接比较 motion/planning。

| 实验 | 权重来源 | 可确认信息 | AMOTA | AMOTP | Recall | MOTA | MOTP | 备注 |
|------|----------|------------|-------|-------|--------|------|------|------|
| exp8 | 未单独整理 | 仅保留 tracking 结果 | 0.3866 | 1.2526 | 0.4831 | 0.3539 | 0.6268 | 早期较强 tracking 基线 |
| exp9 | 未单独整理 | 仅保留 tracking 结果 | 0.3289 | 1.2932 | 0.4537 | 0.3114 | 0.6733 | 相比 exp8 明显退化 |
| exp13 | `ckpt/sparsedrive_stage2.pth` | `lr=3.5e-5` | 0.1865 | 1.5158 | 0.2837 | 0.1933 | 0.8917 | 明显异常，建议视为失败实验 |
| exp14 | `ckpt/sparsedrive_stage2.pth` | `lr=3.5e-5` | 0.3749 | 1.2624 | 0.5087 | 0.3368 | 0.6538 | 恢复正常 |
| exp15 | `ckpt/sparsedrive_stage2.pth` | `lr=3.5e-5` | 0.3702 | 1.2695 | 0.4812 | 0.3406 | 0.6536 | 与 exp14 接近 |
| exp16 | `exp15/iter_210960.pth` | `lr=3.5e-5` | 0.3730 | 1.2587 | 0.4913 | 0.3418 | 0.6408 | 在 exp15 基础上小幅提升 |
| exp17 | `ckpt/sparsedrive_stage2.pth` | `lr=1.5e-5` | 0.3715 | 1.2616 | 0.4850 | 0.3370 | 0.6373 | 进入较稳定区间 |
| exp18 | `ckpt/sparsedrive_stage2.pth` | 启用规划引导补全相关配置 | 0.3696 | 1.2691 | 0.4915 | 0.3380 | 0.6517 | 为 exp19 之前版本 |
| 官方权重无mask | 官方 checkpoint | 无相机缺失评估 | 0.3706 | 1.2550 | 0.5014 | 0.3486 | - | 官方参考 |
| 官方权重mask后 | 官方 checkpoint | 有相机缺失评估 | 0.3543 | 1.2833 | 0.5075 | 0.3298 | - | 相机缺失会拉低 tracking |

## 三、非同口径结果与未纳入比较的目录

以下目录不放进上面的主表，原因可能是任务口径不同、缺少完整评估文件，或者本身不是实验输出目录：

- `ckpt/`
- `docs/`
- `mystart_train_smoke/`
- `projects/`
- `quickeval_test/`
- `resources/`
- `scripts/`
- `sparsedrive_small_stage1/`
- `sparsedrive_small_stage1_before/`：`stage1` 配置，tracking 指标约为 `AMOTA=0.0014`、`MOTA=0.0070`，与 `stage2` e2e 实验不具可比性
- `sparsedrive_small_stage2_before/`
- `stability_check_local/`
- `tools/`
- `work_dirs/`

## 四、建议的命名和归档规则

- 将 `exp22-0` 明确标注为“误加载权重，无效实验”
- 以后新实验目录建议同时在文件名和 `work_dir` 中保持一致，避免出现目录名和实际初始化来源不一致
- 每次实验至少保留配置快照
- 每次实验至少保留 `metrics_summary.json`
- 每次实验至少保留 `e2e_metrics.json`
- 每次实验至少补一句实验目的说明
