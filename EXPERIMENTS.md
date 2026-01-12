# SparseDrive 实验记录

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
