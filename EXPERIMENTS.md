# 实验记录

记录每次改进的实验，方便追踪和对比。

---

## 实验列表

### EXP_001 - 示例：eval阶段添加torch.no_grad()防止显存溢出
- **Git Hash**: `f32dd45`
- **改进描述**: 在评估阶段添加torch.no_grad()上下文管理器，减少显存占用
- **实验结果**:
  - mAP: XX.XX%
  - NDS: XX.XX%
- **Log路径**: `work_dirs/xxx/xxx.log`
- **Checkpoint**: `work_dirs/xxx/epoch_24.pth`
- **备注**:

---

### EXP_002 - [改进描述]
- **Git Hash**: `xxxxxxx`
- **改进描述**: [简短描述这次改了什么]
- **实验结果**:
  - mAP:
  - NDS:
- **Log路径**:
- **Checkpoint**:
- **备注**:

---

### EXP_003 - [改进描述]
- **Git Hash**:
- **改进描述**:
- **实验结果**:
  - mAP:
  - NDS:
- **Log路径**:
- **Checkpoint**:
- **备注**:

---

### EXP_010 - 显存优化与稳定性改进
- **Git Hash**: `f32dd45`
- **改进描述**:
  - eval阶段添加torch.no_grad()防止显存溢出
  - 添加img_metas空值检查增强鲁棒性
  - 修复simple_test中temporal_completion调用接口
  - 使用显存优化版运动补偿时序补全模块
- **实验结果**:
  - mAP:
  - NDS:
- **Log路径**:
- **Checkpoint**:
- **备注**:

---

## 快速对比表

| ID | Git Hash | 改进描述 | mAP | NDS | Checkpoint |
|----|----------|----------|-----|-----|------------|
| EXP_001 | f32dd45 | 示例 | XX.XX | XX.XX | work_dirs/xxx/epoch_24.pth |
| EXP_002 |  |  |  |  |  |
| EXP_003 |  |  |  |  |  |
| EXP_010 | f32dd45 | 显存优化与稳定性改进 |  |  |  |

---

## 使用说明

每次做实验前：
1. 复制一个实验模板（EXP_XXX部分）
2. 填写改进描述
3. 运行 `git rev-parse --short HEAD` 获取当前git哈希，填入Git Hash
4. 训练完成后，填写实验结果、log路径和checkpoint路径
5. 在快速对比表中也添加一行，方便快速查看

---
