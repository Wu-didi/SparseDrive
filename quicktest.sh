#!/bin/bash
# 快速测试脚本 - 使用mini数据集（只有323个样本，而不是28130个）
cd /mnt/home/wudi/code_v3/SparseDrive

CONFIG="projects/configs/sparsedrive_small_stage2.py"
CHECKPOINT="work_dirs/sparsedrive_small_stage2/iter_70320.pth"

echo "=== 快速测试：使用mini数据集（323个样本） ==="
PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --eval bbox \
    --deterministic \
    --cfg-options data.val.ann_file=data/nuscenes/nuscenes_infos_temporal_val_mini.pkl

echo ""
echo "测试完成！如果没有报错，说明修复成功。"
