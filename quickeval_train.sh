#!/bin/bash
# 快速测试eval流程：只训练1个iteration，然后立即eval
cd /mnt/home/wudi/code_v3/SparseDrive

echo "=================================="
echo "快速测试Eval流程"
echo "- 只运行1个训练iteration"
echo "- 然后立即触发eval"
echo "- 使用完整验证集（6019个样本）"
echo "- 预计需要5-10分钟"
echo "=================================="

PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=0 \
python tools/train.py \
    projects/configs/sparsedrive_quickeval_test.py \
    --deterministic

echo ""
echo "如果eval过程没有报错，说明修复成功！"
