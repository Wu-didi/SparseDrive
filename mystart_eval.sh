#!/bin/bash
cd /mnt/home/wudi/code_v3/SparseDrive

# 评估配置
CONFIG="projects/configs/sparsedrive_small_stage2.py"
CHECKPOINT="work_dirs/sparsedrive_small_stage2/iter_70320.pth"  # 修改为你的checkpoint路径
GPUS=1

# 单GPU评估
PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --eval bbox \
    --deterministic

# 如果要使用多GPU评估，使用下面的命令（取消注释）：
# bash ./tools/dist_test.sh \
#     $CONFIG \
#     $CHECKPOINT \
#     $GPUS \
#     --deterministic \
#     --eval bbox
