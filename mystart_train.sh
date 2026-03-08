#!/bin/bash
set -euo pipefail

cd /home/wudi/code/mySparseDrive/SparseDrive

export PYTHONPATH="/home/wudi/code/mySparseDrive/SparseDrive:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1  # 仅调试 CUDA 报错时开启

python tools/train.py projects/configs/sparsedrive_small_stage2_exp23.py "$@"
