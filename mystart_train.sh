#!/bin/bash
set -euo pipefail

cd /home/wudi/code/mySparseDrive/SparseDrive

export PYTHONPATH="/home/wudi/code/mySparseDrive/SparseDrive:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1  # 仅调试 CUDA 报错时开启

CONFIG="${CONFIG:-projects/configs/sparsedrive_small_stage2_exp32.py}"
SEED="${SEED:-3407}"
WORK_DIR="${WORK_DIR:-./work_dirs/sparsedrive_small_stage2_exp32_seed3407}"

python tools/train.py "${CONFIG}" \
  --seed "${SEED}" \
  --deterministic \
  --work-dir "${WORK_DIR}" \
  "$@"
