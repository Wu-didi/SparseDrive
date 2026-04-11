#!/bin/bash
set -euo pipefail

cd /home/wudidi/code_v6/SparseDrive

export PYTHONPATH="/home/wudidi/code_v6/SparseDrive:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=7
# export CUDA_LAUNCH_BLOCKING=1  # 仅调试 CUDA 报错时开启

CONFIG="${CONFIG:-projects/configs/sparsedrive_small_stage2_exp36_mask_only.py}"
SEED="${SEED:-3407}"
WORK_DIR="${WORK_DIR:-./work_dirs/sparsedrive_small_stage2_exp36_mask_only}"

python tools/train.py "${CONFIG}" \
  --seed "${SEED}" \
  --deterministic \
  --work-dir "${WORK_DIR}" \
  "$@"
