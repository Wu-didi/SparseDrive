#!/bin/bash
set -euo pipefail

export PYTHONPATH="/home/wudi/code/mySparseDrive/SparseDrive:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=6

CONFIG="${CONFIG:-projects/configs/sparsedrive_small_stage2_exp38.py}"
SEED="${SEED:-42}"
WORK_DIR="${WORK_DIR:-./work_dirs/sparsedrive_small_stage2_exp38}"

python tools/train.py "${CONFIG}" \
  --seed "${SEED}" \
  --deterministic \
  --work-dir "${WORK_DIR}"
