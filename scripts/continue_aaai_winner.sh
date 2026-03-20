#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <config> <checkpoint> <work_dir> <gpus> [max_iters]"
    exit 1
fi

CONFIG=$1
CHECKPOINT=$2
WORK_DIR=$3
GPUS=$4
MAX_ITERS=${5:-28130}
INTERVAL=${AAAI_EVAL_INTERVAL:-7032}

mkdir -p "${WORK_DIR}"

echo "[AAAI continue] config=${CONFIG}"
echo "[AAAI continue] checkpoint=${CHECKPOINT}"
echo "[AAAI continue] work_dir=${WORK_DIR}"
echo "[AAAI continue] max_iters=${MAX_ITERS}"

bash ./tools/dist_train.sh "${CONFIG}" "${GPUS}" \
    --deterministic \
    --cfg-options \
    "load_from=${CHECKPOINT}" \
    "work_dir=${WORK_DIR}" \
    "runner.max_iters=${MAX_ITERS}" \
    "checkpoint_config.interval=${INTERVAL}" \
    "evaluation.interval=${INTERVAL}"
