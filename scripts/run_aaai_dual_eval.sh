#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <config> <checkpoint> <gpus> [output_root]"
    exit 1
fi

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
OUTPUT_ROOT=${4:-$(dirname "$CHECKPOINT")/evals}
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
CHECKPOINT_STEM=${CHECKPOINT_NAME%.pth}
STANDARD_DIR="${OUTPUT_ROOT}/${CHECKPOINT_STEM}/standard"
MASKED_DIR="${OUTPUT_ROOT}/${CHECKPOINT_STEM}/masked"

mkdir -p "${STANDARD_DIR}" "${MASKED_DIR}"

COMMON_ARGS=(
    --deterministic
    --eval
    bbox
)

echo "[AAAI eval] standard -> ${STANDARD_DIR}"
bash ./tools/dist_test.sh "${CONFIG}" "${CHECKPOINT}" "${GPUS}" \
    "${COMMON_ARGS[@]}" \
    --out "${STANDARD_DIR}/results.pkl" \
    --cfg-options \
    "work_dir=${STANDARD_DIR}" \
    "model.test_cam_missing=False"

echo "[AAAI eval] masked -> ${MASKED_DIR}"
bash ./tools/dist_test.sh "${CONFIG}" "${CHECKPOINT}" "${GPUS}" \
    "${COMMON_ARGS[@]}" \
    --out "${MASKED_DIR}/results.pkl" \
    --cfg-options \
    "work_dir=${MASKED_DIR}" \
    "model.test_cam_missing=True"

echo "[AAAI eval] done: ${OUTPUT_ROOT}/${CHECKPOINT_STEM}"
