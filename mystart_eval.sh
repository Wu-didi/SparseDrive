#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-projects/configs/sparsedrive_small_stage2_exp28.py}"
CONFIG_NAME="$(basename "${CONFIG%.py}")"
CHECKPOINT="${CHECKPOINT:-work_dirs/${CONFIG_NAME}/latest.pth}"
EVAL_METRIC="${EVAL_METRIC:-bbox}"
WORK_DIR="${WORK_DIR:-}"
EVAL_TAG="${EVAL_TAG:-}"
ORIG_ARGS=("$@")

EXTRA_ARGS=()
USER_CFG_OPTIONS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg-options)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        USER_CFG_OPTIONS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${EVAL_TAG}" ]]; then
  EVAL_TAG="mask"
  for opt in "${USER_CFG_OPTIONS[@]}"; do
    case "${opt}" in
      model.test_cam_missing=False)
        EVAL_TAG="standard"
        ;;
      model.test_cam_missing=True)
        EVAL_TAG="mask"
        ;;
    esac
  done
fi

if [[ -z "${WORK_DIR}" ]]; then
  WORK_DIR="./work_dirs/${CONFIG_NAME}_${EVAL_TAG}_eval"
fi

CFG_OPTIONS=(
  "work_dir=${WORK_DIR}"
  "data.workers_per_gpu=0"
  "${USER_CFG_OPTIONS[@]}"
)

# 优先保证在指定 conda 环境下运行
if [[ "${CONDA_DEFAULT_ENV:-}" != "sparsedrive" ]]; then
  if command -v conda >/dev/null 2>&1; then
    echo "[INFO] Re-running inside conda env: sparsedrive"
    exec conda run --no-capture-output -n sparsedrive bash "$0" "${ORIG_ARGS[@]}"
  else
    echo "[WARN] conda not found; continue with current python" >&2
  fi
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${WORK_DIR}"
echo "[INFO] Eval tag: ${EVAL_TAG}"
echo "[INFO] Work dir: ${WORK_DIR}"

python tools/test.py \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  --eval "${EVAL_METRIC}" \
  --deterministic \
  --cfg-options "${CFG_OPTIONS[@]}" \
  "${EXTRA_ARGS[@]}"

if [[ -f "${WORK_DIR}/e2e_metrics.json" ]]; then
  echo "[INFO] Saved: ${WORK_DIR}/e2e_metrics.json"
fi
