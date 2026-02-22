#!/bin/bash
set -euo pipefail

cd /home/wudi/code/mySparseDrive/SparseDrive

PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
python tools/train.py projects/configs/sparsedrive_small_stage2.py "$@"
