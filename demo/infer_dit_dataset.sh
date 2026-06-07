#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIT_DIR="${PROJECT_ROOT}/dit/resca-dit"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/datasets/dit_resca"

torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  --master_port 29500 \
  "${DIT_DIR}/sample_ddp.py" \
  --sample-dir "${OUTPUT_DIR}" \
  --num-sampling-steps 250 \
  --interval 4 \
  --max-order 1 \
  --cluster-num 16 \
  --k 1 \
  --resca-proxy-method "center-random"
