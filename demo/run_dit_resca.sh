#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIT_DIR="${PROJECT_ROOT}/dit/resca-dit"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/demo/dit_resca"

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

python "${DIT_DIR}/sample.py" \
  --class-labels "207,360,387,974" \
  --num-sampling-steps 50 \
  --interval 4 \
  --max-order 1 \
  --cluster-num 16 \
  --k 1 \
  --resca-proxy-method "center-random"
