#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLUX_DIR="${PROJECT_ROOT}/flux/resca-flux"

python "${FLUX_DIR}/sample.py" \
  --prompt_file "${PROJECT_ROOT}/assets/prompts/PartiPrompts1632.txt" \
  --output_dir "${PROJECT_ROOT}/outputs/datasets/flux_resca" \
  --model_name "flux-dev" \
  --width 1024 \
  --height 1024 \
  --num_steps 50 \
  --fresh_threshold 5 \
  --max_order 1 \
  --cluster_num 16 \
  --k 1 \
  --resca_proxy_method "center-random"
