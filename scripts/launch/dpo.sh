#!/usr/bin/env bash
# Stage 4/4 -- Direct Preference Optimization on paired preference data.
#
# All ScriptArguments in dpo.py can be overridden by appending flags, e.g.:
#   bash scripts/launch/dpo.sh --beta 0.05 --sanity_check True
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_configs/ds_stage2.yaml}"

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  dpo.py \
  "$@"
