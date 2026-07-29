#!/usr/bin/env bash
# Stage 1/4 -- pretraining from scratch on raw text corpora.
#
# Override any setting from the environment, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1 ACCELERATE_CONFIG=configs/accelerate_configs/ds_stage3.yaml \
#     bash scripts/launch/pre_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_configs/ds_stage2.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/pretrain_config.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_configs/vocab_32k_gpt2.json}"

# Uncomment on hosts where NCCL needs tuning to saturate the interconnect.
# export NCCL_SOCKET_NTHREADS=16

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  train.py \
  --train_config "$TRAIN_CONFIG" \
  --model_config "$MODEL_CONFIG" \
  "$@"
