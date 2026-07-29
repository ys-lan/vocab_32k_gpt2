#!/usr/bin/env bash
# Stage 3/4 -- extra SFT pass on the prompts used for preference data.
#
# This keeps the policy in-distribution with the DPO dataset, which measurably
# stabilises the DPO stage that follows.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_configs/ds_stage2.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/dpo_instruct_config.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_configs/vocab_32k_gpt2.json}"

# Uncomment when peer-to-peer or InfiniBand transport misbehaves on this host.
# export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  train.py \
  --train_config "$TRAIN_CONFIG" \
  --model_config "$MODEL_CONFIG" \
  "$@"
