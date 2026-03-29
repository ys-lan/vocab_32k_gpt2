#export NCCL_P2P_LEVEL=NVL
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config_file configs/accelerate_configs/ds_stage2.yaml train.py --train_config configs/instruct_config.yaml --model_config configs/model_configs/vocab_32k_gpt2.json 

