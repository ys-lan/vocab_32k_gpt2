SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate_configs/ds_stage2.yaml dpo.py