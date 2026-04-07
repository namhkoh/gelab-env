#!/usr/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_DIR="${ENV_DIR:-/home/irteam/data-vol1/amex_sft_hf_448}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/irteam/data-vol1/gelab-env/datas/st_rl_amex.json}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/data}"

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_PATH")"

time_start=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/generate_amex_st_rl_data_${time_start}.log"

echo "============================================================"
echo "GENERATE AMEX ST-RL DATA"
echo "============================================================"
echo "Start time: $(date)"
echo "Env dir:    $ENV_DIR"
echo "Output:     $OUTPUT_PATH"
echo "Log file:   $LOG_FILE"
echo "============================================================"

cd "$REPO_ROOT"
python data_engine/generate_amex_st_rl_data.py \
    --env_dir "$ENV_DIR" \
    --output "$OUTPUT_PATH" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "AMEX ST-RL DATA GENERATION COMPLETE"
echo "End time: $(date)"
echo "Output:   $OUTPUT_PATH"
echo "Log file: $LOG_FILE"
echo "============================================================"
