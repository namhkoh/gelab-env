#!/usr/bin/bash
# =============================================================================
# LoRA experiment chain: 4 runs × [LoRA FT → eval → upload → append]
# Goal: test whether LoRA (frozen base) produces better grounding than full-FT.
# =============================================================================
set -eo pipefail
cd /workspace/gelab-env

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
export WANDB_ENTITY="${WANDB_ENTITY:-namhokoh-korea-advanced-institute-of-science-and-technology}"
export WANDB_PROJECT="${WANDB_PROJECT:-gelab}"
export DATASET_ROOT="${DATASET_ROOT:-/workspace/gelab-env/datas_amex/amex-augmented-sft}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/workspace/gelab-env/checkpoint/gui_exp}"
export LOG_DIR="${LOG_DIR:-/workspace/gelab-env/logs/train}"
mkdir -p "$LOG_DIR" "$BASE_OUTPUT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate gelab

AMEX="/workspace/gelab-env/datas_amex"

# Each entry: RUN_NAME|DATASET|TRAIN_TYPE|LEARNING_RATE
RUNS=(
  "lora_mix_21k_lr5e5|$AMEX/pb_t1c_mix_21k.json|lora|5e-5"
  "lora_mix_21k_lr1e4|$AMEX/pb_t1c_mix_21k.json|lora|1e-4"
  "lora_mix_80k_lr5e5|$AMEX/pb_t1c_mix_80k.json|lora|5e-5"
  "lora_mix_80k_lr1e4|$AMEX/pb_t1c_mix_80k.json|lora|1e-4"
)

CHAIN_STAMP=$(date '+%Y%m%d_%H%M%S')
CHAIN_LOG="$LOG_DIR/chain_${CHAIN_STAMP}.log"
echo "[chain] LoRA series starting @ $(date) — log: $CHAIN_LOG" | tee -a "$CHAIN_LOG"

for spec in "${RUNS[@]}"; do
  IFS="|" read -r name path ttype lr <<< "$spec"
  if [ ! -f "$path" ]; then
    echo "[chain] SKIP $name: dataset $path missing" | tee -a "$CHAIN_LOG"; continue
  fi
  n=$(python -c "import json; print(len(json.load(open('$path'))))")
  echo "" | tee -a "$CHAIN_LOG"
  echo "============================================================" | tee -a "$CHAIN_LOG"
  echo "[chain] $name — $n samples — $ttype $lr — $(date)" | tee -a "$CHAIN_LOG"
  echo "============================================================" | tee -a "$CHAIN_LOG"

  env RUN_NAME="$name" DATASET_PATH="$path" TRAIN_TYPE="$ttype" LEARNING_RATE="$lr" \
    bash /workspace/gelab-env/scripts/train_eval_upload.sh 2>&1 | tee -a "$CHAIN_LOG" \
    || echo "[chain] WARN: $name exited non-zero, continuing" | tee -a "$CHAIN_LOG"

  echo "[chain] $name done @ $(date)" | tee -a "$CHAIN_LOG"
done

echo "[chain] LoRA series COMPLETE @ $(date)" | tee -a "$CHAIN_LOG"
