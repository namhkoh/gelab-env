#!/usr/bin/bash
# =============================================================================
# C chain: COMBINED nav-only (orig + aug unioned)
#   C1 = 10.5k orig-nav + 10.5k aug-nav (all trajectories)
#   C2 = 10.5k orig-success + 10.5k aug-success (success filtered)
# Both at 21k budget for apples-to-apples vs T2.A/B and S1/S3.
# Same recipe as every Path B run: base Qwen, LR=1e-6, 1 ep, max_pixels=1M.
# =============================================================================
set -eo pipefail
cd /workspace/gelab-env

export HF_TOKEN="${HF_TOKEN:?}"
export WANDB_API_KEY="${WANDB_API_KEY:?}"
export WANDB_ENTITY="${WANDB_ENTITY:-namhokoh-korea-advanced-institute-of-science-and-technology}"
export WANDB_PROJECT="${WANDB_PROJECT:-gelab}"
export LOG_DIR="${LOG_DIR:-/workspace/gelab-env/logs/train}"
mkdir -p "$LOG_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate gelab

AMEX="/workspace/gelab-env/datas_amex"

# RUN_NAME | DATASET | TRAIN_TYPE | LR
RUNS=(
  "c1_combined_nav_21k|$AMEX/pb_c1_combined_nav_21k.json|full|1e-6"
  "c2_combined_success_nav_21k|$AMEX/pb_c2_combined_success_nav_21k.json|full|1e-6"
)

CHAIN_STAMP=$(date '+%Y%m%d_%H%M%S')
CHAIN_LOG="$LOG_DIR/chain_${CHAIN_STAMP}.log"
echo "[chain] C chain starting @ $(date) — log: $CHAIN_LOG" | tee -a "$CHAIN_LOG"

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

echo "[chain] C chain COMPLETE @ $(date)" | tee -a "$CHAIN_LOG"
