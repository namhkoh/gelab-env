#!/usr/bin/bash
# =============================================================================
# Re-evaluate all trained full-FT checkpoints using the per-benchmark eval
# scripts (computer_use guided generation). Excludes LoRA checkpoints (adapters
# not supported by eval_*.py). Includes base Qwen as anchor.
# =============================================================================
set -eo pipefail
cd /workspace/gelab-env

export HF_TOKEN="${HF_TOKEN:?}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

source /opt/conda/etc/profile.d/conda.sh
conda activate gelab

LOG_DIR="/workspace/gelab-env/logs/eval"
mkdir -p "$LOG_DIR"

STAMP=$(date '+%Y%m%d_%H%M%S')
MASTER_LOG="$LOG_DIR/sweep_${STAMP}.log"
echo "[sweep] starting @ $(date) — log: $MASTER_LOG" | tee -a "$MASTER_LOG"

latest_ckpt () {
  ls -td "/workspace/gelab-env/checkpoint/gui_exp/aug_$1/v0-"*/checkpoint-* 2>/dev/null | head -1
}

# Models to evaluate. Order: base first (cheap anchor), then results roughly
# best→worst to surface meaningful numbers early.
MODELS=(
  "base_qwen|Qwen/Qwen2.5-VL-7B-Instruct"
  "t1a_aug_21k|$(latest_ckpt t1a_aug_21k)"
  "t1b_orig_21k|$(latest_ckpt t1b_orig_21k)"
  "t1c_mix_21k|$(latest_ckpt t1c_mix_21k)"
  "t2a_aug_nav_21k|$(latest_ckpt t2a_aug_nav_21k)"
  "t2b_orig_nav_21k|$(latest_ckpt t2b_orig_nav_21k)"
  "r0_real_only_21k|$(latest_ckpt r0_real_only_21k)"
  "r1_real_aug_8020_21k|$(latest_ckpt r1_real_aug_8020_21k)"
  "r2_real_aug_5050_21k|$(latest_ckpt r2_real_aug_5050_21k)"
  "r_real_aug_9010_21k|$(latest_ckpt r_real_aug_9010_21k)"
  "r_real_aug_8515_21k|$(latest_ckpt r_real_aug_8515_21k)"
  "r_real_aug_7030_21k|$(latest_ckpt r_real_aug_7030_21k)"
  "s1_success_nav_orig|$(latest_ckpt s1_success_nav_orig)"
  "s3_success_nav_aug_21k|$(latest_ckpt s3_success_nav_aug_21k)"
  "s2_success_nav_aug|$(latest_ckpt s2_success_nav_aug)"
  "c1_combined_nav_21k|$(latest_ckpt c1_combined_nav_21k)"
  "c2_combined_success_nav_21k|$(latest_ckpt c2_combined_success_nav_21k)"
)

for spec in "${MODELS[@]}"; do
  IFS="|" read -r name path <<< "$spec"
  if [ -z "$path" ]; then
    echo "[sweep] SKIP $name: no checkpoint path" | tee -a "$MASTER_LOG"
    continue
  fi
  if [[ "$path" != Qwen/* ]] && [ ! -f "$path/config.json" ]; then
    echo "[sweep] SKIP $name: $path/config.json missing" | tee -a "$MASTER_LOG"
    continue
  fi
  echo "" | tee -a "$MASTER_LOG"
  echo "================================================================" | tee -a "$MASTER_LOG"
  echo "[sweep] >>> $name @ $(date)" | tee -a "$MASTER_LOG"
  echo "[sweep] >>> model: $path" | tee -a "$MASTER_LOG"
  echo "================================================================" | tee -a "$MASTER_LOG"

  env RUN_NAME="$name" MODEL_PATH="$path" \
    bash /workspace/gelab-env/scripts/eval_one_model.sh 2>&1 | tee -a "$MASTER_LOG" \
    || echo "[sweep] WARN $name eval exited non-zero, continuing" | tee -a "$MASTER_LOG"
done

# Aggregate results into a single JSON + append to results_individual.md
python /workspace/gelab-env/scripts/aggregate_individual_evals.py || echo "[sweep] WARN aggregator failed"
echo "[sweep] COMPLETE @ $(date)" | tee -a "$MASTER_LOG"
