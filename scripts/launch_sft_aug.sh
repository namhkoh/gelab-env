#!/usr/bin/bash
# Thin launcher: waits for HF snapshot to finish, then runs sft_amex_augmented.sh
# in the background with logs tailed to /workspace/gelab-env/logs/train/

set -e
cd /workspace/gelab-env

SNAP_DIR="/workspace/gelab-env/datas_amex/amex-augmented-sft"
LOG_DIR="/workspace/gelab-env/logs/train"
mkdir -p "$LOG_DIR"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
export WANDB_ENTITY="${WANDB_ENTITY:-namhokoh-korea-advanced-institute-of-science-and-technology}"
export WANDB_PROJECT="${WANDB_PROJECT:-gelab}"

# Wait for train.json and all 10k images
have_imgs=$(ls "$SNAP_DIR/images" 2>/dev/null | wc -l)
echo "[launcher] snapshot state: $have_imgs/10000 images, train.json=$(test -f $SNAP_DIR/train.json && echo yes || echo no)"
if [ "$have_imgs" -lt 10000 ] || [ ! -f "$SNAP_DIR/train.json" ]; then
  echo "[launcher] ERROR: snapshot incomplete; rerun /workspace/gelab-env/scripts/resume_snapshot.py first"
  exit 2
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate gelab

TS=$(date '+%Y%m%d_%H%M%S')
SCRIPT_LOG="$LOG_DIR/launch_${TS}.log"
echo "[launcher] starting sft_amex_augmented.sh — log: $SCRIPT_LOG"
nohup bash /workspace/gelab-env/gui_scripts/sft_amex_augmented.sh > "$SCRIPT_LOG" 2>&1 &
LP=$!
echo "[launcher] PID=$LP"
echo "$LP" > /workspace/gelab-env/logs/train/latest_pid.txt
