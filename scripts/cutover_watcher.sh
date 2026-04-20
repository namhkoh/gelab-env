#!/usr/bin/bash
# =============================================================================
# Watches the running tmux chain for T1.A training completion. Once T1.A's
# swift run logs `train_runtime`, this script:
#   1) Kills the current tmux session (stops the old chain before it starts T1.B)
#   2) Starts a new tmux session running the NEW chain (run_aug_chain.sh v2)
#   3) The new chain sees T1.A's checkpoint already exists -> SKIP_TRAIN=1 for
#      T1.A (just eval + upload + append), then trains T1.B-T2.D with full
#      eval + upload + append per run.
# =============================================================================
set -eo pipefail
OLD_SESSION="gelab-chain"
NEW_SESSION="gelab-chain"
LOG="/workspace/gelab-env/logs/train/cutover.log"

source /workspace/gelab-env/scripts/_chain_env.sh 2>/dev/null || {
  echo "ERROR: _chain_env.sh missing"; exit 2
}

echo "[cutover] watcher started @ $(date)" | tee -a "$LOG"
echo "[cutover] waiting for T1.A train_runtime signal..." | tee -a "$LOG"

while true; do
  CHAIN_LOG=$(ls -t /workspace/gelab-env/logs/train/chain_*.log 2>/dev/null | head -1)
  if [ -n "$CHAIN_LOG" ] && grep -qa "train_runtime" "$CHAIN_LOG" 2>/dev/null; then
    echo "[cutover] detected T1.A train_runtime @ $(date)" | tee -a "$LOG"
    grep -a "train_runtime" "$CHAIN_LOG" | tail -1 >> "$LOG"
    break
  fi
  sleep 60
done

# Wait an extra 30s for swift to finalize checkpoint save, then cut over
sleep 30

echo "[cutover] killing old tmux session ($OLD_SESSION)..." | tee -a "$LOG"
tmux kill-session -t "$OLD_SESSION" 2>/dev/null || true

# kill any leftover swift/python workers defensively
pkill -9 -f "swift/cli/sft.py" 2>/dev/null || true
pkill -9 -f "torch.distributed.run" 2>/dev/null || true
sleep 5

# Verify GPUs freed
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG"

echo "[cutover] starting new tmux session with v2 chain..." | tee -a "$LOG"
tmux new-session -d -s "$NEW_SESSION" -n train \
  "bash -lc 'source /workspace/gelab-env/scripts/_chain_env.sh && bash /workspace/gelab-env/scripts/run_aug_chain.sh; echo; echo \"[tmux] chain exited. Press any key.\"; read -n 1'"

sleep 3
tmux ls 2>&1 | tee -a "$LOG"
echo "[cutover] done @ $(date)" | tee -a "$LOG"
