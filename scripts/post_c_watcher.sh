#!/usr/bin/bash
# Waits for C chain to finish, then launches the individual-eval sweep
# in a fresh tmux session.
set -eo pipefail
LOG="/workspace/gelab-env/logs/post_c_watcher.log"
mkdir -p "$(dirname "$LOG")"
source /workspace/gelab-env/scripts/_chain_env.sh

echo "[post-c] watcher armed @ $(date)" | tee -a "$LOG"

while tmux has-session -t gelab-chain 2>/dev/null; do
  LOG_FILE=$(ls -t /workspace/gelab-env/logs/train/chain_*.log 2>/dev/null | head -1)
  if [ -n "$LOG_FILE" ] && grep -q "C chain COMPLETE" "$LOG_FILE" 2>/dev/null; then
    echo "[post-c] C chain COMPLETE marker seen @ $(date)" | tee -a "$LOG"
    tmux kill-session -t gelab-chain 2>/dev/null || true
    break
  fi
  sleep 60
done

sleep 10
pkill -9 -f "swift/cli/sft.py" 2>/dev/null || true
pkill -9 -f "torch.distributed.run" 2>/dev/null || true
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG"

echo "[post-c] launching individual-eval sweep @ $(date)" | tee -a "$LOG"
tmux kill-session -t gelab-chain 2>/dev/null || true
tmux new-session -d -s gelab-chain -n eval \
  "bash -lc 'source /workspace/gelab-env/scripts/_chain_env.sh && bash /workspace/gelab-env/scripts/run_parallel_eval_sweep.sh; echo; echo \"[tmux] sweep exited\"; read -n 1'"
sleep 3
tmux ls 2>&1 | tee -a "$LOG"
echo "[post-c] done @ $(date)" | tee -a "$LOG"
