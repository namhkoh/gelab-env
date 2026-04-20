#!/usr/bin/bash
# =============================================================================
# Parallel re-eval sweep: 4 concurrent workers, each pinned to one GPU,
# each running all 5 benchmarks serially for its assigned model before
# grabbing the next one from a shared queue.
#
# ~4× speedup over serial sweep. ~3 hr wall for 18 models × 5 benchmarks.
# =============================================================================
set -eo pipefail
cd /workspace/gelab-env

export HF_TOKEN="${HF_TOKEN:?}"
source /opt/conda/etc/profile.d/conda.sh
conda activate gelab

LOG_DIR="/workspace/gelab-env/logs/eval"
mkdir -p "$LOG_DIR"
STAMP=$(date '+%Y%m%d_%H%M%S')
MASTER_LOG="$LOG_DIR/sweep_${STAMP}.log"
echo "[sweep] parallel sweep starting @ $(date) — log: $MASTER_LOG" | tee -a "$MASTER_LOG"

latest_ckpt () {
  ls -td "/workspace/gelab-env/checkpoint/gui_exp/aug_$1/v0-"*/checkpoint-* 2>/dev/null | head -1
}

# Queue file — one "name|path" per line
QUEUE="$LOG_DIR/sweep_queue_${STAMP}.txt"
> "$QUEUE"
add_if_valid () {
  local name="$1" path="$2"
  if [ "$path" = "" ]; then return; fi
  if [[ "$path" != Qwen/* ]] && [ ! -f "$path/config.json" ]; then
    echo "[sweep] SKIP $name: no config.json" | tee -a "$MASTER_LOG"; return
  fi
  echo "$name|$path" >> "$QUEUE"
}

add_if_valid base_qwen                   "Qwen/Qwen2.5-VL-7B-Instruct"
for run in t1a_aug_21k t1b_orig_21k t1c_mix_21k \
           t2a_aug_nav_21k t2b_orig_nav_21k \
           r0_real_only_21k \
           r1_real_aug_8020_21k r2_real_aug_5050_21k \
           r_real_aug_9010_21k r_real_aug_8515_21k r_real_aug_7030_21k \
           s1_success_nav_orig s3_success_nav_aug_21k s2_success_nav_aug \
           c1_combined_nav_21k c2_combined_success_nav_21k; do
  add_if_valid "$run" "$(latest_ckpt $run)"
done

total=$(wc -l < "$QUEUE")
echo "[sweep] queued $total models. 4 workers will process in parallel." | tee -a "$MASTER_LOG"

LOCK="$QUEUE.lock"

# Atomically pop the first line of the queue
pop_next () {
  (
    flock -x 9
    line=$(head -n1 "$QUEUE")
    if [ -z "$line" ]; then echo ""; else
      sed -i '1d' "$QUEUE"
      echo "$line"
    fi
  ) 9>"$LOCK"
}

worker () {
  local gpu=$1
  local worker_log="$LOG_DIR/worker_${gpu}_${STAMP}.log"
  echo "[worker gpu=$gpu] start @ $(date)" >> "$worker_log"
  while true; do
    local spec=$(pop_next)
    [ -z "$spec" ] && break
    local name="${spec%|*}"
    local path="${spec#*|}"
    echo "[worker gpu=$gpu] >>> $name @ $(date)" >> "$worker_log"
    echo "[worker gpu=$gpu] >>> $name @ $(date)" >> "$MASTER_LOG"
    env RUN_NAME="$name" MODEL_PATH="$path" \
        CUDA_VISIBLE_DEVICES="$gpu" NUM_GPUS=1 \
        bash /workspace/gelab-env/scripts/eval_one_model.sh >> "$worker_log" 2>&1 \
        || echo "[worker gpu=$gpu] WARN $name non-zero" >> "$MASTER_LOG"
    echo "[worker gpu=$gpu] <<< $name done @ $(date)" >> "$MASTER_LOG"
  done
  echo "[worker gpu=$gpu] exit @ $(date)" >> "$worker_log"
}

# Fire 4 workers in parallel
worker 0 &
worker 1 &
worker 2 &
worker 3 &
wait

echo "[sweep] all workers drained @ $(date)" | tee -a "$MASTER_LOG"
python /workspace/gelab-env/scripts/aggregate_individual_evals.py || echo "[sweep] WARN aggregator failed"
echo "[sweep] COMPLETE @ $(date)" | tee -a "$MASTER_LOG"
