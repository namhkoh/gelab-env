#!/usr/bin/bash
# =============================================================================
# Runs the 5 individual benchmark eval scripts for one trained model.
# By default uses all 4 GPUs; set NUM_GPUS and CUDA_VISIBLE_DEVICES to run
# on a subset (e.g. single GPU for parallel-sweep workers).
#
# Args (env):
#   RUN_NAME                  e.g. t1a_aug_21k  OR  base_qwen
#   MODEL_PATH                absolute checkpoint dir (or HF model id for base)
#   NUM_GPUS                  default 4
#   CUDA_VISIBLE_DEVICES      default "0,1,2,3"
# =============================================================================
set -eo pipefail
cd /workspace/gelab-env

: "${RUN_NAME:?}"
: "${MODEL_PATH:?}"

source /opt/conda/etc/profile.d/conda.sh
conda activate gelab

export HF_HOME="${HF_HOME:-/workspace/gelab-env/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
CACHE="/workspace/gelab-env/.cache/huggingface/datasets"
OUT_DIR="/workspace/gelab-env/eval_results/individual/$RUN_NAME"
mkdir -p "$OUT_DIR"

run_one () {
  local name="$1"; shift
  local out="$OUT_DIR/${name}.json"
  if [ -f "$out" ] && [ "$(python -c "import json,sys;d=json.load(open('$out'));print('ok' if d.get('total',0)>0 else 'bad')" 2>/dev/null)" = "ok" ]; then
    echo "[eval-one $RUN_NAME gpu=$CUDA_VISIBLE_DEVICES] SKIP $name (cached)"
    return
  fi
  echo "[eval-one $RUN_NAME gpu=$CUDA_VISIBLE_DEVICES] === $name @ $(date +%H:%M:%S) ==="
  "$@" --output_file "$out" 2>&1 | tail -5 || echo "[eval-one] WARN $name non-zero"
}

run_one screenspot    python eval/eval_screenspot.py --model_path "$MODEL_PATH" --num_gpus "$NUM_GPUS" --cache_dir "$CACHE" --dataset rootsautomation/ScreenSpot
run_one screenspot_v2 python eval/eval_screenspot.py --model_path "$MODEL_PATH" --num_gpus "$NUM_GPUS" --cache_dir "$CACHE" --dataset HongxinLi/ScreenSpot_v2
run_one motif         python eval/eval_motif.py      --model_path "$MODEL_PATH" --num_gpus "$NUM_GPUS" --cache_dir "$CACHE"
run_one vwb_ag        python eval/eval_vwb_ag.py     --model_path "$MODEL_PATH" --num_gpus "$NUM_GPUS" --cache_dir "$CACHE"
run_one vwb_eg        python eval/eval_vwb_eg.py     --model_path "$MODEL_PATH" --num_gpus "$NUM_GPUS" --cache_dir "$CACHE"

echo "[eval-one $RUN_NAME] DONE @ $(date +%H:%M:%S)"
