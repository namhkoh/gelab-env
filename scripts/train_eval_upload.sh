#!/usr/bin/bash
# =============================================================================
# Path B wrapper: gentle fine-tune (full OR LoRA) from base Qwen on the T*
# dataset -> eval on 6 grounding benchmarks -> upload adapter/checkpoint to HF
# -> append row + details to results.md.
#
# Required env (caller provides):
#   RUN_NAME         e.g. t1a_aug_21k OR lora_mix_21k_lr5e5
#   DATASET_PATH     absolute path to train JSON
#   HF_TOKEN, WANDB_API_KEY
# Optional env:
#   TRAIN_TYPE       "full" (default) or "lora"
#   LEARNING_RATE    default 1e-6 (full) or 5e-5 (lora)
#   LORA_RANK        default 16
#   LORA_ALPHA       default 32
#   NUM_TRAIN_EPOCHS default 1
#   SKIP_TRAIN=1 / SKIP_EVAL=1 / SKIP_UPLOAD=1
# =============================================================================
set -eo pipefail
cd /workspace/gelab-env

latest_ckpt() {
  local pat="$1" first
  first=$(ls -td $pat 2>/dev/null | head -1 || true)
  echo "$first"
}

start_swift_watchdog() {
  local name_pat="$1"
  (
    exec >/dev/null 2>&1
    local log=""
    for i in $(seq 1 180); do
      log=$(ls -t /workspace/gelab-env/logs/train/${name_pat}_*.log 2>/dev/null | head -1 || true)
      [ -n "$log" ] && break
      sleep 10
    done
    [ -z "$log" ] && exit 0
    while true; do
      if grep -q "train_runtime" "$log" 2>/dev/null; then
        sleep 90
        pkill -9 -f "swift/cli/sft.py" 2>/dev/null || true
        pkill -9 -f "torch.distributed.run" 2>/dev/null || true
        exit 0
      fi
      sleep 60
    done
  ) &
  echo $!
}

: "${RUN_NAME:?RUN_NAME required}"
: "${DATASET_PATH:?DATASET_PATH required}"
: "${HF_TOKEN:?HF_TOKEN required}"
: "${WANDB_API_KEY:?WANDB_API_KEY required}"

export HF_TOKEN WANDB_API_KEY
export WANDB_ENTITY="${WANDB_ENTITY:-namhokoh-korea-advanced-institute-of-science-and-technology}"
export WANDB_PROJECT="${WANDB_PROJECT:-gelab}"
export HF_HOME="${HF_HOME:-/workspace/gelab-env/.cache/huggingface}"
export USE_HF=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

HF_REPO_PREFIX="${HF_REPO_PREFIX:-namhokaist/qwen25vl-7b-augexp-}"
HF_REPO="${HF_REPO_PREFIX}${RUN_NAME}"
SAVE_NAME_FULL="aug_${RUN_NAME}"

RESULTS_DIR="/workspace/gelab-env/eval_results"
mkdir -p "$RESULTS_DIR"
RESULTS_MD="/workspace/gelab-env/results.md"
BENCH_CACHE="/workspace/gelab-env/.cache/huggingface/datasets"
TS=$(date '+%Y%m%d_%H%M%S')

source /opt/conda/etc/profile.d/conda.sh
conda activate gelab

# -------- Training recipe --------
export MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export TRAIN_TYPE="${TRAIN_TYPE:-full}"

if [ "$TRAIN_TYPE" = "lora" ]; then
  export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
  export LORA_RANK="${LORA_RANK:-16}"
  export LORA_ALPHA="${LORA_ALPHA:-32}"
  # zero2 shards optimizer state; grad_ckpt=false avoids DDP reentrant-backward
  # conflict with LoRA ("mark a variable ready only once" error).
  export DEEPSPEED_CONFIG="zero2"
  export GRADIENT_CHECKPOINTING="false"
else
  export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
  export DEEPSPEED_CONFIG="zero3"
  export GRADIENT_CHECKPOINTING="true"
fi

export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_PIXELS="1003520"
export MAX_LENGTH="5120"
export NPROC_PER_NODE="4"
export PER_DEVICE_TRAIN_BATCH_SIZE="4"
export GRADIENT_ACCUMULATION_STEPS="16"
export TORCH_DTYPE="bfloat16"
export WARMUP_RATIO="0.05"
export LR_SCHEDULER_TYPE="cosine"
export ATTN_IMPL="flash_attn"
export PACKING="false"
export EVAL_STEPS=500
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=2
export SAVE_ONLY_MODEL="true"
export LOGGING_STEPS=10
export DATALOADER_NUM_WORKERS=0
export DATASET_NUM_PROC=1

read -r -d '' SYSTEM_PROMPT << 'PROMPT_EOF' || true
You are a Multifaceted Mobile Interface Assistant. Your responsibilities include:

- 1. Navigating a mobile phone interface to reach a target page based on user instructions, task history, and the current screen state.
- 2. Understanding icons by identifying their name or function based on their location on the screen.
- 3. Grounding icons by locating the coordinates of an icon based on its name or description.

Task Types and Output Formats

General GUI Task
- Goal: Reach a target page step-by-step.
- Possible Actions:
  - click: Tap a specific element. Provide coordinates (x, y) relative to a (0,0) top-left and (1000,1000) bottom-right system.
  - type: Type text into an input field. Example: TYPE("Texas BBQ")
  - scroll: Scroll the screen. Example: SCROLL(5)
  - wait: Pause. Example: WAIT(3)
  - complete: Task finished, current screen is the target.
- Output Format:
Explain: [brief explanation]	Action: [click(start_box=<|box_start|>(x,y)<|box_end|>) or TYPE("...") or SCROLL(N) or WAIT(N) or complete]

General Instructions
- For actions involving coordinates (click), use the (0,0) to (1000,1000) system.
- Strictly adhere to the specified output format. Use a tab character (\t) between Explain and Action.
PROMPT_EOF
export SYSTEM_PROMPT

# Resolve data root for relative image paths
if python -c "import json,os; d=json.load(open('$DATASET_PATH')); p=d[0]['images'][0]; exit(0 if os.path.isabs(p) else 1)" 2>/dev/null; then
  CWD_DIR="/workspace/gelab-env"
else
  CWD_DIR="/workspace/gelab-env/datas_amex/amex-augmented-sft"
fi

# -------- Stage 1: Train --------
if [ "${SKIP_TRAIN:-0}" = "0" ]; then
  CKPT_EXISTING=$(latest_ckpt "/workspace/gelab-env/checkpoint/gui_exp/$SAVE_NAME_FULL/v0-*/checkpoint-*")
  if [ -n "$CKPT_EXISTING" ] && [ -f "$CKPT_EXISTING/config.json" -o -f "$CKPT_EXISTING/adapter_config.json" ]; then
    echo "[wrapper] checkpoint exists: $CKPT_EXISTING -> auto SKIP_TRAIN"
    SKIP_TRAIN=1
  fi
fi

if [ "${SKIP_TRAIN:-0}" = "0" ]; then
  echo "[wrapper] === TRAIN ($TRAIN_TYPE, lr=$LEARNING_RATE) $RUN_NAME @ $(date) ==="
  WD_PID=$(start_swift_watchdog "$SAVE_NAME_FULL")
  mkdir -p /workspace/gelab-env/logs/train
  OUT="/workspace/gelab-env/checkpoint/gui_exp/$SAVE_NAME_FULL/v0-${TS}"
  LOG="/workspace/gelab-env/logs/train/${SAVE_NAME_FULL}_${TS}.log"
  export WANDB_NAME="${RUN_NAME}_${TRAIN_TYPE}_${TS}"

  # Build arg list conditionally
  SFT_ARGS=(
    --model "$MODEL_PATH"
    --train_type "$TRAIN_TYPE"
    --torch_dtype "$TORCH_DTYPE"
    --dataset "$DATASET_PATH"
    --max_length "$MAX_LENGTH"
    --learning_rate "$LEARNING_RATE"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --per_device_eval_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
    --eval_steps "$EVAL_STEPS"
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --save_only_model "$SAVE_ONLY_MODEL"
    --logging_steps "$LOGGING_STEPS"
    --warmup_ratio "$WARMUP_RATIO"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --dataset_num_proc "$DATASET_NUM_PROC"
    --output_dir "$OUT"
    --system "$SYSTEM_PROMPT"
    --add_version False
    --max_pixels "$MAX_PIXELS"
    --attn_impl "$ATTN_IMPL"
    --packing "$PACKING"
    --report_to wandb
  )
  [ -n "$DEEPSPEED_CONFIG" ] && SFT_ARGS+=(--deepspeed "$DEEPSPEED_CONFIG")
  if [ "$TRAIN_TYPE" = "lora" ]; then
    SFT_ARGS+=(
      --lora_rank "$LORA_RANK"
      --lora_alpha "$LORA_ALPHA"
      --target_modules all-linear
    )
  fi

  (
    cd "$CWD_DIR"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=$NPROC_PER_NODE \
    MAX_PIXELS=$MAX_PIXELS \
    swift sft "${SFT_ARGS[@]}"
  ) 2>&1 | tee "$LOG" \
    || echo "[wrapper] WARN: train exited non-zero (watchdog may have killed NCCL hang)"
  kill "$WD_PID" 2>/dev/null || true
  pkill -9 -f "swift/cli/sft.py" 2>/dev/null || true
  pkill -9 -f "torch.distributed.run" 2>/dev/null || true
  sleep 5
fi

CKPT=$(latest_ckpt "/workspace/gelab-env/checkpoint/gui_exp/$SAVE_NAME_FULL/v0-*/checkpoint-*")
if [ -z "$CKPT" ]; then
  echo "[wrapper] ERROR: no checkpoint produced"; exit 3
fi
echo "[wrapper] checkpoint: $CKPT"
N_SAMPLES=$(python -c "import json; print(len(json.load(open('$DATASET_PATH'))))" 2>/dev/null || echo 0)

# -------- Stage 2: Eval --------
EVAL_JSON="$RESULTS_DIR/results_${RUN_NAME}.json"
if [ "${SKIP_EVAL:-0}" = "0" ]; then
  echo "[wrapper] === EVAL $RUN_NAME @ $(date) ==="
  EVAL_EXTRA=()
  if [ "$TRAIN_TYPE" = "lora" ]; then
    # LoRA: load base + adapter. `--lora_path` flag in eval script takes the adapter dir.
    EVAL_EXTRA=(--model_path "$MODEL_PATH" --lora_path "$CKPT")
  else
    EVAL_EXTRA=(--model_path "$CKPT")
  fi
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python /workspace/gelab-env/eval/evaluate_real_world.py \
    "${EVAL_EXTRA[@]}" \
    --base_model \
    --benchmarks screenspot screenspot_v2 motif refexp vwb_eg vwb_ag \
    --num_gpus 4 \
    --cache_dir "$BENCH_CACHE" \
    --output_file "$EVAL_JSON" \
    || echo "[wrapper] WARN: eval exited non-zero; continuing"
fi

# -------- Stage 3: Upload --------
if [ "${SKIP_UPLOAD:-0}" = "0" ]; then
  echo "[wrapper] === UPLOAD $RUN_NAME -> $HF_REPO @ $(date) ==="
  COMMIT_MSG="Path B ${TRAIN_TYPE} FT (lr=$LEARNING_RATE) on ${RUN_NAME}"
  python /workspace/gelab-env/scripts/hf_upload_checkpoint.py \
    --checkpoint "$CKPT" \
    --repo "$HF_REPO" \
    --commit-message "$COMMIT_MSG" \
    || echo "[wrapper] WARN: upload failed; continuing"
fi

# -------- Stage 4: Append --------
if [ -f "$EVAL_JSON" ]; then
  echo "[wrapper] === APPEND results.md ==="
  NOTE="Path B: $TRAIN_TYPE FT, LR=$LEARNING_RATE, 1 ep, max_pixels=1M, eff_batch=256"
  [ "$TRAIN_TYPE" = "lora" ] && NOTE="$NOTE, rank=$LORA_RANK alpha=$LORA_ALPHA"
  python /workspace/gelab-env/scripts/append_result.py \
    --run "$RUN_NAME" \
    --eval "$EVAL_JSON" \
    --checkpoint "$CKPT" \
    --hf-repo "$HF_REPO" \
    --samples "$N_SAMPLES" \
    --notes "$NOTE" \
    --results-md "$RESULTS_MD" \
    || echo "[wrapper] WARN: results.md update failed"
fi

echo "[wrapper] === DONE $RUN_NAME @ $(date) ==="
