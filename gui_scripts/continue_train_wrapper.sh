#!/usr/bin/bash
# =============================================================================
# Continue-train wrapper. Takes an SFT checkpoint as init and trains 1 epoch
# on 21k real-world GUI mix (AITW/AITZ/AMEX/Mind2Web) at LR=1e-6 — the gentle
# recipe from continue_train.md that preserves Qwen's native grounding while
# absorbing real-world distributions.
#
# Expected env vars:
#   SFT_CHECKPOINT    absolute path to SFT checkpoint to init from
#   SAVE_NAME         output name, e.g. ct_t1a_aug_full
#   WANDB_NAME        wandb run name
#   HF_TOKEN, WANDB_API_KEY
# =============================================================================
set -eo pipefail

export HF_TOKEN="${HF_TOKEN:?}"
export WANDB_API_KEY="${WANDB_API_KEY:?}"
export WANDB_ENTITY="${WANDB_ENTITY:-namhokoh-korea-advanced-institute-of-science-and-technology}"
export WANDB_PROJECT="${WANDB_PROJECT:-gelab}"
export HF_HOME="${HF_HOME:-/workspace/gelab-env/.cache/huggingface}"
export USE_HF=1

: "${SFT_CHECKPOINT:?SFT_CHECKPOINT required}"
: "${SAVE_NAME:?SAVE_NAME required}"

export MODEL_PATH="$SFT_CHECKPOINT"
export DATASET_PATH="${DATASET_PATH:-/workspace/gelab-env/datas/continue_train_24k.json}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/workspace/gelab-env/checkpoint/gui_exp}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-/workspace/gelab-env/logs/train}"

# Paper Appendix A.5 continue-train hyperparameters, adapted from continue_train.md
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"   # gentle — preserves native grounding
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"      # ≈1000×1000, matches continue_train.md

# 4x H200: per_device=4 × grad_accum=16 × 4 GPUs = eff_batch 256 (paper target)
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

export TRAIN_TYPE="full"
export TORCH_DTYPE="bfloat16"
export DEEPSPEED_CONFIG="zero3"
export GRADIENT_CHECKPOINTING="true"
export MAX_LENGTH="${MAX_LENGTH:-5120}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
export PACKING="${PACKING:-false}"

export EVAL_STEPS=500
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=2
export SAVE_ONLY_MODEL="true"
export LOGGING_STEPS=10
export DATALOADER_NUM_WORKERS=0   # /dev/shm 64 MB limit
export DATASET_NUM_PROC=1

# Simplified continue-train system prompt (Navigation-focused, matches real-world data format)
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

mkdir -p "$BASE_LOG_DIR" "$HF_HOME"
ts=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$BASE_OUTPUT_DIR/$SAVE_NAME/v0-${ts}"
LOG_FILE="$BASE_LOG_DIR/${SAVE_NAME}_${ts}.log"

echo "============================================================"
echo "CONTINUE-TRAIN ($SAVE_NAME)"
echo "============================================================"
echo "Init from: $SFT_CHECKPOINT"
echo "Data:      $DATASET_PATH"
echo "LR:        $LEARNING_RATE (gentle — preserves native grounding)"
echo "Eff batch: $((PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))"
echo "Max px:    $MAX_PIXELS"
echo "Output:    $OUTPUT_DIR"
echo "============================================================"

source /opt/conda/etc/profile.d/conda.sh
conda activate gelab

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$NPROC_PER_NODE \
MAX_PIXELS=$MAX_PIXELS \
swift sft \
    --model "$MODEL_PATH" \
    --train_type "$TRAIN_TYPE" \
    --torch_dtype "$TORCH_DTYPE" \
    --dataset "$DATASET_PATH" \
    --max_length "$MAX_LENGTH" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --save_only_model "$SAVE_ONLY_MODEL" \
    --logging_steps "$LOGGING_STEPS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --output_dir "$OUTPUT_DIR" \
    --system "$SYSTEM_PROMPT" \
    --add_version False \
    --max_pixels "$MAX_PIXELS" \
    --attn_impl "$ATTN_IMPL" \
    --packing "$PACKING" \
    --report_to wandb \
    2>&1 | tee "$LOG_FILE"

echo "[ct] done @ $(date)"
