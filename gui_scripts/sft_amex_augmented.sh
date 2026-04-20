#!/usr/bin/bash
set -e

# =============================================================================
# SFT Training Script - AMEX Augmented Dataset (namhokaist/amex-augmented-sft)
# Base: Qwen2.5-VL-7B-Instruct. Full SFT on the augmented dataset.
# Tuned for 4x H200 (141 GB each).
#
# Dataset composition (161,091 samples):
#   - 40,285 navigation (GE-Lab click(start_box='<|box_start|>(x,y)<|box_end|>'))
#   - 60,403 grounding
#   - 60,403 understanding
# Format is GE-Lab (click), not AMEX (tap) — so we use the GE-Lab-style system prompt.
# =============================================================================

# Environment Variables (user-provided)
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in your environment}"
export WANDB_ENTITY="${WANDB_ENTITY:-namhokoh-korea-advanced-institute-of-science-and-technology}"
export WANDB_PROJECT="${WANDB_PROJECT:-gelab}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment}"
export HF_HOME="${HF_HOME:-/workspace/gelab-env/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/gelab-env/.cache}"
export TORCH_HOME="${TORCH_HOME:-/workspace/gelab-env/.cache/torch}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export USE_HF=1
export REPORT_TO="${REPORT_TO:-wandb}"

# Where the HF dataset was snapshotted (train.json + images/)
export DATASET_ROOT="${DATASET_ROOT:-/workspace/gelab-env/datas_amex/amex-augmented-sft}"
export DATASET_PATH="${DATASET_PATH:-$DATASET_ROOT/train.json}"

# Model / output
export SAVE_NAME="${SAVE_NAME:-sft_amex_augmented}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/workspace/gelab-env/checkpoint/gui_exp}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-/workspace/gelab-env/logs/train}"

# =============================================================================
# Hyperparameters
#
# Native AMEX page resolution is 1080x2400 = 2,592,000 px — grounding requires
# full res (200,704 = 448² downsamples 13× and turns ~24-px icons into ~6 px).
# Use max_pixels=2,592,000 to preserve native resolution.
#
# At 2.59M px the image contributes ~3,300 tokens — max_length must jump to
# ~5120 and deepspeed must shard memory (zero3 + grad_ckpt), mirroring the
# paper continue-train recipe.
#
# 4x H200 (141 GB) effective batch target = 32:
#   per_device=2 x grad_accum=4 (zero3+grad_ckpt) at 2.59M px fits comfortably.
#
# Learnings from continue_train doc: zero2+grad_ckpt+bf16 NaNs; use zero3.
# =============================================================================
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_PIXELS="${MAX_PIXELS:-2592000}"  # 1080x2400 native

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"

export TRAIN_TYPE="${TRAIN_TYPE:-full}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero3}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export MAX_LENGTH="${MAX_LENGTH:-5120}"  # ~3.3K image tokens + text + sys prompt
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attn}"  # requires flash-attn installed
export PACKING="${PACKING:-false}"  # enable once flash_attn verified

# Logging / checkpointing
export EVAL_STEPS="${EVAL_STEPS:-500}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
# /dev/shm is capped at 64 MB in this container -> workers must be 0 to avoid
# Bus errors on shared-memory DataLoader transfers at 2.59M px.
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"

# Wandb run name
export WANDB_NAME="${WANDB_NAME:-sft_amex_augmented_$(date '+%Y%m%d_%H%M%S')}"

# System prompt — GE-Lab-aligned (data uses `click(start_box=...)`, not AMEX `tap(...)`)
read -r -d '' SYSTEM_PROMPT << 'PROMPT_EOF' || true
You are a Multifaceted Mobile Interface Assistant. Your responsibilities include:

- 1. Navigating a mobile phone interface to reach a target page based on user instructions, task history, and the current screen state.
- 2. Understanding icons by identifying their name or function based on their location on the screen.
- 3. Grounding icons by locating the coordinates of an icon based on its name or description.

You will receive input that typically includes:

- User Request: Specifies the goal (navigation, understanding, or grounding). This might be a complex instruction for navigation or a direct question/command for icon tasks.
- Task History (Optional, primarily for Navigation): Records previous steps.
- Current Screen State: Represents the current screen, an image (indicated by <image>).

Based on the user request and the current screen state (and history if applicable), you must first determine the type of task requested and then provide the appropriate output.

Task Types and Output Formats

1. Task: Navigation
- Goal: Reach a target page step-by-step.
- Possible Actions:
  - click: Tap a specific element. Provide coordinates (x, y) relative to a (0,0) top-left and (1000,1000) bottom-right system.
  - swipe: Drag/scroll from one point to another. Provide start and end coordinates.
  - type: Enter text at a location. Provide coordinates and the text string.
  - press_enter: Submit or confirm the current input.
  - press_back: Press the system back button.
  - press_home: Press the system home button.
  - complete: Task finished, current screen is the target.
- Output Format:
Explain: [Your brief explanation]	Action: [click(start_box='<|box_start|>(x,y)<|box_end|>') or swipe(...) or type(...) or press_enter() or press_back() or press_home() or complete]

2. Task: Icon Grounding
- Goal: Identify the coordinates of a requested icon.
- Output Format:
Action: click(start_box='<|box_start|>(x,y)<|box_end|>')

3. Task: Icon Understanding
- Goal: Provide the name or function of an icon at given coordinates.
- Output Format:
[Icon Name or Description]

--- General Instructions ---

- Carefully analyze the user request to determine the task (Navigation, Grounding, Understanding).
- Analyze the current screen state thoroughly.
- For actions involving coordinates, use the (0,0) to (1000,1000) system.
- Strictly adhere to the specified output format. Use a tab character (\t) as a separator where indicated.
PROMPT_EOF
export SYSTEM_PROMPT

# Create directories
mkdir -p "$BASE_LOG_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$BASE_OUTPUT_DIR"

time_start=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$BASE_OUTPUT_DIR/$SAVE_NAME/v0-${time_start}"
LOG_FILE="$BASE_LOG_DIR/${SAVE_NAME}_${time_start}.log"

echo "============================================================"
echo "SFT TRAINING - AMEX Augmented Dataset"
echo "============================================================"
echo "Start time: $(date)"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE x H200"
echo ""
echo "Hyperparameters:"
echo "  Learning rate:     $LEARNING_RATE"
echo "  Per-device batch:  $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  Grad accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Epochs:            $NUM_TRAIN_EPOCHS"
echo "  Max length:        $MAX_LENGTH"
echo "  Max pixels:        $MAX_PIXELS"
echo "  DeepSpeed:         $DEEPSPEED_CONFIG (grad_ckpt=$GRADIENT_CHECKPOINTING)"
echo ""
echo "Effective batch: $((PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))"
echo "============================================================"

# Dataset size sanity check
python -c "import json; data=json.load(open('$DATASET_PATH')); print(f'Dataset samples: {len(data)}')"

# Run from the dataset root so relative image paths in train.json resolve.
cd "$DATASET_ROOT"

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
    --report_to "$REPORT_TO" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "SFT Training Complete"
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "Log:    $LOG_FILE"
echo "============================================================"
