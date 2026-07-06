#!/usr/bin/bash
set -e

# =============================================================================
# Continue Train Script - Real-World Continual Training (GE-Lab Section 6.2)
# SFT on 24k real-world GUI samples, adapted for 4x H200 143GB
#
# Paper: global_batch=256, lr=1e-5, max_length=5120, epochs=2, 16 GPUs
# Ours:  global_batch=256, lr=1e-5, max_length=5120, epochs=2, 4 GPUs
#
# Usage:
#   MODEL_STAGE=base    bash gui_scripts/continue_train_448.sh  # Base Qwen2.5-VL
#   MODEL_STAGE=sft     bash gui_scripts/continue_train_448.sh  # SFT checkpoint
#   MODEL_STAGE=st_rl   bash gui_scripts/continue_train_448.sh  # ST-RL checkpoint
#   MODEL_STAGE=mt_rl   bash gui_scripts/continue_train_448.sh  # MT-RL checkpoint
# =============================================================================

# Environment Variables (user-provided)
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in your environment}"
export WANDB_ENTITY="namhokoh-korea-advanced-institute-of-science-and-technology"
export WANDB_PROJECT="gelab"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment}"
export HF_HOME="${HF_HOME:-/workspace/data-vol1/huggingface_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/data-vol1/.cache}"
export TORCH_HOME="${TORCH_HOME:-/workspace/data-vol1/.cache/torch}"
export USE_HF=1
export REPORT_TO="wandb"

# =============================================================================
# Model Stage Selection
# =============================================================================
MODEL_STAGE="${MODEL_STAGE:-base}"

case "$MODEL_STAGE" in
    base)
        MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
        SAVE_NAME="continue_train_448_base"
        ;;
    sft)
        MODEL_PATH="checkpoint/gui_exp/sft_448/v0-20260221_074940/checkpoint-1275"
        SAVE_NAME="continue_train_448_sft"
        ;;
    sft_hf)
        MODEL_PATH="namhokaist/SFT_True_Final"
        SAVE_NAME="continue_train_448_sft_hf"
        ;;
    st_rl)
        MODEL_PATH="checkpoint/gui_exp/st_rl_448/v0-20260221_215851/checkpoint-3420"
        SAVE_NAME="continue_train_448_st_rl"
        ;;
    mt_rl)
        # MT_RL_CKPT overrides the default (original run's checkpoint from the old host)
        MODEL_PATH="${MT_RL_CKPT:-checkpoint/gui_exp/mt_rl_448/v0-20260225_062227/checkpoint-1300}"
        SAVE_NAME="continue_train_448_mt_rl"
        ;;
    sft_luca)
        MODEL_PATH="luca0621/sft-448"
        SAVE_NAME="continue_train_448_sft_luca"
        ;;
    sft_luca_combined)
        MODEL_PATH="luca0621/sft-448"
        SAVE_NAME="continue_train_448_sft_luca_combined"
        ;;
    base_combined)
        MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
        SAVE_NAME="continue_train_448_base_combined"
        ;;
    sft_amex_full)
        # Stage 1: SFT base Qwen on amex-gelab (3-task) — produces FullSFT-v1
        MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
        SAVE_NAME="sft_amex_gelab_full"
        ;;
    ct_amex_full)
        # Stage 3: continue-train Stage 1 SFT on real-world no-AMEX
        # MODEL_PATH must be supplied via env var (Stage 1 checkpoint or HF repo)
        MODEL_PATH="${CT_MODEL_PATH:?Set CT_MODEL_PATH to Stage 1 SFT checkpoint}"
        SAVE_NAME="ct_amex_full_no_amex"
        ;;
    *)
        echo "ERROR: MODEL_STAGE must be one of: base, base_combined, sft, sft_hf, sft_luca, sft_luca_combined, sft_amex_full, ct_amex_full, st_rl, mt_rl"
        echo "Got: $MODEL_STAGE"
        exit 1
        ;;
esac

export SAVE_NAME
export MODEL_PATH
# Allow per-stage dataset override
if [ "$MODEL_STAGE" = "sft_luca_combined" ] || [ "$MODEL_STAGE" = "base_combined" ]; then
    export DATASET_PATH="${DATASET_PATH:-datas/combined_amex_gelab_full_no_amex_ct.json}"
    export NUM_TRAIN_EPOCHS_OVERRIDE=1
elif [ "$MODEL_STAGE" = "sft_amex_full" ]; then
    export DATASET_PATH="${DATASET_PATH:-datas/sft_amex_gelab_23k.json}"
    export NUM_TRAIN_EPOCHS_OVERRIDE=1
elif [ "$MODEL_STAGE" = "ct_amex_full" ]; then
    export DATASET_PATH="${DATASET_PATH:-datas/continue_train_no_amex_15k.json}"
    export NUM_TRAIN_EPOCHS_OVERRIDE=2
else
    export DATASET_PATH="${DATASET_PATH:-datas/continue_train_24k.json}"
fi
export BASE_OUTPUT_DIR="./checkpoint/gui_exp"
export BASE_LOG_DIR="./logs/train"

# =============================================================================
# Paper Appendix A.5 - Continue Train Hyperparameters
# "identical hyperparameters, including a global batch size of 256,
#  a learning rate of 1e-5, a max length of 5120, and 2 epochs"
#
# Paper: 16 GPUs, batch=16/gpu, grad_accum=1 -> 256 effective
# Ours:  4 GPUs,  batch=2/gpu,  grad_accum=32 -> 256 effective
# =============================================================================
export LEARNING_RATE="${LEARNING_RATE_OVERRIDE:-1e-5}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS_OVERRIDE:-2}"
# Use Qwen2.5-VL-7B base preprocessor max_pixels (12,845,056). At this cap, native
# screenshot resolution (~2.6M for 1080x2400 mobile shots) is preserved.
export MAX_PIXELS="${MAX_PIXELS_OVERRIDE:-12845056}"

# GPU Adjustment for 4x H200 143GB
# At native resolution, lower per-device batch and raise grad_accum to fit VRAM
# while preserving paper effective batch=256
export NPROC_PER_NODE=4
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE_OVERRIDE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS_OVERRIDE:-16}"  # 4*4*16=256

# Training Configuration
# NOTE: zero2 + gradient_checkpointing + bf16 causes NaN (observed in SFT training).
# Use zero3 which partitions model weights and is numerically stable with grad ckpt.
export TRAIN_TYPE="full"
export TORCH_DTYPE="bfloat16"
export DEEPSPEED_CONFIG="zero3"
export GRADIENT_CHECKPOINTING="true"
export MAX_LENGTH="${MAX_LENGTH_OVERRIDE:-5120}"
export WARMUP_RATIO=0.05
export LR_SCHEDULER_TYPE="cosine"

# Logging
export EVAL_STEPS=500
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=2
export SAVE_ONLY_MODEL="true"
export LOGGING_STEPS=10
export DATALOADER_NUM_WORKERS=0
# num_proc>1 shares tensors via /dev/shm during packing; this container has 64MB
export DATASET_NUM_PROC="${DATASET_NUM_PROC_OVERRIDE:-1}"

# =============================================================================
# System Prompt - Real-World GUI (Paper Appendix A.5)
# Extended action space: click, TYPE, SCROLL, WAIT, complete
# =============================================================================
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
- Typical Input: Multi-turn instruction, history, and state. screen description and screenshot.
- Possible Actions:
  - click: Tap a specific element. Provide coordinates (x, y) relative to a (0,0) top-left and (1000,1000) bottom-right system.
  - type: Type text into an input field. Provide the string content to be typed. Example: TYPE("Texas BBQ")
  - scroll: Scroll the screen. Provide the scroll distance. Example: SCROLL(5)
  - wait: Pause the execution. Provide the duration to wait in seconds. Example: WAIT(3)
  - complete: Task finished, current screen is the target.
- Output Format:
Explain: [Your brief explanation, e.g., 'click xxx icon on yyy page.', 'this is the target page.']	Action: [click(start_box=<|box_start|>(x,y)<|box_end|>) or TYPE("Text to type") or SCROLL(5) or WAIT(3) or complete]

2. Task: Icon Grounding (Locating an Icon)

- Goal: Identify the coordinates of a requested icon.
- Typical Input: User request like "Click on [icon name/description] in the image.", screen image (<image>).
- Action: Implicitly click (meaning "identify location").
- Output Format:
Action: click(start_box=<|box_start|>(x,y)<|box_end|>)

3. Task: Icon Understanding (Identifying an Icon)

- Goal: Provide the name or function of an icon at given coordinates.
- Typical Input: User request like "What is the icon at point (x, y) in the image?", screen image (<image>).
- Action: Provide textual information.
- Output Format: Just the direct answer as text.

--- General Instructions ---

- Carefully analyze the user request to determine the task (Navigation, Grounding, Understanding).
- Analyze the current screen state (description or image) thoroughly.
- For actions involving coordinates (click), use the (0,0) to (1000,1000) system.
- Strictly adhere to the specified output format for the determined task type. Use a tab character (\t) as a separator where indicated.
PROMPT_EOF
export SYSTEM_PROMPT

# Create directories
mkdir -p "$BASE_LOG_DIR"
mkdir -p "$HF_HOME"
mkdir -p "$XDG_CACHE_HOME"

time_start=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$BASE_OUTPUT_DIR/$SAVE_NAME/v0-${time_start}"
LOG_FILE="$BASE_LOG_DIR/${SAVE_NAME}_${time_start}.log"

echo "============================================================"
echo "CONTINUE TRAIN (Paper Section 6.2 - Continual Training)"
echo "============================================================"
echo "Start time: $(date)"
echo "Model stage: $MODEL_STAGE"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE x H200 143GB"
echo ""
echo "Paper Parameters (Appendix A.5):"
echo "  Learning rate: $LEARNING_RATE"
echo "  Per-device batch: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  Grad accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Epochs: $NUM_TRAIN_EPOCHS"
echo "  Max length: $MAX_LENGTH"
echo ""
echo "Effective batch: $((PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))"
echo "============================================================"

python -c "import json; data=json.load(open('$DATASET_PATH')); print(f'Dataset samples: {len(data)}')"

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
    --ddp_timeout 7200 \
    --system "$SYSTEM_PROMPT" \
    --add_version False \
    --max_pixels "$MAX_PIXELS" \
    --packing "${PACKING_OVERRIDE:-true}" \
    --attn_impl flash_attn \
    --report_to "$REPORT_TO" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Continue Train Complete"
echo "End time: $(date)"
echo "Model stage: $MODEL_STAGE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
