#!/usr/bin/bash
set -e

# =============================================================================
# SFT Training Script - Paper-Aligned (GE-Lab Table 8)
# Adjusted for 3x A100 80GB
# =============================================================================

# Environment Variables (user-provided)
export WANDB_API_KEY="${WANDB_API_KEY:}"
export WANDB_PROJECT="gelab"
export HF_TOKEN="${HF_TOKEN:}"
export HF_HOME="${HF_HOME:-/home/irteam/data-vol1/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/irteam/data-vol1/.cache}"
export TORCH_HOME="${TORCH_HOME:-/home/irteam/data-vol1/.cache/torch}"
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export USE_HF=1
export REPORT_TO="wandb"

# Model and Data Paths
export SAVE_NAME="sft_448"
export MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export DATASET_PATH="datas/sft_aligned.json"
export BASE_OUTPUT_DIR="./checkpoint/gui_exp"
export BASE_LOG_DIR="./logs/train"

# =============================================================================
# Paper Table 8 - SFT Hyperparameters
# Paper: lr=1e-5, batch=2, grad_accum=2, epochs=1, 8 GPUs → effective 32
# Ours:  lr=5e-6, batch=2, grad_accum=4, epochs=1, 3 GPUs → effective 24
# Reduced from 1e-5: NaN with gradient_checkpointing + ZeRO-2 + bf16
# =============================================================================
export LEARNING_RATE=1e-5
export NUM_TRAIN_EPOCHS=1
export MAX_PIXELS=200704

# GPU Adjustment for 3x A100 80GB
export NPROC_PER_NODE=4
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=4

# Training Configuration
export TRAIN_TYPE="full"
export TORCH_DTYPE="bfloat16"
export DEEPSPEED_CONFIG="zero2"
export GRADIENT_CHECKPOINTING="false"
export MAX_LENGTH=2048
export WARMUP_RATIO=0.05
export LR_SCHEDULER_TYPE="cosine"

# Logging
export EVAL_STEPS=500
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=2
export SAVE_ONLY_MODEL="true"
export LOGGING_STEPS=10
export DATALOADER_NUM_WORKERS=4
export DATASET_NUM_PROC=4

# System Prompt (Paper Appendix A.10)
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

--- Task Types and Output Formats ---

1. Task: Navigation

- Goal: Reach a target page step-by-step.
- Typical Input: Multi-turn instruction, history, and state. screen description and screenshot.
- Possible Actions:
  - click: Tap a specific element. Provide coordinates (x, y) relative to a (0,0) top-left and (1000,1000) bottom-right system.
  - complete: Task finished, current screen is the target.
- Output Format:
Explain: [Your brief explanation, e.g., 'click xxx icon on yyy page.', 'this is the target page.']	Action: [click(start_box=<|box_start|>(x,y)<|box_end|>) or complete]

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
- Output Format:
[Icon Name or Description]

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
LOG_FILE="$BASE_LOG_DIR/sft_448_${time_start}.log"

echo "============================================================"
echo "SFT TRAINING (Paper Table 8 Aligned)"
echo "============================================================"
echo "Start time: $(date)"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE x A100 80GB"
echo ""
echo "Paper Parameters (Table 8):"
echo "  Learning rate: $LEARNING_RATE"
echo "  Per-device batch: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  Grad accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Epochs: $NUM_TRAIN_EPOCHS"
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
    --system "$SYSTEM_PROMPT" \
    --add_version False \
    --max_pixels "$MAX_PIXELS" \
    --report_to "$REPORT_TO" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "SFT Training Complete"
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
