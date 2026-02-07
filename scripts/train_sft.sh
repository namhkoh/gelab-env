#!/usr/bin/bash
# =============================================================================
# GE-Lab SFT Training Script
# Single-node multi-GPU training with Qwen2.5-VL-7B-Instruct
# Optimized for 3x A100 80GB GPUs
# =============================================================================

set -e

# --- Environment Setup ---
cd /ext_hdd2/nhkoh/gelab-env

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gelab

# Set cache directories
export HF_HOME="/ext_hdd2/nhkoh/.cache/huggingface"
export TRANSFORMERS_CACHE="/ext_hdd2/nhkoh/.cache/huggingface/transformers"
export TORCH_HOME="/ext_hdd2/nhkoh/.cache/torch"
export MODELSCOPE_CACHE="/ext_hdd2/nhkoh/.cache/modelscope"
export USE_HF=1  # Use HuggingFace instead of ModelScope

# --- WANDB Configuration ---
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="namhokoh-korea-advanced-institute-of-science-and-technology"
export WANDB_PROJECT="gelab"
export REPORT_TO="wandb"  # Set to "none" to disable

# --- GPU Configuration ---
export CUDA_VISIBLE_DEVICES=0,1,2
export NPROC_PER_NODE=3

# --- Model Configuration ---
export MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export DATASET_PATH="datas/sft.json"

# --- Training Configuration ---
export SAVE_NAME="sft_qwen25vl_7b"
export OUTPUT_DIR="./checkpoint/gui_exp/${SAVE_NAME}"
export LOG_DIR="./logs/train"

# Training hyperparameters (paper-aligned)
export TRAIN_TYPE="full"
export TORCH_DTYPE="bfloat16"
export NUM_TRAIN_EPOCHS=1
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=4  # Effective batch = 2 * 3 GPUs * 4 = 24
export LEARNING_RATE=1e-5
export WARMUP_RATIO=0.05
export MAX_LENGTH=4096
export MAX_PIXELS=200704  # ~448x448 for VL model

# Save settings
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=5
export LOGGING_STEPS=10

# --- System Prompt (Paper-aligned) ---
read -r -d '' SYSTEM_PROMPT << 'EOF' || true
You are a Multifaceted Mobile Interface Assistant. Your responsibilities include:
1. Navigating a mobile phone interface to reach a target page based on user instructions, task history, and the current screen state.
2. Understanding icons by identifying their name or function based on their location on the screen.
3. Grounding icons by locating the coordinates of an icon based on its name or description.

You will receive input that typically includes:
- User Request: Specifies the goal (navigation, understanding, or grounding).
- Task History (Optional): Records previous steps.
- Current Screen State: An image of the current screen.

--- Task Types and Output Formats ---

1. Task: Navigation
   - Goal: Reach a target page step-by-step.
   - Possible Actions: click, complete
   - Output Format:
     Explain: [Your brief explanation]\tAction: [click(start_box='<|box_start|>(x,y)<|box_end|>') or complete]

2. Task: Icon Grounding
   - Goal: Identify the coordinates of a requested icon.
   - Output Format:
     Action: click(start_box='<|box_start|>(x,y)<|box_end|>')

3. Task: Icon Understanding
   - Goal: Provide the name or function of an icon.
   - Output Format:
     [Icon Name or Description]

--- General Instructions ---
- Use coordinates in (0,0) to (1000,1000) system.
- Use tab character (\t) as separator where indicated.
EOF

# --- Create directories ---
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# --- Print configuration ---
echo "=============================================="
echo "GE-Lab SFT Training Configuration"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NPROC_PER_NODE GPUs)"
echo "Batch size per GPU: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective batch size: $((PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_TRAIN_EPOCHS"
echo "Max length: $MAX_LENGTH"
echo "WANDB: $WANDB_PROJECT"
echo "=============================================="

# --- Log file ---
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/sft_${SAVE_NAME}_${TIMESTAMP}.log"

echo "Starting training at $(date)"
echo "Log file: $LOG_FILE"
echo ""

# --- Run Training ---
MAX_PIXELS=$MAX_PIXELS swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --train_type "$TRAIN_TYPE" \
    --torch_dtype "$TORCH_DTYPE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_length "$MAX_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --freeze_vit true \
    --target_modules all-linear \
    --attn_impl sdpa \
    --norm_bbox norm1000 \
    --report_to "$REPORT_TO" \
    --system "$SYSTEM_PROMPT" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training completed at $(date)"
echo "Checkpoint saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"
