#!/usr/bin/bash
set -e

# =============================================================================
# SFT Training Script - Paper-Aligned (GE-Lab Table 8)
# Adjusted for 3x A100 80GB
# =============================================================================

# Environment Variables (user-provided)
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in your environment}"
export WANDB_ENTITY="namhokoh-korea-advanced-institute-of-science-and-technology"
export WANDB_PROJECT="gelab"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment}"
export HF_HOME="/ext_hdd2/nhkoh/.cache/huggingface"
export XDG_CACHE_HOME="/ext_hdd2/nhkoh/.cache"
export TORCH_HOME="/ext_hdd2/nhkoh/.cache/torch"
export CUDA_HOME=/ext_hdd2/nhkoh/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
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
export NPROC_PER_NODE=3
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
export SYSTEM_PROMPT="You are a GUI Navigation Agent. Navigate to the target page by clicking icons.

Input format:
- Instruction: from <source> to <target>. History: <previous_steps>
- Current screen image

Output format:
Explain: click <icon_name> icon on <page>.	Action: click(start_box='<|box_start|>(x,y)<|box_end|>')
OR
Explain: this is target page.	Action: complete

Coordinates use (0,0) top-left to (1000,1000) bottom-right system."

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
CUDA_VISIBLE_DEVICES=0,1,2 \
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
