#!/usr/bin/bash
set -e

# =============================================================================
# SFT Training Script - 448x448 Environment
# Optimized for 3x A100 80GB
# =============================================================================

# Environment Variables
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

# Redirect temp files to avoid disk quota issues
export WANDB_DIR="/ext_hdd2/nhkoh/wandb"
export WANDB_CACHE_DIR="/ext_hdd2/nhkoh/.cache/wandb"
export TMPDIR="/ext_hdd2/nhkoh/tmp"
export TRITON_CACHE_DIR="/ext_hdd2/nhkoh/.cache/triton"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TMPDIR" "$TRITON_CACHE_DIR"

# Model and Data Paths
export SAVE_NAME="sft_448_retrain"
export MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export DATASET_PATH="datas/448_retrain/sft_aligned.json"
export BASE_OUTPUT_DIR="./checkpoint/gui_exp"
export BASE_LOG_DIR="./logs/train"

# =============================================================================
# Paper Table 8 - SFT Hyperparameters
# =============================================================================
export LEARNING_RATE=1e-5
export NUM_TRAIN_EPOCHS=1
export MAX_PIXELS=200704  # 448x448

# GPU Configuration for 3x A100 80GB
# Paper: 16 GPUs × batch=2 = 32 effective batch
# Ours: 3 GPUs × batch=4 × grad_acc=3 = 36 effective batch (close to paper)
export NPROC_PER_NODE=3
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export PER_DEVICE_EVAL_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=3

# Training Configuration
export TRAIN_TYPE="full"
export TORCH_DTYPE="bfloat16"
export DEEPSPEED_CONFIG="zero2"
export MAX_LENGTH=4096
export WARMUP_RATIO=0.05
export DATALOADER_NUM_WORKERS=4

# Logging
export LOGGING_STEPS=10
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=3

# Create directories
mkdir -p "$BASE_LOG_DIR"
mkdir -p "$BASE_OUTPUT_DIR"

time_start=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$BASE_OUTPUT_DIR/$SAVE_NAME/v0-${time_start}"
LOG_FILE="$BASE_LOG_DIR/sft_448_${time_start}.log"

echo "============================================================"
echo "SFT TRAINING - 448x448 Environment"
echo "============================================================"
echo "Start time: $(date)"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE x A100 80GB"
echo ""
echo "Parameters:"
echo "  Learning rate: $LEARNING_RATE"
echo "  Per-device batch: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  Grad accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Epochs: $NUM_TRAIN_EPOCHS"
echo "  Max pixels: $MAX_PIXELS"
echo ""
echo "Effective batch: $((PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))"
echo "============================================================"

# Show dataset info
python -c "import json; data=json.load(open('$DATASET_PATH')); print(f'Dataset samples: {len(data)}')"

CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=$NPROC_PER_NODE \
MAX_PIXELS=$MAX_PIXELS \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --train_type "$TRAIN_TYPE" \
    --torch_dtype "$TORCH_DTYPE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --logging_steps "$LOGGING_STEPS" \
    --max_length "$MAX_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio "$WARMUP_RATIO" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --add_version False \
    --report_to "$REPORT_TO" \
    --gradient_checkpointing true \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "SFT Training Complete"
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
