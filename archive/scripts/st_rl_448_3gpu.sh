#!/usr/bin/bash
set -e

# =============================================================================
# ST-RL Training Script - 448x448 Environment
# Optimized for 3x A100 80GB
# Run this AFTER SFT training completes
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

# =============================================================================
# SFT Checkpoint Path
# =============================================================================
export MODEL_PATH="./checkpoint/gui_exp/sft_448_retrain/v0-20260201_054616/checkpoint-850"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Model and Data Paths
export SAVE_NAME="st_rl_448_balanced"
export DATASET_PATH="datas/448_retrain/st_rl_balanced.json"
export BASE_OUTPUT_DIR="./checkpoint/gui_exp"
export BASE_LOG_DIR="./logs/train"

# =============================================================================
# Paper Table 8 - RL Hyperparameters
# =============================================================================
# Note: Table 8 specifies learning_rate=1e-6 for RL
export LEARNING_RATE=1e-6
# Paper Table 8: num_train_epochs=5, num_generations=8
# Note: With 3 GPUs, only num_generations=3 is valid (must equal effective batch size)
export NUM_TRAIN_EPOCHS=5
export NUM_GENERATIONS=3
export TEMPERATURE=1.2
export TOP_P=1.0
export TOP_K=8
export MAX_PIXELS=200704

# GPU Configuration for 3x A100 80GB
# Using LoRA for faster training
# Maximizing batch size for 80GB VRAM
export NPROC_PER_NODE=3
export PER_DEVICE_TRAIN_BATCH_SIZE=96
export GRADIENT_ACCUMULATION_STEPS=1

# GRPO Configuration
export RLHF_TYPE="grpo"
export TRAIN_TYPE="lora"
export LORA_RANK=64
export LORA_ALPHA=128
export TORCH_DTYPE="bfloat16"
export DEEPSPEED_CONFIG="zero2"
export MAX_COMPLETION_LENGTH=512
export MAX_LENGTH=2048
export WARMUP_RATIO=0.05

# Reward Functions (Paper Section 3.3.1)
export REWARD_FUNCS="web_action_match web_coordinate_match_bbox web_intent_match format_constraint"
export REWARD_WEIGHTS="0.25 0.25 0.25 0.25"

# Logging
export EVAL_STEPS=100
export SAVE_STEPS=100
export SAVE_TOTAL_LIMIT=3
export LOGGING_STEPS=5
export DATALOADER_NUM_WORKERS=4
export DATASET_NUM_PROC=4
export LOG_COMPLETIONS="false"

# Create directories
mkdir -p "$BASE_LOG_DIR"
mkdir -p "$BASE_OUTPUT_DIR"

time_start=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$BASE_OUTPUT_DIR/$SAVE_NAME/v0-${time_start}"
LOG_FILE="$BASE_LOG_DIR/st_rl_448_${time_start}.log"

echo "============================================================"
echo "ST-RL TRAINING - 448x448 Environment"
echo "============================================================"
echo "Start time: $(date)"
echo "Model (SFT checkpoint): $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE x A100 80GB"
echo ""
echo "Parameters:"
echo "  Learning rate: $LEARNING_RATE"
echo "  Per-device batch: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  Grad accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Epochs: $NUM_TRAIN_EPOCHS"
echo "  Num generations: $NUM_GENERATIONS"
echo "  Train type: $TRAIN_TYPE (LoRA rank=$LORA_RANK)"
echo ""
echo "Effective batch: $((PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))"
echo "============================================================"

# Show dataset info
python -c "import json; data=json.load(open('$DATASET_PATH')); print(f'Dataset samples: {len(data)}')"

CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=$NPROC_PER_NODE \
MAX_PIXELS=$MAX_PIXELS \
swift rlhf \
    --rlhf_type "$RLHF_TYPE" \
    --model "$MODEL_PATH" \
    --reward_funcs $REWARD_FUNCS \
    --reward_weights $REWARD_WEIGHTS \
    --train_type "$TRAIN_TYPE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --torch_dtype "$TORCH_DTYPE" \
    --dataset "$DATASET_PATH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps "$LOGGING_STEPS" \
    --max_length "$MAX_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio "$WARMUP_RATIO" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --num_generations "$NUM_GENERATIONS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --add_version False \
    --report_to "$REPORT_TO" \
    --log_completions "$LOG_COMPLETIONS" \
    --gradient_checkpointing true \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "ST-RL Training Complete"
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
