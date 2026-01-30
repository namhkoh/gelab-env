#!/usr/bin/bash
set -e

# =============================================================================
# ST-RL Training Script (Paper-Aligned, Optimized for 8x H200 143GB)
# =============================================================================
# Paper Table 8 - RL Hyperparameters:
# - learning rate: 1e-6
# - per device train batch size: 8
# - num train epochs: 5
# - num generations: 8
# - temperature: 1.2
# - top p: 1.0
# - top k: 8
# =============================================================================

# WANDB Configuration
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in your environment}"
export WANDB_ENTITY="namhokoh-korea-advanced-institute-of-science-and-technology"
export WANDB_PROJECT="gelab"
export REPORT_TO="wandb"

# Model and Data Paths
export SAVE_NAME="st_rl_448"

# Use the SFT checkpoint as starting point (paper: ST-RL builds on SFT)
export MODEL_PATH="/root/.cursor/worktrees/gelab-env__SSH__vast_/ofz/checkpoint/gui_exp/sft_448/v1-20260130-013131/v0-20260130-013206/checkpoint-956"
export DATASET_PATH="datas/448_paper/st_rl_path_only.json"

export BASE_OUTPUT_DIR="./checkpoint/gui_exp"
export BASE_LOG_DIR="./logs/train"

# =============================================================================
# Optimized Parameters for 8x H200 (143GB each = 1.1TB total VRAM)
# =============================================================================
export MAX_PIXELS=200704  # 448x448
export RLHF_TYPE="grpo"

# Reward functions (paper Section 3.3.1):
# - Action Type Reward: correct action (click/complete)
# - Coordinate Accuracy Reward: coords within bbox
# - Intent Matching Reward: icon name match
export REWARD_FUNCS="web_action_match web_coordinate_match_bbox web_intent_match"
export REWARD_WEIGHTS="0.25 0.5 0.25"  # Weight coordinate matching more heavily

export TRAIN_TYPE="full"
export TORCH_DTYPE="bfloat16"
export MAX_COMPLETION_LENGTH=512  # Outputs are short (~50 tokens)

# Paper hyperparameters (Table 8)
export NUM_TRAIN_EPOCHS=5
export LEARNING_RATE=1e-6
export TEMPERATURE=1.2
export TOP_P=1.0
export TOP_K=8
export NUM_GENERATIONS=8

# GPU Optimization - push harder with H200's 143GB
# Paper uses batch_size=8, we can increase to 16 with H200
export PER_DEVICE_TRAIN_BATCH_SIZE=16
export PER_DEVICE_EVAL_BATCH_SIZE=16
export GRADIENT_ACCUMULATION_STEPS=1
# Effective batch = 16 * 8 GPUs * 8 generations = 1024 samples per step

export DEEPSPEED_CONFIG="zero2"
export EVAL_STEPS=200
export SAVE_STEPS=200
export SAVE_TOTAL_LIMIT=5
export LOGGING_STEPS=5
export MAX_LENGTH=2048
export WARMUP_RATIO=0.05
export DATALOADER_NUM_WORKERS=8
export DATASET_NUM_PROC=8
export LOG_COMPLETIONS="true"

# System Prompt (from paper Appendix A.10)
export SYSTEM_PROMPT="You are a GUI Navigation Agent. Navigate to the target page by clicking icons.

Input format:
- Instruction: from <source> to <target>. History: <previous_steps>
- Current screen image

Output format:
Explain: click <icon_name> icon on <page>.	Action: click(start_box='<|box_start|>(x,y)<|box_end|>')
OR
Explain: this is target page.	Action: complete

Coordinates use (0,0) top-left to (1000,1000) bottom-right system."

# Resource Allocation (8 GPUs)
export NPROC_PER_NODE=8

# Create directories
mkdir -p "$BASE_LOG_DIR"

time_start=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$BASE_OUTPUT_DIR/$SAVE_NAME/v0-${time_start}"
LOG_FILE="$BASE_LOG_DIR/st_rl_448_${time_start}.log"

echo "============================================================"
echo "ST-RL TRAINING (Paper-Aligned, H200 Optimized)"
echo "============================================================"
echo "Start time: $(date)"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE x H200 (143GB each)"
echo ""
echo "Paper Parameters (Table 8):"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size: $PER_DEVICE_TRAIN_BATCH_SIZE (paper: 8)"
echo "  Epochs: $NUM_TRAIN_EPOCHS"
echo "  Num generations: $NUM_GENERATIONS"
echo "  Temperature: $TEMPERATURE"
echo ""
echo "Reward functions: $REWARD_FUNCS"
echo "Effective batch: $((PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE)) samples/step"
echo "============================================================"

# Check dataset sample count
python -c "import json; data=json.load(open('$DATASET_PATH')); print(f'Dataset samples: {len(data)}')"

# Run training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$NPROC_PER_NODE \
MAX_PIXELS=$MAX_PIXELS \
swift rlhf \
    --rlhf_type "$RLHF_TYPE" \
    --model "$MODEL_PATH" \
    --reward_funcs $REWARD_FUNCS \
    --reward_weights $REWARD_WEIGHTS \
    --train_type "$TRAIN_TYPE" \
    --torch_dtype "$TORCH_DTYPE" \
    --dataset "$DATASET_PATH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
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
    --system "$SYSTEM_PROMPT" \
    --add_version False \
    --report_to "$REPORT_TO" \
    --log_completions "$LOG_COMPLETIONS" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "ST-RL Training Complete"
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
