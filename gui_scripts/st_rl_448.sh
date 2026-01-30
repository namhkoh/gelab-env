#!/usr/bin/bash
set -e

# =============================================================================
# ST-RL Training Script (Paper-Aligned)
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
export MODEL_PATH="checkpoint/gui_exp/sft_448/v1-20260130-013131/v0-20260130-013206/checkpoint-956"
export DATASET_PATH="datas/448_paper/st_rl_448.json"

export BASE_OUTPUT_DIR="./checkpoint/gui_exp"
export BASE_LOG_DIR="./logs/train"

# Paper RL Hyperparameters (Table 8)
export MAX_PIXELS=200704  # 448x448
export RLHF_TYPE="grpo"
export REWARD_FUNCS="web_action_match web_coordinate_match web_intent_match"
export TRAIN_TYPE="full"
export TARGET_MODULES="all-linear"
export TORCH_DTYPE="bfloat16"
export MAX_COMPLETION_LENGTH=1024
export NUM_TRAIN_EPOCHS=5  # Paper: 5
export PER_DEVICE_TRAIN_BATCH_SIZE=8  # Paper: 8
export PER_DEVICE_EVAL_BATCH_SIZE=8
export LEARNING_RATE=1e-6  # Paper: 1e-6
export GRADIENT_ACCUMULATION_STEPS=1  # With 8 GPUs, effective batch = 64
export DEEPSPEED_CONFIG="zero3"
export EVAL_STEPS=500
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=5
export LOGGING_STEPS=10
export MAX_LENGTH=4096
export WARMUP_RATIO=0.05  # From SFT settings
export DATALOADER_NUM_WORKERS=4
export DATASET_NUM_PROC=4
export NUM_GENERATIONS=8  # Paper: 8
export TEMPERATURE=1.2  # Paper: 1.2
export TOP_P=1.0  # Paper: 1.0
export TOP_K=8  # Paper: 8
export LOG_COMPLETIONS="true"

# System Prompt (from paper Appendix A.10)
export SYSTEM_PROMPT="You are a Multifaceted Mobile Interface Assistant. Your responsibilities include:
1. Navigating a mobile phone interface to reach a target page based on user instructions, task history, and the current screen state.
2. Understanding icons by identifying their name or function based on their location on the screen.
3. Grounding icons by locating the coordinates of an icon based on its name or description.

You will receive input that typically includes:
- User Request: Specifies the goal (navigation, understanding, or grounding).
- Task History (Optional, primarily for Navigation): Records previous steps.
- Current Screen State: Represents the current screen, an image (indicated by <image>).

--- Task Types and Output Formats ---

1. Task: Navigation
   - Goal: Reach a target page step-by-step.
   - Possible Actions:
     - click: Tap a specific element. Provide coordinates (x, y) relative to a (0,0) top-left and (1000,1000) bottom-right system.
     - complete: Task finished, current screen is the target.
   - Output Format:
     Explain: [Your brief explanation]\tAction: [click(start_box='<|box_start|>(x,y)<|box_end|>') or complete]

--- General Instructions ---
- Analyze the current screen state thoroughly.
- For actions involving coordinates (click), use the (0,0) to (1000,1000) system.
- Strictly adhere to the specified output format."

# Resource Allocation (8 GPUs)
export NPROC_PER_NODE=8

# Create directories
mkdir -p "$BASE_LOG_DIR"

time_start=$(date '+%Y-%m-%d_%H%M%S')
OUTPUT_DIR="$BASE_OUTPUT_DIR/$SAVE_NAME/v0-${time_start}"
LOG_FILE="$BASE_LOG_DIR/st_rl_448_${time_start}.log"

echo "============================================================"
echo "ST-RL TRAINING (Paper-Aligned)"
echo "============================================================"
echo "Start time: $(date)"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE"
echo ""
echo "Paper Parameters (Table 8):"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  Epochs: $NUM_TRAIN_EPOCHS"
echo "  Num generations: $NUM_GENERATIONS"
echo "  Temperature: $TEMPERATURE"
echo "============================================================"

# Run training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=$NPROC_PER_NODE MAX_PIXELS=$MAX_PIXELS \
swift rlhf \
    --rlhf_type "$RLHF_TYPE" \
    --model "$MODEL_PATH" \
    --reward_funcs $REWARD_FUNCS \
    --train_type "$TRAIN_TYPE" \
    --target_modules "$TARGET_MODULES" \
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
    --system "$SYSTEM_PROMPT" \
    --add_version False \
    --report_to "$REPORT_TO" \
    --log_completions "$LOG_COMPLETIONS" \
    --use_vllm false \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "ST-RL Training Complete"
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
