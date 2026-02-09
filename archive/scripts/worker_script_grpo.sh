#!/usr/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# This script reads configuration from environment variables set by the launcher.

# --- Environment Check (Optional but recommended) ---
required_vars=(
    NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT # Usually set by rlaunch
    WANDB_API_KEY WANDB_PROJECT REPORT_TO
    SAVE_NAME BASE_OUTPUT_DIR BASE_LOG_DIR MODEL_PATH DATASET_PATH
    RLHF_TYPE REWARD_FUNCS TRAIN_TYPE TARGET_MODULES TORCH_DTYPE
    MAX_COMPLETION_LENGTH NUM_TRAIN_EPOCHS PER_DEVICE_TRAIN_BATCH_SIZE
    PER_DEVICE_EVAL_BATCH_SIZE LEARNING_RATE GRADIENT_ACCUMULATION_STEPS
    DEEPSPEED_CONFIG EVAL_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT LOGGING_STEPS
    MAX_LENGTH WARMUP_RATIO DATALOADER_NUM_WORKERS DATASET_NUM_PROC
    NUM_GENERATIONS TEMPERATURE LOG_COMPLETIONS SYSTEM_PROMPT MAX_PIXELS # Make sure MAX_PIXELS is passed
)

# Check for NPROC_PER_NODE specifically as it's needed for the command prefix
if [ -z "$NPROC_PER_NODE" ]; then
    echo "Error: NPROC_PER_NODE environment variable is not set. Cannot construct swift command prefix."
    exit 1
fi
# Check for MAX_PIXELS specifically as it's needed for the command prefix
if [ -z "$MAX_PIXELS" ]; then
    echo "Error: MAX_PIXELS environment variable is not set. Cannot construct swift command prefix."
    exit 1
fi


# --- Prepare Directories and Log File ---
time_start=$(TZ= date '+%Y-%m-%d_%H%M%S') # Use format suitable for filenames
export OUTPUT_DIR="$BASE_OUTPUT_DIR/$SAVE_NAME"
export LOG_DIR="$BASE_LOG_DIR/gui_exp_grpo_launch_${SAVE_NAME}-${time_start}.log"

# Create directories if they don't exist (especially important on rank 0)
if [ "$NODE_RANK" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$(dirname "$LOG_DIR")"
    echo "Worker script started on Rank $NODE_RANK. Time: $(TZ= date '+%Y-%m-%d_%H:%M:%S_%Z')"
    echo "Node Count: $NNODES"
    echo "Node Rank: $NODE_RANK"
    echo "Master Addr: $MASTER_ADDR"
    echo "Master Port: $MASTER_PORT"
    echo "Processes per Node used in command: $NPROC_PER_NODE" # Log the value being used
    echo "Max Pixels used in command: $MAX_PIXELS"           # Log the value being used
    echo "Output Directory: $OUTPUT_DIR"
    echo "Log File: $LOG_DIR"
    echo "----------------------------------------"
fi

# --- Execute the Swift RLHF Command with Specific Prefix ---

echo "Starting swift rlhf command on Rank $NODE_RANK with prefix..."
echo "Command Prefix: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=$NPROC_PER_NODE MAX_PIXELS=$MAX_PIXELS"

# Construct and execute the command exactly as requested
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=$NPROC_PER_NODE MAX_PIXELS=$MAX_PIXELS swift rlhf \
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
    --use_vllm true \
    --vllm_max_model_len 2048 \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_device auto \
    --num_infer_workers 8 \
    2>&1 | tee -a "$LOG_DIR" # Append swift sft stdout/stderr to the worker log file

# --- Completion ---
if [ "$NODE_RANK" -eq 0 ]; then
    time_end=$(TZ= date '+%Y-%m-%d_%H:%M:%S_%Z')
    echo "----------------------------------------"
    echo "Worker script finished on Rank $NODE_RANK. Time: $time_end" | tee -a "$LOG_DIR"
fi