#!/usr/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# This script reads configuration from environment variables set by the launcher.

# --- Environment Check (Optional but recommended) ---
required_vars=(
    NNODES NPROC_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT # Usually set by rlaunch
    WANDB_API_KEY WANDB_PROJECT REPORT_TO
    SAVE_NAME BASE_OUTPUT_DIR BASE_LOG_DIR MODEL_PATH DATASET_PATH
    SFT_MAX_PIXELS # Specifically needed for command prefix
    TRAIN_TYPE TORCH_DTYPE NUM_TRAIN_EPOCHS PER_DEVICE_TRAIN_BATCH_SIZE
    PER_DEVICE_EVAL_BATCH_SIZE LEARNING_RATE TARGET_MODULES FREEZE_VIT
    GRADIENT_ACCUMULATION_STEPS DEEPSPEED_CONFIG LOGGING_STEPS MAX_LENGTH
    TRUNCATION_STRATEGY MAX_NEW_TOKENS WARMUP_RATIO DATALOADER_NUM_WORKERS
    SAVE_STEPS SAVE_TOTAL_LIMIT SYSTEM_PROMPT packing max_steps
)

# Check for SFT_MAX_PIXELS specifically as it's needed for the command prefix
if [ -z "$SFT_MAX_PIXELS" ]; then
    echo "Error: SFT_MAX_PIXELS environment variable is not set. Cannot construct swift command prefix."
    exit 1
fi

# --- Prepare Directories and Log File for Swift SFT Output ---
time_start_worker=$(TZ= date '+%Y-%m-%d_%H%M%S') # Use format suitable for filenames
export OUTPUT_DIR="$BASE_OUTPUT_DIR/$SAVE_NAME"
# Define a separate log file for the swift sft command output from all ranks
export LOG_DIR_WORKER="$BASE_LOG_DIR/${SAVE_NAME}_worker_${time_start_worker}.log"

# Create directories if they don't exist (especially important on rank 0)
# Also create the log directory (parent of LOG_DIR_WORKER)
if [ "$NODE_RANK" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$(dirname "$LOG_DIR_WORKER")"
    echo "Worker script started on Rank $NODE_RANK. Time: $(TZ= date '+%Y-%m-%d_%H:%M:%S_%Z')"
    echo "Node Count: $NNODES"
    echo "Node Rank: $NODE_RANK"
    echo "Master Addr: $MASTER_ADDR"
    echo "Master Port: $MASTER_PORT"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Worker Log File (for swift sft output): $LOG_DIR_WORKER"
    echo "Max Pixels used in command prefix: $SFT_MAX_PIXELS" # Log the value being used
    echo "----------------------------------------"
fi

# --- Execute the Swift SFT Command with Specific Prefix ---

echo "Starting swift sft command on Rank $NODE_RANK with prefix..."
echo "Command Prefix: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_PIXELS=$SFT_MAX_PIXELS"

# Construct and execute the command exactly as requested in the original bash2 script
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_PIXELS=$SFT_MAX_PIXELS swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --train_type "$TRAIN_TYPE" \
    --torch_dtype "$TORCH_DTYPE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --target_modules "$TARGET_MODULES" \
    --freeze_vit "$FREEZE_VIT" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --logging_steps "$LOGGING_STEPS" \
    --max_length "$MAX_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --system "$SYSTEM_PROMPT" \
    --add_version False \
    --report_to "$REPORT_TO" \
    --save_total_limit 5 \
    --attn_impl flash_attn \
    --streaming true \
    --shuffle_buffer_size 1000 \
    --packing true \
    --save_strategy epoch \
    --max_steps 1000 \
    --max_epochs 5 \
    2>&1 | tee -a "$LOG_DIR_WORKER" # Append swift sft stdout/stderr to the worker log file

# --- Completion ---
if [ "$NODE_RANK" -eq 0 ]; then
    time_end_worker=$(TZ= date '+%Y-%m-%d_%H:%M:%S_%Z')
    echo "----------------------------------------"
    echo "Worker script finished swift sft command on Rank $NODE_RANK. Time: $time_end_worker" | tee -a "$LOG_DIR_WORKER"
    echo "Worker log saved to: $LOG_DIR_WORKER"
fi