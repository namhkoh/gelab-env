#!/usr/bin/bash
set -e

# =============================================================================
# ST-RL Training Script - Paper-Aligned (GE-Lab Table 8)
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
export SAVE_NAME="st_rl_448"
export MODEL_PATH="checkpoint/gui_exp/sft_448/v0-20260209_210621/checkpoint-1275"
export DATASET_PATH="datas/st_rl_path_only.json"
export BASE_OUTPUT_DIR="./checkpoint/gui_exp"
export BASE_LOG_DIR="./logs/train"

# =============================================================================
# Paper Table 8 - RL Hyperparameters (EXACT)
# =============================================================================
export LEARNING_RATE=1e-6
export NUM_TRAIN_EPOCHS=3
export NUM_GENERATIONS=8
export TEMPERATURE=1.2
export TOP_P=1.0
export TOP_K=8
export MAX_PIXELS=200704

# GPU Adjustment for 3x A100 80GB
# Paper: 16 GPUs × batch=8 × num_gen=8 = 128 effective batch
# Ours: 3 GPUs × batch=16 × grad_acc=1 = 48 effective batch
# num_gen=8 divides effective batch (3×16=48), matches paper exactly
export NPROC_PER_NODE=3
export PER_DEVICE_TRAIN_BATCH_SIZE=16
export GRADIENT_ACCUMULATION_STEPS=1

# GRPO Configuration
export RLHF_TYPE="grpo"
export TRAIN_TYPE="full"
export TORCH_DTYPE="bfloat16"
export DEEPSPEED_CONFIG="zero3"
export GRADIENT_CHECKPOINTING="true"
export MAX_COMPLETION_LENGTH=512
export MAX_LENGTH=2048
export WARMUP_RATIO=0.05

# Reward Functions (Paper Section 3.3.1 - 4 equally-weighted components)
# r_type + r_coord + r_intent + r_format
export REWARD_FUNCS="web_action_match web_coordinate_match_bbox web_intent_match format_constraint"
export REWARD_WEIGHTS="0.25 0.25 0.25 0.25"

# Logging
export EVAL_STEPS=500
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=2
export SAVE_ONLY_MODEL="true"
export LOGGING_STEPS=5
export DATALOADER_NUM_WORKERS=4
export DATASET_NUM_PROC=4
export LOG_COMPLETIONS="true"

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
LOG_FILE="$BASE_LOG_DIR/st_rl_448_${time_start}.log"

echo "============================================================"
echo "ST-RL TRAINING (Paper Table 8 Aligned)"
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
echo "  Num generations: $NUM_GENERATIONS"
echo "  Temperature: $TEMPERATURE"
echo ""
echo "Effective batch: $((PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))"
echo "============================================================"

python -c "import json; data=json.load(open('$DATASET_PATH')); print(f'Dataset samples: {len(data)}')"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2 \
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
    ${MAX_STEPS:+--max_steps $MAX_STEPS} \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --save_only_model "$SAVE_ONLY_MODEL" \
    --logging_steps "$LOGGING_STEPS" \
    --max_length "$MAX_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio "$WARMUP_RATIO" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --per_device_eval_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
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
