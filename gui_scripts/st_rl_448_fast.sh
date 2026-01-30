#!/usr/bin/bash
set -e

# =============================================================================
# ST-RL Training WITH VLLM (fixed compatibility)
# =============================================================================

export WANDB_API_KEY="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"
export WANDB_ENTITY="namhokoh-korea-advanced-institute-of-science-and-technology"
export WANDB_PROJECT="gelab"
export REPORT_TO="wandb"

export MODEL_PATH="/root/.cursor/worktrees/gelab-env__SSH__vast_/ofz/checkpoint/gui_exp/sft_448/v1-20260130-013131/v0-20260130-013206/checkpoint-956"
export DATASET_PATH="datas/448_paper/st_rl_path_only.json"
export BASE_OUTPUT_DIR="./checkpoint/gui_exp"
export BASE_LOG_DIR="./logs/train"

export MAX_PIXELS=200704
export RLHF_TYPE="grpo"
export REWARD_FUNCS="web_action_match web_coordinate_match_bbox web_intent_match"
export REWARD_WEIGHTS="0.25 0.5 0.25"
export TRAIN_TYPE="full"
export TORCH_DTYPE="bfloat16"
export MAX_COMPLETION_LENGTH=128

# Paper hyperparameters
export NUM_TRAIN_EPOCHS=5
export LEARNING_RATE=1e-6
export TEMPERATURE=1.2
export TOP_P=1.0
export TOP_K=8
export NUM_GENERATIONS=8

# GPU settings with vLLM
export PER_DEVICE_TRAIN_BATCH_SIZE=32
export PER_DEVICE_EVAL_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1

# vLLM ENABLED
export USE_VLLM="true"
export VLLM_GPU_MEMORY_UTILIZATION=0.5
export VLLM_MAX_MODEL_LEN=2048

export DEEPSPEED_CONFIG="zero2"
export EVAL_STEPS=100
export SAVE_STEPS=100
export SAVE_TOTAL_LIMIT=3
export LOGGING_STEPS=5
export MAX_LENGTH=2048
export WARMUP_RATIO=0.05
export DATALOADER_NUM_WORKERS=8
export DATASET_NUM_PROC=8
export LOG_COMPLETIONS="true"

export SYSTEM_PROMPT="You are a GUI Navigation Agent. Navigate to the target page by clicking icons.

Input: Instruction: from <source> to <target>. History: <steps>
Output: Explain: click <icon> icon on <page>.	Action: click(start_box='<|box_start|>(x,y)<|box_end|>')
OR: Explain: this is target page.	Action: complete

Coordinates: (0,0) top-left to (1000,1000) bottom-right."

export NPROC_PER_NODE=8

mkdir -p "$BASE_LOG_DIR"
time_start=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$BASE_OUTPUT_DIR/st_rl_448/v1-${time_start}"
LOG_FILE="$BASE_LOG_DIR/st_rl_448_fast_${time_start}.log"

echo "============================================================"
echo "ST-RL TRAINING WITH VLLM"
echo "============================================================"
echo "Start: $(date)"
echo "Batch: 32/GPU"
echo "vLLM: ENABLED (fixed device param for 0.14+)"
echo "============================================================"

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
    --use_vllm "$USE_VLLM" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
    --add_version False \
    --report_to "$REPORT_TO" \
    --log_completions "$LOG_COMPLETIONS" \
    2>&1 | tee "$LOG_FILE"

echo "Done: $(date)"
