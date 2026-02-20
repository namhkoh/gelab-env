#!/usr/bin/bash
set -e

# =============================================================================
# MT-RL Training Script - Paper-Aligned (GE-Lab Table 8)
# Multi-Turn RL with trajectory-level rewards
# =============================================================================

# Environment Variables
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in your environment}"
export WANDB_ENTITY="namhokoh-korea-advanced-institute-of-science-and-technology"
export WANDB_PROJECT="gelab"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment}"
export HF_HOME="/ext_hdd2/nhkoh/.cache/huggingface"
export XDG_CACHE_HOME="/ext_hdd2/nhkoh/.cache"
export TORCH_HOME="/ext_hdd2/nhkoh/.cache/torch"
export TRANSFORMERS_CACHE="/ext_hdd2/nhkoh/.cache/huggingface/transformers"
export CUDA_HOME=/ext_hdd2/nhkoh/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export USE_HF=1
export REPORT_TO="wandb"

# NCCL robustness settings
export NCCL_TIMEOUT=3600000  # 60 min (default 30 min), survive transient stalls
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200  # 2 hour heartbeat
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000  # Enable flight recorder for debugging

# Model and Data Paths
export SAVE_NAME="mt_rl_448"
export MODEL_PATH="./checkpoint/gui_exp/st_rl_448/v0-20260210_095510/checkpoint-12315"  # ST-RL checkpoint (pipeline: SFT -> ST-RL -> MT-RL)
export DATASET_PATH="datas/mt_rl_aligned.json"  # 2200 tasks from sub2+sub3
export BASE_OUTPUT_DIR="./checkpoint/gui_exp"
export BASE_LOG_DIR="./logs/train"

# =============================================================================
# Paper Table 8 - RL Hyperparameters (EXACT)
# =============================================================================
export LEARNING_RATE=1e-6
export NUM_TRAIN_EPOCHS=10  # Paper uses 5; doubled to compensate for smaller effective batch
export NUM_GENERATIONS=3  # Paper: 8. Must divide total batch (3 GPUs x 1 = 3). Reduced from 6 to prevent multi-turn OOM.
export TEMPERATURE=1.2
export TOP_P=1.0
export TOP_K=8
export MAX_PIXELS=200704

# GPU Adjustment for 3x A100 80GB (80GB each)
# Each generation runs up to 12 multi-turn steps of inference
# batch=1, num_gen=3: 1 episode/GPU, proven stable at ~55 GiB (25 GiB headroom)
# 3 GPUs × batch=1 × grad_acc=16 = 48 effective batch
export NPROC_PER_NODE=3
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=16

# GRPO Configuration 
export RLHF_TYPE="grpo"
export TRAIN_TYPE="full"
export TORCH_DTYPE="bfloat16"
export DEEPSPEED_CONFIG="zero3"
export GRADIENT_CHECKPOINTING="true"
export MAX_COMPLETION_LENGTH=1024
export MAX_LENGTH=4096  # Original: 4096
export WARMUP_RATIO=0.05

# Reward Function (sparse A2B: +1 if target reached and "complete" action, 0 otherwise)
export REWARD_FUNCS="a2b"
export REWARD_WEIGHTS="1.0"

# Multi-Turn Environment Function (enables interactive rollouts)
export MULTI_TURN_FUNC="gelab_multi_turn"

# Logging
export EVAL_STEPS=500
export SAVE_STEPS=100
export SAVE_TOTAL_LIMIT=2
export SAVE_ONLY_MODEL="true"
export LOGGING_STEPS=1
export DATALOADER_NUM_WORKERS=4
export DATASET_NUM_PROC=4
export LOG_COMPLETIONS="true"

# System Prompt (Paper Appendix A.10)
read -r -d '' SYSTEM_PROMPT << 'PROMPT_EOF'
You are a Multifaceted Mobile Interface Assistant. Your responsibilities include:

- 1. Navigating a mobile phone interface to reach a target page based on user instructions, task history, and the current screen state.
- 2. Understanding icons by identifying their name or function based on their location on the screen.
- 3. Grounding icons by locating the coordinates of an icon based on its name or description.

You will receive input that typically includes:

- User Request: Specifies the goal (navigation, understanding, or grounding). This might be a complex instruction for navigation or a direct question/command for icon tasks.
- Task History (Optional, primarily for Navigation): Records previous steps.
- Current Screen State: Represents the current screen, an image (indicated by <image>).

Based on the user request and the current screen state (and history if applicable), you must first determine the type of task requested and then provide the appropriate output.

--- Task Types and Output Formats ---

1. Task: Navigation

- Goal: Reach a target page step-by-step.
- Typical Input: Multi-turn instruction, history, and state. screen description and screenshot.
- Possible Actions:
  - click: Tap a specific element. Provide coordinates (x, y) relative to a (0,0) top-left and (1000,1000) bottom-right system.
  - complete: Task finished, current screen is the target.
- Output Format:
Explain: [Your brief explanation, e.g., 'click xxx icon on yyy page.', 'this is the target page.']	Action: [click(start_box=<|box_start|>(x,y)<|box_end|>) or complete]

2. Task: Icon Grounding (Locating an Icon)

- Goal: Identify the coordinates of a requested icon.
- Typical Input: User request like "Click on [icon name/description] in the image.", screen image (<image>).
- Action: Implicitly click (meaning "identify location").
- Output Format:
Action: click(start_box=<|box_start|>(x,y)<|box_end|>)

3. Task: Icon Understanding (Identifying an Icon)

- Goal: Provide the name or function of an icon at given coordinates.
- Typical Input: User request like "What is the icon at point (x, y) in the image?", screen image (<image>).
- Action: Provide textual information.
- Output Format:
[Icon Name or Description]

--- General Instructions ---

- Carefully analyze the user request to determine the task (Navigation, Grounding, Understanding).
- Analyze the current screen state (description or image) thoroughly.
- For actions involving coordinates (click), use the (0,0) to (1000,1000) system.
- Strictly adhere to the specified output format for the determined task type. Use a tab character (\t) as a separator where indicated.
PROMPT_EOF
export SYSTEM_PROMPT

# Create directories
mkdir -p "$BASE_LOG_DIR"
mkdir -p "$HF_HOME"
mkdir -p "$XDG_CACHE_HOME"

time_start=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$BASE_OUTPUT_DIR/$SAVE_NAME/v0-${time_start}"
LOG_FILE="$BASE_LOG_DIR/mt_rl_448_${time_start}.log"

echo "============================================================"
echo "MT-RL TRAINING (Paper Table 8 Aligned)"
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
    --multi_turn_func "$MULTI_TURN_FUNC" \
    --system "$SYSTEM_PROMPT" \
    --add_version False \
    --report_to "$REPORT_TO" \
    --log_completions "$LOG_COMPLETIONS" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "MT-RL Training Complete"
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
