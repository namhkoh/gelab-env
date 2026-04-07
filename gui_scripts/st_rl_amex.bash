#!/usr/bin/bash
set -e

export WANDB_API_KEY="${WANDB_API_KEY:}"
export WANDB_PROJECT="${WANDB_PROJECT:-gelab}"
export HF_TOKEN="${HF_TOKEN:}"
export HF_HOME="${HF_HOME:-/home/irteam/data-vol1/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/irteam/data-vol1/.cache}"
export TORCH_HOME="${TORCH_HOME:-/home/irteam/data-vol1/.cache/torch}"
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH}"
export USE_HF="${USE_HF:-1}"
export REPORT_TO="${REPORT_TO:-wandb}"

export SAVE_NAME="${SAVE_NAME:-st_rl_amex}"
export MODEL_PATH="${MODEL_PATH:-/home/irteam/gelab-env/checkpoint/gui_exp/sft_amex/v0-20260405_112032/checkpoint-2473}"
# Generate with: python data_engine/generate_amex_st_rl_data.py \
#   --env_dir /home/irteam/data-vol1/amex_sft \
#   --output /home/irteam/data-vol1/amex_sft/st_rl_amex.json
export DATASET_PATH="${DATASET_PATH:-/home/irteam/data-vol1/gelab-env/datas/st_rl_amex.json}"
export CUSTOM_REGISTER_PATH="${CUSTOM_REGISTER_PATH:-/home/irteam/gelab-env/swift/plugin/orm_amex.py}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./checkpoint/gui_exp}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-./logs/train}"

export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:-8}"
export MAX_PIXELS="${MAX_PIXELS:-1048576}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

export RLHF_TYPE="${RLHF_TYPE:-grpo}"
export TRAIN_TYPE="${TRAIN_TYPE:-full}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero3}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

export REWARD_FUNCS="${REWARD_FUNCS:-amex_action_match amex_coordinate_match amex_intent_match amex_format_constraint}"
export REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.25 0.25 0.25 0.25}"

export EVAL_STEPS="${EVAL_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
export LOGGING_STEPS="${LOGGING_STEPS:-5}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
export LOG_COMPLETIONS="${LOG_COMPLETIONS:-true}"

read -r -d '' SYSTEM_PROMPT << 'PROMPT_EOF' || true
You are a Multifaceted Mobile Interface Assistant.

Your responsibility is to navigate a mobile phone interface based on the task instruction, the action history, and the current screen state.

You will receive input that typically includes:
- Task Instruction: Describes the user goal you should help complete.
- Action History: Records previous actions taken in the trajectory.
- Current Screen State: The current mobile screenshot.

Based on the task instruction, action history, and current screen state, you must first determine the type of next-step decision required and then provide the appropriate output.

--- Task Types and Output Formats ---

#### 1. Task: Navigation

- Goal: Reach the task goal step-by-step by navigating the current mobile interface.
- Typical Input: Task instruction, action history, and the current screenshot.
- Possible Actions:
  - tap: Tap a specific element on the current screen.
  - swipe: Drag from one point to another to scroll or reveal more content.
  - press_back: Use the system back button when returning to a previous screen is the best next step.
  - press_home: Use the system home button when restarting from the home screen is the best next step.
- Output Format:
Explain: [Your brief explanation of the next navigation step]	Action: [tap(start_box='<|box_start|>(x,y)<|box_end|>') or swipe(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x2,y2)<|box_end|>') or press_back() or press_back(start_box='<|box_start|>(x,y)<|box_end|>') or press_home() or press_home(start_box='<|box_start|>(x,y)<|box_end|>')]

#### 2. Task: Text Entry

- Goal: Enter text or submit typed input when the task requires interaction with a text field.
- Typical Input: A screen containing an editable field and a task that requires typing, searching, logging in, or filling a form.
- Possible Actions:
  - type: Type the required text into the appropriate input field.
  - press_enter: Submit the currently typed input when submission is the best next step.
- Output Format:
Explain: [Your brief explanation of the text entry step]	Action: [type(start_box='<|box_start|>(x,y)<|box_end|>', text='...') or press_enter()]

#### 3. Task: Terminal State Decision

- Goal: Decide whether the task is already finished or cannot be completed from the current state.
- Typical Input: A screen that already satisfies the task goal, or a screen where no valid progress can be made.
- Possible Actions:
  - task complete: The task is finished.
  - task impossible: The task cannot be completed from the current state.
- Output Format:
Explain: [Your brief explanation of why the task is complete or impossible]	Action: [complete or impossible]

--- Coordinate Rules ---

- The screen resolution is 672 x 1512.
- All coordinates must be in the screen resolution coordinate space (0-672 for x, 0-1512 for y).

--- Valid Action Formats ---

- tap(start_box='<|box_start|>(x,y)<|box_end|>')
- swipe(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x2,y2)<|box_end|>')
- type(start_box='<|box_start|>(x,y)<|box_end|>', text='...')
- press_enter()
- press_back()
- press_back(start_box='<|box_start|>(x,y)<|box_end|>')
- press_home()
- press_home(start_box='<|box_start|>(x,y)<|box_end|>')
- complete
- impossible

--- General Instructions ---
- Analyze the task instruction, action history, and screenshot before deciding.
- Predict exactly one next action per step.
- Use action history to avoid repeating unhelpful actions.
- Strictly follow the output format. Output only a valid response.
PROMPT_EOF
export SYSTEM_PROMPT

mkdir -p "$BASE_LOG_DIR"
mkdir -p "$HF_HOME"
mkdir -p "$XDG_CACHE_HOME"

time_start=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="$BASE_OUTPUT_DIR/$SAVE_NAME/v0-${time_start}"
LOG_FILE="$BASE_LOG_DIR/st_rl_amex_${time_start}.log"

echo "============================================================"
echo "ST-RL TRAINING - AMEX"
echo "============================================================"
echo "Start time: $(date)"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Custom reward register: $CUSTOM_REGISTER_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NPROC_PER_NODE"
echo "Effective batch: $((PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))"
echo "============================================================"

python -c "import json; data=json.load(open('$DATASET_PATH')); print(f'Dataset samples: {len(data)}')"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
MAX_PIXELS="$MAX_PIXELS" \
swift rlhf \
    --rlhf_type "$RLHF_TYPE" \
    --model "$MODEL_PATH" \
    --custom_register_path "$CUSTOM_REGISTER_PATH" \
    --reward_funcs $REWARD_FUNCS \
    --reward_weights $REWARD_WEIGHTS \
    --train_type "$TRAIN_TYPE" \
    --torch_dtype "$TORCH_DTYPE" \
    --dataset "$DATASET_PATH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
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
    --num_generations "$NUM_GENERATIONS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --system "$SYSTEM_PROMPT" \
    --add_version False \
    --max_pixels "$MAX_PIXELS" \
    --report_to "$REPORT_TO" \
    --log_completions "$LOG_COMPLETIONS" \
    2>&1 | tee "$LOG_FILE" | grep --line-buffered -E "^\{|%\||train_loss|eval_loss|completed|steps|ETA|Epoch|Step"

echo ""
echo "============================================================"
echo "ST-RL AMEX Complete"
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

