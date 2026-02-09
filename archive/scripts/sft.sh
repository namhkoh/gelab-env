#!/usr/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Experiment Configuration ---

# ** WANDB Configuration **
export WANDB_API_KEY="" # Consider setting via environment or secrets
export WANDB_PROJECT="demo"
export REPORT_TO="wandb" # Set to "none" or remove line to disable wandb


export SAVE_NAME="sft" # Unique name for this run
export MODEL_PATH="checkpoint-1"
export DATASET_PATH="datas/sft.json"
export BASE_OUTPUT_DIR="./checkpoint/gui_exp" # Base directory for checkpoints
export BASE_LOG_DIR="./logs/train" # Base directory for logs

# ** Swift SFT Parameters **
export SFT_MAX_PIXELS=200704 # Value for the command prefix in worker script
export TRAIN_TYPE="full"
export TORCH_DTYPE="bfloat16"
export NUM_TRAIN_EPOCHS=1
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export PER_DEVICE_EVAL_BATCH_SIZE=1
export LEARNING_RATE=1e-5
export TARGET_MODULES="all-linear"
export FREEZE_VIT="true"
export GRADIENT_ACCUMULATION_STEPS=2
export DEEPSPEED_CONFIG="zero3" # Or path to a deepspeed config json
export LOGGING_STEPS=1
export MAX_LENGTH=10240 # Max sequence length for model/tokenizer
export TRUNCATION_STRATEGY="delete"
export MAX_NEW_TOKENS=512
export WARMUP_RATIO=0.05
export DATALOADER_NUM_WORKERS=4
export SAVE_STEPS=6000
export SAVE_TOTAL_LIMIT=10

# ** System Prompt (Ensure proper quoting if it contains special characters) **
export SYSTEM_PROMPT="You are a **Multifaceted Mobile Interface Assistant**. Your responsibilities include:\n1.  **Navigating** a mobile phone interface to reach a target page based on user instructions, task history, and the current screen state.\n2.  **Understanding icons** by identifying their name or function based on their location on the screen.\n3.  **Grounding icons** by locating the coordinates of an icon based on its name or description.\n\nYou will receive input that typically includes:\n*   A **User Request:** Specifies the goal (navigation, understanding, or grounding). This might be a complex instruction for navigation or a direct question/command for icon tasks.\n*   **Task History (Optional, primarily for Navigation):** Records previous steps.\n*   **Current Screen State:** Represents the current screen, an image (indicated by <image>).\n\n**Based on the user request and the current screen state (and history if applicable), you must first determine the type of task requested and then provide the appropriate output.**\n\n--- Task Types and Output Formats ---\n\n**1. Task: Navigation**\n   *   **Goal:** Reach a target page step-by-step.\n   *   **Typical Input:** Multi-turn instruction, history, and state. screen description and screenshot.\n   *   **Possible Actions:**\n      *   CLICK: Tap a specific element. Provide coordinates (x, y) relative to a (0,0) top-left and (1000,1000) bottom-right system.\n      *   COMPLETE: Task finished, current screen is the target.\n   *   **Output Format:**\n      \n      explain: [Your brief explanation, e.g., 'click xxx icon on yyy page.', 'this is the target page.']\\taction: [CLICK or COMPLETE]\\tpoint:(x, y)  # Include point only for CLICK\n      \n\n**2. Task: Icon Grounding (Locating an Icon)**\n   *   **Goal:** Identify the coordinates of a requested icon.\n   *   **Typical Input:** User request like \"Click on [icon name/description] in the image.\", screen image (<image>).\n   *   **Action:** Implicitly CLICK (meaning \"identify location\").\n   *   **Output Format:**\n      \n      action: CLICK\\tpoint:(x, y)\n      \n      *(Note: The explanation is often implicit in the grounding request itself).*\n\n**3. Task: Icon Understanding (Identifying an Icon)**\n   *   **Goal:** Provide the name or function of an icon at given coordinates.\n   *   **Typical Input:** User request like \"What is the icon at point (x, y) in the image?\", screen image (<image>).\n   *   **Action:** Provide textual information.\n   *   **Output Format:**\n      \n      [Icon Name or Description]\n      \n      *(Just the direct answer as text).*\n\n--- General Instructions ---\n\n*   Carefully analyze the user request to determine the task (Navigation, Grounding, Understanding).\n*   Analyze the current screen state (description or image) thoroughly.\n*   For actions involving coordinates (CLICK), use the (0,0) to (1000,1000) system.\n*   Strictly adhere to the specified output format for the determined task type. Use a tab character (\\t) as a separator where indicated.\n\nOutput:"

# --- Execution ---

# Define the path to the worker script
WORKER_SCRIPT="./gui_scripts/worker_script_sft_packing.sh" # Make sure this path is correct

# Check if worker script exists and is executable
if [ ! -f "$WORKER_SCRIPT" ]; then
    echo "Error: Worker script '$WORKER_SCRIPT' not found."
    exit 1
fi
if [ ! -x "$WORKER_SCRIPT" ]; then
    echo "Error: Worker script '$WORKER_SCRIPT' is not executable. Run: chmod +x $WORKER_SCRIPT"
    exit 1
fi

# Prepare Log directory and file for rlaunch output
mkdir -p "$BASE_LOG_DIR"
time_launch=$(TZ= date '+%Y-%m-%d_%H%M%S')
LOG_FILE_LAUNCHER="$BASE_LOG_DIR/gui_exp_sft_launch_${SAVE_NAME}_${time_launch}.log" # Log file for rlaunch itself

# ** Resource Allocation **
export NNODES=2
export NPROC_PER_NODE=8 
export CPU_PER_NODE=40
export MEMORY_PER_NODE_MB=$((1024*400)) # Memory in MB


time_start=$(TZ= date '+%Y-%m-%d_%H:%M:%S_%Z')
echo "Starting SFT training launch at: $time_start"
echo "Experiment Name: $SAVE_NAME"
echo "Number of Nodes: $NNODES"
echo "Processes per Node: $NPROC_PER_NODE"
echo "SFT Max Pixels: $SFT_MAX_PIXELS"
echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Output Base Directory: $BASE_OUTPUT_DIR"
echo "Log Base Directory: $BASE_LOG_DIR"
echo "rlaunch Log File: $LOG_FILE_LAUNCHER"
echo "----------------------------------------"


# Construct rlaunch command
bash $WORKER_SCRIPT