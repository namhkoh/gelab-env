#!/usr/bin/bash
# Pipeline orchestrator: Stage 1 base_model eval -> Stage 3 train -> HF upload -> Stage 3 eval -> doc update
# Assumes: Stage 1 sysprompt eval is currently running and will finish; Stage 1 already uploaded.
set -e

LOG=/tmp/pipeline_stages23.log
exec > >(tee -a "$LOG") 2>&1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

echo "=========================================="
echo "Pipeline orchestrator started: $(date)"
echo "=========================================="

source /opt/conda/etc/profile.d/conda.sh
conda activate gelab

: "${WANDB_API_KEY:?WANDB_API_KEY must be set in environment (e.g. via .env)}"
export WANDB_API_KEY
export WANDB_ENTITY="namhokoh-korea-advanced-institute-of-science-and-technology"
export WANDB_PROJECT="gelab"
: "${HF_TOKEN:?HF_TOKEN must be set in environment (e.g. via .env)}"
export HF_TOKEN
export PATH="/opt/conda/envs/gelab/bin:$PATH"

STAGE1_CKPT="checkpoint/gui_exp/sft_amex_gelab_full/v0-20260407_090358/checkpoint-196"

# -----------------------------------------------------------------
# Step A: wait for the in-flight Stage 1 sysprompt eval to finish
# -----------------------------------------------------------------
echo "[A] Waiting for Stage 1 sysprompt eval to finish..."
while pgrep -f "evaluate_real_world.*results_stage1_sft_amex_gelab_50k_sysprompt" > /dev/null; do
    sleep 60
done
echo "[A] Sysprompt eval done at $(date)"
[ -f eval/results_stage1_sft_amex_gelab_50k_sysprompt.json ] && cat eval/results_stage1_sft_amex_gelab_50k_sysprompt.json

# -----------------------------------------------------------------
# Step B: Stage 1 base_model eval (for comparison vs Run 5 / Run 7)
# -----------------------------------------------------------------
echo ""
echo "[B] Running Stage 1 --base_model eval..."
python eval/evaluate_real_world.py \
  --model_path "$STAGE1_CKPT" \
  --num_gpus 4 \
  --base_model \
  --output_file eval/results_stage1_sft_amex_gelab_50k_base.json
echo "[B] Stage 1 base_model eval done at $(date)"
cat eval/results_stage1_sft_amex_gelab_50k_base.json

# -----------------------------------------------------------------
# Step C: Launch Stage 3 - continue-train Stage 1 SFT on 15k no-AMEX
# -----------------------------------------------------------------
echo ""
echo "[C] Launching Stage 3 continue-train..."
CT_MODEL_PATH="$STAGE1_CKPT" \
MODEL_STAGE=ct_amex_full \
bash gui_scripts/continue_train_448.sh
echo "[C] Stage 3 training done at $(date)"

# Locate the latest Stage 3 checkpoint
STAGE3_DIR=$(ls -td checkpoint/gui_exp/ct_amex_full_no_amex/v0-* 2>/dev/null | head -1)
STAGE3_CKPT=$(ls -td "$STAGE3_DIR"/checkpoint-* 2>/dev/null | head -1)
echo "[C] Stage 3 checkpoint: $STAGE3_CKPT"
if [ -z "$STAGE3_CKPT" ]; then
    echo "FATAL: no Stage 3 checkpoint found"; exit 1
fi

# -----------------------------------------------------------------
# Step D: Upload Stage 3 model to HF
# -----------------------------------------------------------------
echo ""
echo "[D] Uploading Stage 3 to HF..."
python <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo_id = "namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-ContinueTrain-v1"
import subprocess
ckpt = subprocess.check_output(
    "ls -td checkpoint/gui_exp/ct_amex_full_no_amex/v0-*/checkpoint-* | head -1",
    shell=True, text=True).strip()
print(f"Uploading {ckpt} -> {repo_id}")
api = HfApi(token=os.environ['HF_TOKEN'])
create_repo(repo_id, repo_type="model", private=False, exist_ok=True, token=os.environ['HF_TOKEN'])
api.upload_folder(
    folder_path=ckpt,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Stage 3: continue-train Stage 1 SFT on 15k real-world no-AMEX (mind2web+aitw+aitz), LR=1e-5, 2 epochs",
)
print("UPLOAD DONE")
PY
echo "[D] Stage 3 HF upload done at $(date)"

# -----------------------------------------------------------------
# Step E: Stage 3 grounding eval (both modes)
# -----------------------------------------------------------------
echo ""
echo "[E1] Running Stage 3 --use_system_prompt eval..."
python eval/evaluate_real_world.py \
  --model_path "$STAGE3_CKPT" \
  --num_gpus 4 \
  --use_system_prompt \
  --output_file eval/results_stage3_ct_amex_full_sysprompt.json
echo "[E1] done at $(date)"
cat eval/results_stage3_ct_amex_full_sysprompt.json

echo ""
echo "[E2] Running Stage 3 --base_model eval..."
python eval/evaluate_real_world.py \
  --model_path "$STAGE3_CKPT" \
  --num_gpus 4 \
  --base_model \
  --output_file eval/results_stage3_ct_amex_full_base.json
echo "[E2] done at $(date)"
cat eval/results_stage3_ct_amex_full_base.json

# -----------------------------------------------------------------
# Step F: Update continue_train.md
# -----------------------------------------------------------------
echo ""
echo "[F] Generating doc update..."
python <<'PY'
import json
def load(p):
    try: return json.load(open(p))
    except: return None

def fmt_row(name, r):
    if not r: return f"| {name} | - | - | - | - | - | - | - |"
    b = r['benchmarks']
    def pct(k): return f"{b[k]['accuracy']*100:.2f}" if k in b else "-"
    return (f"| {name} | {pct('ScreenSpot')} | {pct('ScreenSpot-v2')} | {pct('MoTIF')} | "
            f"{pct('Refexp')} | {pct('VWB-EG')} | {pct('VWB-AG')} | {r['average_accuracy']*100:.2f} |")

results = {
    'Stage 1 (50k luca SFT, native px) [base eval]':  load('eval/results_stage1_sft_amex_gelab_50k_base.json'),
    'Stage 1 (50k luca SFT, native px) [sys eval]':   load('eval/results_stage1_sft_amex_gelab_50k_sysprompt.json'),
    'Stage 1+3 (Continue-Train, no AMEX) [base eval]': load('eval/results_stage3_ct_amex_full_base.json'),
    'Stage 1+3 (Continue-Train, no AMEX) [sys eval]': load('eval/results_stage3_ct_amex_full_sysprompt.json'),
}
block_lines = ["", "### Run 14: Paper-style 2-stage pipeline (Stage 1 SFT 50k luca + Stage 3 CT 15k no-AMEX)", ""]
block_lines.append("- **Stage 1**: Base Qwen2.5-VL-7B-Instruct -> SFT on 50k luca amex-gelab (16.7k each task), MAX_PIXELS=12,845,056 (Qwen base default native), LR=1e-5, 1 epoch, eff batch 256, max_length 5120. 196 steps in 4h 24m, final loss 0.281, token_acc 90.97%. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-v1")
block_lines.append("- **Stage 3**: Stage 1 SFT -> continue-train on 15k mind2web/aitw/aitz (no AMEX), MAX_PIXELS=12,845,056, LR=1e-5, 2 epochs, eff batch 256, max_length 5120. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-ContinueTrain-v1")
block_lines.append("")
block_lines.append("| Variant | ScreenSpot | SS-v2 | MoTIF | Refexp | VWB-EG | VWB-AG | Avg |")
block_lines.append("|---------|-----------|-------|-------|--------|--------|--------|-----|")
for n, r in results.items():
    block_lines.append(fmt_row(n, r))
block_lines.append("")

new_block = "\n".join(block_lines)
with open('continue_train.md','r') as f: doc = f.read()
# Append at end of file
doc = doc.rstrip() + "\n\n" + new_block + "\n"
with open('continue_train.md','w') as f: f.write(doc)
print("Doc updated")
print(new_block)
PY

echo ""
echo "=========================================="
echo "Pipeline complete at $(date)"
echo "=========================================="
