#!/usr/bin/bash
# v2 pipeline: full 2-stage at LR=1e-6 + MAX_PIXELS=1003520 (Run 5/Run 7 calibration)
# Stage 1 train -> upload -> base eval ; Stage 3 train -> upload -> base eval ; doc update

set -e
LOG=/tmp/pipeline_v2.log
exec > >(tee -a "$LOG") 2>&1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

echo "=========================================="
echo "v2 pipeline (LR=1e-6, 23k luca) started: $(date)"
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

# Run-5 calibrated overrides
export LEARNING_RATE_OVERRIDE=1e-6
export MAX_PIXELS_OVERRIDE=1003520
export PER_DEVICE_TRAIN_BATCH_SIZE_OVERRIDE=8
export GRADIENT_ACCUMULATION_STEPS_OVERRIDE=8

# -----------------------------------------------------------------
# Stage 1: SFT base Qwen on 23k luca amex-gelab @ LR=1e-6, 1003520 px
# -----------------------------------------------------------------
echo ""
echo "[1A] Stage 1 SFT training (LR=1e-6, MAX_PIXELS=1003520, 23k luca)..."
MODEL_STAGE=sft_amex_full bash gui_scripts/continue_train_448.sh
echo "[1A] Stage 1 train done at $(date)"

STAGE1_DIR=$(ls -td checkpoint/gui_exp/sft_amex_gelab_full/v0-* 2>/dev/null | head -1)
STAGE1_CKPT=$(ls -td "$STAGE1_DIR"/checkpoint-* 2>/dev/null | head -1)
echo "[1A] Stage 1 checkpoint: $STAGE1_CKPT"

# -----------------------------------------------------------------
# Stage 1 base_model eval (compare to Run 5 = 64.03%)
# -----------------------------------------------------------------
echo ""
echo "[1B] Stage 1 base_model eval..."
python eval/evaluate_real_world.py --model_path "$STAGE1_CKPT" --num_gpus 4 --base_model \
  --output_file eval/results_stage1_v2_base.json
cat eval/results_stage1_v2_base.json
echo "[1B] done at $(date)"

# -----------------------------------------------------------------
# Stage 1 HF upload (parallel-friendly: after eval to avoid disk I/O contention)
# -----------------------------------------------------------------
echo ""
echo "[1C] Uploading Stage 1 to HF (v2)..."
STAGE1_CKPT_PATH="$STAGE1_CKPT" python <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo_id = "namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-v2"
ckpt = os.environ['STAGE1_CKPT_PATH']
print(f"Uploading {ckpt} -> {repo_id}")
api = HfApi(token=os.environ['HF_TOKEN'])
create_repo(repo_id, repo_type="model", private=False, exist_ok=True, token=os.environ['HF_TOKEN'])
api.upload_folder(folder_path=ckpt, repo_id=repo_id, repo_type="model",
    commit_message="Stage 1 v2: SFT base Qwen on 23k luca amex-gelab, LR=1e-6, MAX_PIXELS=1003520, 1 epoch (Run 5 calibration)")
print("STAGE 1 UPLOAD DONE")
PY
echo "[1C] Stage 1 upload done at $(date)"

# -----------------------------------------------------------------
# Stage 3: continue-train Stage 1 v2 on 15k no-AMEX @ LR=1e-6
# -----------------------------------------------------------------
echo ""
echo "[3A] Stage 3 continue-train (LR=1e-6, 15k no-AMEX)..."
CT_MODEL_PATH="$STAGE1_CKPT" \
MODEL_STAGE=ct_amex_full \
bash gui_scripts/continue_train_448.sh
echo "[3A] Stage 3 train done at $(date)"

STAGE3_DIR=$(ls -td checkpoint/gui_exp/ct_amex_full_no_amex/v0-* 2>/dev/null | head -1)
STAGE3_CKPT=$(ls -td "$STAGE3_DIR"/checkpoint-* 2>/dev/null | head -1)
echo "[3A] Stage 3 checkpoint: $STAGE3_CKPT"

# -----------------------------------------------------------------
# Stage 3 base_model eval (compare to Run 7 = 70.11%)
# -----------------------------------------------------------------
echo ""
echo "[3B] Stage 3 base_model eval..."
python eval/evaluate_real_world.py --model_path "$STAGE3_CKPT" --num_gpus 4 --base_model \
  --output_file eval/results_stage3_v2_base.json
cat eval/results_stage3_v2_base.json
echo "[3B] done at $(date)"

# -----------------------------------------------------------------
# Stage 3 HF upload
# -----------------------------------------------------------------
echo ""
echo "[3C] Uploading Stage 3 to HF (v2)..."
STAGE3_CKPT_PATH="$STAGE3_CKPT" python <<'PY'
import os
from huggingface_hub import HfApi, create_repo
repo_id = "namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-ContinueTrain-v2"
ckpt = os.environ['STAGE3_CKPT_PATH']
print(f"Uploading {ckpt} -> {repo_id}")
api = HfApi(token=os.environ['HF_TOKEN'])
create_repo(repo_id, repo_type="model", private=False, exist_ok=True, token=os.environ['HF_TOKEN'])
api.upload_folder(folder_path=ckpt, repo_id=repo_id, repo_type="model",
    commit_message="Stage 3 v2: continue-train Stage 1 v2 on 15k no-AMEX (mind2web+aitw+aitz), LR=1e-6, MAX_PIXELS=1003520, 2 epochs")
print("STAGE 3 UPLOAD DONE")
PY
echo "[3C] Stage 3 upload done at $(date)"

# -----------------------------------------------------------------
# Doc update
# -----------------------------------------------------------------
echo ""
echo "[F] Updating continue_train.md..."
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
    'Run 15 Stage 1 v2 (23k luca, LR=1e-6, 1003520 px) [base eval]':  load('eval/results_stage1_v2_base.json'),
    'Run 15 Stage 1+3 v2 (Continue-Train 15k no-AMEX, LR=1e-6) [base eval]': load('eval/results_stage3_v2_base.json'),
}
block = []
block.append("")
block.append("### Run 15: Paper-style 2-stage pipeline at LR=1e-6 (Run 5/Run 7 calibration)")
block.append("")
block.append("- **Motivation**: Run 14 used paper LR=1e-5 + native pixels which collapsed Stage 1 to 15.36% (vs Run 5's 64.03%) due to catastrophic forgetting of base Qwen grounding. Re-running with Run 5/Run 7 proven calibration.")
block.append("- **Stage 1 v2**: Base Qwen2.5-VL-7B-Instruct -> SFT on 23k luca amex-gelab (~7.7k each task), MAX_PIXELS=1,003,520 (Run 5 baseline), **LR=1e-6**, 1 epoch, eff batch 256, max_length 5120. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-v2")
block.append("- **Stage 3 v2**: Stage 1 v2 -> continue-train on 15k mind2web/aitw/aitz (no AMEX), MAX_PIXELS=1,003,520, **LR=1e-6**, 2 epochs, eff batch 256, max_length 5120. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-ContinueTrain-v2")
block.append("")
block.append("| Variant | ScreenSpot | SS-v2 | MoTIF | Refexp | VWB-EG | VWB-AG | Avg |")
block.append("|---------|-----------|-------|-------|--------|--------|--------|-----|")
for n, r in results.items():
    block.append(fmt_row(n, r))
block.append("")
new_block = "\n".join(block)
with open('continue_train.md','r') as f: doc = f.read()
doc = doc.rstrip() + "\n\n" + new_block + "\n"
with open('continue_train.md','w') as f: f.write(doc)
print("Doc updated")
print(new_block)
PY

echo ""
echo "=========================================="
echo "v2 pipeline complete at $(date)"
echo "=========================================="
