#!/usr/bin/bash
# v3 pipeline: FULL luca dataset (2.02M) + native pixels + LR=1e-6
# Stage 1 train -> eval -> upload ; Stage 3 train -> eval -> upload ; doc update

set -e
LOG=/tmp/pipeline_v3.log
exec > >(tee -a "$LOG") 2>&1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

echo "=========================================="
echo "v3 pipeline (225k balanced luca, native px, LR=1e-6) started: $(date)"
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

# Config: LR=1e-6 (Run 15 calibration), native MAX_PIXELS, batch for H200
export LEARNING_RATE_OVERRIDE=1e-6
export MAX_PIXELS_OVERRIDE=12845056
export PER_DEVICE_TRAIN_BATCH_SIZE_OVERRIDE=4
export GRADIENT_ACCUMULATION_STEPS_OVERRIDE=16
# Override dataset to full 2.02M
export DATASET_PATH=datas/sft_amex_gelab_225k.json

# -----------------------------------------------------------------
# Stage 1: SFT base Qwen on 225k balanced luca amex-gelab
# -----------------------------------------------------------------
echo ""
echo "[1A] Stage 1 SFT training (225k balanced luca, LR=1e-6, native px)..."
echo "     ETA: ~16h. Checkpoints saved every 500 steps."
MODEL_STAGE=sft_amex_full bash gui_scripts/continue_train_448.sh
echo "[1A] Stage 1 train done at $(date)"

STAGE1_DIR=$(ls -td checkpoint/gui_exp/sft_amex_gelab_full/v0-* 2>/dev/null | head -1)
STAGE1_CKPT=$(ls -td "$STAGE1_DIR"/checkpoint-* 2>/dev/null | head -1)
echo "[1A] Stage 1 checkpoint: $STAGE1_CKPT"

# -----------------------------------------------------------------
# Stage 1 base_model eval
# -----------------------------------------------------------------
echo ""
echo "[1B] Stage 1 base_model eval..."
python eval/evaluate_real_world.py --model_path "$STAGE1_CKPT" --num_gpus 4 --base_model \
  --output_file eval/results_stage1_v3_base.json
cat eval/results_stage1_v3_base.json
echo "[1B] done at $(date)"

# -----------------------------------------------------------------
# Stage 1 HF upload with retry
# -----------------------------------------------------------------
echo ""
echo "[1C] Uploading Stage 1 to HF (v3)..."
STAGE1_CKPT_PATH="$STAGE1_CKPT" python <<'PY'
import os, time
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError
repo_id = "namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-v3"
ckpt = os.environ['STAGE1_CKPT_PATH']
api = HfApi(token=os.environ['HF_TOKEN'])
create_repo(repo_id, repo_type="model", private=False, exist_ok=True, token=os.environ['HF_TOKEN'])
for attempt in range(5):
    try:
        api.upload_folder(folder_path=ckpt, repo_id=repo_id, repo_type="model",
            commit_message="Stage 1 v3: SFT base Qwen on 225k balanced luca amex-gelab, LR=1e-6, MAX_PIXELS=12845056 (native), 1 epoch")
        print("STAGE 1 UPLOAD DONE"); break
    except HfHubHTTPError as e:
        print(f"attempt {attempt+1} failed: {e}"); time.sleep(60*(attempt+1))
else: raise RuntimeError("Stage 1 upload failed after 5 attempts")
PY
echo "[1C] done at $(date)"

# -----------------------------------------------------------------
# Stage 3: continue-train on 15k no-AMEX @ native px
# -----------------------------------------------------------------
echo ""
echo "[3A] Stage 3 continue-train (LR=1e-6, 15k no-AMEX, native px)..."
export DATASET_PATH=datas/continue_train_no_amex_15k.json
CT_MODEL_PATH="$STAGE1_CKPT" \
MODEL_STAGE=ct_amex_full \
bash gui_scripts/continue_train_448.sh
echo "[3A] Stage 3 train done at $(date)"

STAGE3_DIR=$(ls -td checkpoint/gui_exp/ct_amex_full_no_amex/v0-* 2>/dev/null | head -1)
STAGE3_CKPT=$(ls -td "$STAGE3_DIR"/checkpoint-* 2>/dev/null | head -1)
echo "[3A] Stage 3 checkpoint: $STAGE3_CKPT"

# -----------------------------------------------------------------
# Stage 3 base_model eval
# -----------------------------------------------------------------
echo ""
echo "[3B] Stage 3 base_model eval..."
python eval/evaluate_real_world.py --model_path "$STAGE3_CKPT" --num_gpus 4 --base_model \
  --output_file eval/results_stage3_v3_base.json
cat eval/results_stage3_v3_base.json
echo "[3B] done at $(date)"

# -----------------------------------------------------------------
# Stage 3 HF upload with retry
# -----------------------------------------------------------------
echo ""
echo "[3C] Uploading Stage 3 to HF (v3)..."
STAGE3_CKPT_PATH="$STAGE3_CKPT" python <<'PY'
import os, time
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError
repo_id = "namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-ContinueTrain-v3"
ckpt = os.environ['STAGE3_CKPT_PATH']
api = HfApi(token=os.environ['HF_TOKEN'])
create_repo(repo_id, repo_type="model", private=False, exist_ok=True, token=os.environ['HF_TOKEN'])
for attempt in range(5):
    try:
        api.upload_folder(folder_path=ckpt, repo_id=repo_id, repo_type="model",
            commit_message="Stage 3 v3: continue-train Stage 1 v3 on 15k no-AMEX, LR=1e-6, MAX_PIXELS=12845056, 2 epochs")
        print("STAGE 3 UPLOAD DONE"); break
    except HfHubHTTPError as e:
        print(f"attempt {attempt+1} failed: {e}"); time.sleep(60*(attempt+1))
else: raise RuntimeError("Stage 3 upload failed after 5 attempts")
PY
echo "[3C] done at $(date)"

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
    'Run 16 Stage 1 v3 (225k balanced luca, LR=1e-6, native px) [base eval]': load('eval/results_stage1_v3_base.json'),
    'Run 16 Stage 1+3 v3 (CT 15k no-AMEX, LR=1e-6, native px) [base eval]': load('eval/results_stage3_v3_base.json'),
}
block = []
block.append("")
block.append("### Run 16: Full 2.02M luca dataset + native resolution (LR=1e-6)")
block.append("")
block.append("- **Motivation**: Test whether the FULL luca amex-gelab dataset (2.02M samples: 24.8k nav + 1M grounding + 1M understanding) at native resolution improves over Run 15 Stage 1 v2 (23k, 69.79%).")
block.append("- **Stage 1 v3**: Base Qwen2.5-VL-7B-Instruct -> SFT on 225k balanced luca amex-gelab, MAX_PIXELS=12,845,056 (native), **LR=1e-6**, 1 epoch, eff batch 256, max_length 5120. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-v3")
block.append("- **Stage 3 v3**: Stage 1 v3 -> continue-train on 15k mind2web/aitw/aitz (no AMEX), MAX_PIXELS=12,845,056, **LR=1e-6**, 2 epochs. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-ContinueTrain-v3")
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
echo "v3 pipeline complete at $(date)"
echo "=========================================="
