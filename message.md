# GE-Lab SFT Training - Server Migration Guide

**Date**: 2026-01-27  
**Branch**: `koh-dev/sft`  
**Status**: Training in progress with format-fixed data

---

## Quick Start on New Server

### 1. Clone and Setup

```bash
git clone <repo-url>
cd gelab-engine
git checkout koh-dev/sft
```

### 2. Create Conda Environment

```bash
# Create environment
conda create -n gelab python=3.10 -y
conda activate gelab

# Install PyTorch (CUDA 12.x)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ms-swift (training framework)
pip install ms-swift[all]

# Install other dependencies
pip install transformers accelerate deepspeed
pip install vllm  # For evaluation
pip install qwen-vl-utils  # For Qwen2.5-VL
pip install wandb networkx pillow tqdm

# Or use requirements file
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export HF_HOME=/path/to/cache/huggingface
export TRANSFORMERS_CACHE=/path/to/cache/huggingface
export WANDB_API_KEY=<WANDB_API_KEY>
export WANDB_PROJECT=gelab-fresh
```

---

## Data Generation

### Generate SFT Training Data

```bash
python scripts/generate_sft_data.py \
    --ui_structure demo/ui_structure.json \
    --pages_dir datas/images \
    --output_dir datas \
    --sft_subtrees 0 1
```

**Output files:**
- `datas/sft.json` - 30,888 samples (navigation + icon captioning/grounding)
- `datas/icon_captioning.json` - 2,320 samples
- `datas/icon_grounding.json` - 2,320 samples
- `datas/st_rl.json` - 24,878 samples
- `datas/test.json` - 2,162 tasks

### Generate Evaluation Data

```bash
python scripts/generate_eval_data.py \
    --ui_structure demo/ui_structure.json \
    --images_dir datas/images \
    --output_dir datas \
    --test_subtree 4
```

---

## Training Command
### SFT Training (2 GPUs with DeepSpeed ZeRO-3) - WORKING

```bash
tmux new -s gelab-sft-fixed

cd /path/to/gelab-engine && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate gelab && \
export CUDA_VISIBLE_DEVICES=0,2 && \
export HF_HOME=/path/to/cache/huggingface && \
export TRANSFORMERS_CACHE=/path/to/cache/huggingface && \
export MODELSCOPE_CACHE=/path/to/cache/modelscope && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
export NCCL_P2P_DISABLE=1 && \
export NCCL_IB_DISABLE=1 && \
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 && \
export WANDB_API_KEY=<WANDB_API_KEY> && \
export WANDB_PROJECT=gelab-fresh && \
NPROC_PER_NODE=2 swift sft \
    --model /path/to/cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
    --train_type full \
    --target_modules all-linear \
    --dataset datas/sft.json \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.05 \
    --num_train_epochs 1 \
    --eval_steps 100 \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir checkpoint/gui_exp/sft_fixed \
    --deepspeed zero3 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --report_to wandb \
    --run_name gelab-sft-fixed-v4
```

**Important Notes:**
- `batch_size=1` required due to 7B VL model memory requirements (~78GB per GPU)
- `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` needed for non-contiguous GPU IDs
- `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` prevents NCCL watchdog timeouts
- Estimated training time: ~25 hours on 2x A100 80GB

### For 3 GPUs (faster, ~18 hours)

Use `CUDA_VISIBLE_DEVICES=0,1,2` and `NPROC_PER_NODE=3` with `batch_size=2`

---

## Evaluation Commands

### Interactive Evaluation

```bash
python eval/interactive_eval.py \
    --model_path checkpoint/gui_exp/sft_fixed/checkpoint-XXXX \
    --ui_structure demo/ui_structure.json \
    --images_dir datas/images \
    --output eval_results/interactive_fixed.json \
    --gpu 0
```

### Static Evaluation

```bash
# Run inference
python eval/inference_qwen2p5_mixed_vllm.py \
    --model checkpoint/gui_exp/sft_fixed/checkpoint-XXXX \
    --test_file datas/eval_sub4_test.json \
    --save_file eval_results/static_predictions.jsonl

# Calculate scores
python eval/calculate_score_refine.py --file eval_results/static_predictions.jsonl
```

---

## Issues Identified & Fixed

### Issue 1: Icon Name Format Mismatch (CRITICAL)

**Problem**: Icon grounding/captioning used space format (`Digital electronics 98`) while navigation used underscore format (`Digital_electronics_98`). Model couldn't transfer knowledge.

**Fix**: Modified `scripts/generate_sft_data.py` to use underscore format consistently.

### Issue 2: Missing Edge Data from Test Subtree

**Problem**: Paper requires edge data from ALL subtrees including test. Our data had 0 edge samples from subtree 4.

**Fix**: Added edge data from all 5 subtrees (274 samples each).

### Issue 3: Prompt Format Mismatch

**Problem**: Eval used `Instruction: from X to Y.` but training used `Instruction: from X to Y. History: Null`

**Fix**: Updated `eval/interactive_eval.py` to include `History: Null` on first step.

---

## Current Results (Before Fix)

| Metric | Our Result | Paper Target | Gap |
|--------|------------|--------------|-----|
| Path@1 Pass@1 | 54.74% | 99.71% | -44.97% |
| Static Overall | 53.51% | 55.46% | -1.95% |

**Key Finding**: Static eval close to paper (~2% gap), but interactive eval had ~45% gap due to format mismatch.

---

## Expected Results (After Fix)

With format fixes, we expect:
- Path@1 Pass@1: ~80-90% (from 54.74%)
- Content icon accuracy: Significant improvement (from 2%)

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/generate_sft_data.py` | Generate all training data |
| `scripts/generate_eval_data.py` | Generate evaluation data |
| `eval/interactive_eval.py` | Interactive benchmark evaluation |
| `eval/interactive_env.py` | Environment simulator |
| `demo/ui_structure.json` | UI environment structure (use this!) |
| `datas/images/` | Page screenshots |

---

## Important Notes

1. **Always use `demo/ui_structure.json`** - NOT `data_engine/ui_environment/`
2. **Images are in `datas/images/`** - These match `demo/ui_structure.json`
3. **Subtree split**: 0,1 for SFT, 2,3 for RL, 4 for Test
4. **Edge data from ALL subtrees** is required per paper

---

## Contact

Check `Progress.md` and `results.md` for detailed logs.
