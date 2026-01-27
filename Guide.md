# GeLab Engine - Project Guide

This document provides context for continuing work on the GeLab (GUI Environment Learning Agent) project.

## Project Overview

GeLab is a vision-language model training framework for GUI navigation agents, based on the paper: https://arxiv.org/pdf/2512.02423

The goal is to train a model that can navigate GUI environments by predicting paths through tree-structured UI graphs.

## What Has Been Done

### 1. Environment Setup

- **Conda environment**: `gelab` (Python 3.10)
- **Location**: `/ext_hdd2/nhkoh/gelab-engine`
- **Base model**: `Qwen/Qwen2.5-VL-7B-Instruct` (cached at `/ext_hdd2/nhkoh/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5`)

### 2. Credentials Configured

```bash
# HuggingFace
export HF_TOKEN=<HF_TOKEN>

# WANDB
export WANDB_API_KEY=<WANDB_API_KEY>
export WANDB_ENTITY=namhokoh-korea-advanced-institute-of-science-and-technology
export WANDB_PROJECT=gelab-fresh

# Cache directories (all point to ext_hdd2, NOT home directory)
export HF_HOME=/ext_hdd2/nhkoh/.cache/huggingface
export TRANSFORMERS_CACHE=/ext_hdd2/nhkoh/.cache/huggingface
export MODELSCOPE_CACHE=/ext_hdd2/nhkoh/.cache/modelscope
export HF_DATASETS_CACHE=/ext_hdd2/nhkoh/.cache/huggingface/datasets
export TORCH_HOME=/ext_hdd2/nhkoh/.cache/torch
```

### 3. UI Environment Generated

- **Location**: `data_engine/ui_environment/`
- **Structure**: Tree-structured graph with max depth 7
- **Node distribution**: `[5, 3, 2, 2, 1, 1, 0]` per level (matching paper)
- **Generated using**: `python data_engine/tree.py`

### 4. Dataset Generation

Created `scripts/generate_sft_data.py` which generates:

| Dataset | File | Samples |
|---------|------|---------|
| SFT (Navigation + Icon Cap/Gnd) | `datas/sft.json` | 29,718 |
| Single-Turn RL | `datas/st_rl.json` | 3,000 |
| Multi-Turn RL | `datas/mt_rl.json` | 3,000 |
| Test | `datas/test.json` | 5,000 |
| Icon Captioning | `datas/icon_captioning.json` | 2,320 |
| Icon Grounding | `datas/icon_grounding.json` | 2,320 |

**Note**: Paper reports ~60k SFT samples. The discrepancy may be due to data augmentation or multi-epoch counting not explicitly documented.

### 5. Git Branches

- `main` - Original codebase
- `koh-dev/dataset` - Dataset generation changes committed
- `koh-dev/sft` - Current branch for SFT training

## Current State: SFT Training

### Training Configuration

```bash
accelerate launch --num_processes 3 --mixed_precision bf16 -m swift.cli.sft \
    --model /ext_hdd2/nhkoh/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
    --model_type qwen2_5_vl \
    --train_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --dataset datas/sft.json \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --max_length 10240 \
    --gradient_checkpointing true \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --eval_strategy epoch \
    --dataloader_num_workers 4 \
    --output_dir checkpoint/gui_exp/sft \
    --run_name gelab-sft-qwen2.5vl-7b_v2 \
    --report_to wandb \
    --deepspeed zero3
```

### Training Details

- **GPUs**: 3x NVIDIA A100 80GB
- **Total steps**: 2,481
- **Checkpoints saved at**: Steps 500, 1000, 1500, 2000, and final
- **Checkpoint directory**: `checkpoint/gui_exp/sft/v3-20260124-120531/`
- **WANDB run**: https://wandb.ai/namhokoh-korea-advanced-institute-of-science-and-technology/gelab-fresh/runs/gvlau5bs
- **tmux session**: `gelab-sft`

### To Check Training Status

```bash
# Attach to tmux session
tmux attach -t gelab-sft

# Or check GPU usage
nvidia-smi

# Or check checkpoint directory
ls -la /ext_hdd2/nhkoh/gelab-engine/checkpoint/gui_exp/sft/v3-20260124-120531/
```

## Next Steps After Training Completes

### 1. Evaluation

The evaluation consists of two steps:

**Step 1: Generate Predictions**
```bash
python eval/inference_qwen2p5_mixed_vllm.py \
    --model_path checkpoint/gui_exp/sft/v3-20260124-120531 \
    --test_file datas/test.json \
    --output_file eval_results/predictions.json
```

**Step 2: Calculate Scores**
```bash
python eval/calculate_score_refine.py \
    --pred_file eval_results/predictions.json \
    --output_file eval_results/scores.json
```

### Evaluation Metrics

- **Path@k** (k=1-7): Path prediction accuracy at different depths
- **Complete**: Full path completion rate
- **Edge**: Edge prediction accuracy
- **Icon Captioning**: Caption accuracy for UI icons
- **Icon Grounding**: Bounding box localization accuracy (IoU-based)

## Important Notes

1. **Do NOT download model** - Use the cached model at `/ext_hdd2/nhkoh/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5`

2. **Cache directories** - All caches should be in `/ext_hdd2/nhkoh/.cache/`, NOT in the home directory

3. **Test data format** - The `datas/test.json` may need reformatting to match `inference_qwen2p5_mixed_vllm.py` expected input format (needs `messages` and `images` fields)

4. **Checkpoints** - Previous failed runs exist in `v0`, `v1`, `v2` directories. Current run is in `v3-20260124-120531`

## File Structure

```
/ext_hdd2/nhkoh/gelab-engine/
├── checkpoint/gui_exp/sft/     # Training checkpoints
├── data_engine/
│   ├── tree.py                 # UI environment generator
│   └── ui_environment/         # Generated UI trees (gitignored)
├── datas/
│   ├── sft.json               # SFT training data
│   ├── st_rl.json             # Single-turn RL data
│   ├── mt_rl.json             # Multi-turn RL data
│   ├── test.json              # Test data
│   ├── icon_captioning.json   # Icon captioning data
│   └── icon_grounding.json    # Icon grounding data
├── eval/
│   ├── inference_qwen2p5_mixed_vllm.py  # Prediction generation
│   └── calculate_score_refine.py        # Score calculation
├── scripts/
│   └── generate_sft_data.py   # Dataset generation script
├── swift/                      # Training framework
├── Progress.md                 # Detailed progress log
└── Guide.md                    # This file
```

## Quick Resume Commands

```bash
# Activate environment
conda activate gelab

# Set all environment variables
export HF_HOME=/ext_hdd2/nhkoh/.cache/huggingface
export TRANSFORMERS_CACHE=/ext_hdd2/nhkoh/.cache/huggingface
export MODELSCOPE_CACHE=/ext_hdd2/nhkoh/.cache/modelscope
export HF_DATASETS_CACHE=/ext_hdd2/nhkoh/.cache/huggingface/datasets
export TORCH_HOME=/ext_hdd2/nhkoh/.cache/torch
export WANDB_API_KEY=<WANDB_API_KEY>
export WANDB_ENTITY=namhokoh-korea-advanced-institute-of-science-and-technology
export WANDB_PROJECT=gelab-fresh
export CUDA_VISIBLE_DEVICES=0,1,2

# Navigate to project
cd /ext_hdd2/nhkoh/gelab-engine

# Check training status
tmux attach -t gelab-sft
```
