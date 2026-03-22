# AMEX SFT Training Guide

## Overview

This guide covers how to merge SFT datasets from multiple team members and run SFT training on the combined data. Each team member generated composed trajectories using `amex_sim2real_compose_action_coord.py` and SFT samples using `collect_amex_sft.py`.

## Step 1: Locate SFT Data

Each team member's composed environments and SFT data are at:

| Member | Env directory | SFT JSON |
|--------|--------------|----------|
| dematsunaga | `/ext_hdd2/dematsunaga/amex_sft/` | `/ext_hdd2/dematsunaga/amex_sft/sft_amex.json` |
| mschoenenberger | `/home/mschoenenberger/gui/gelab_amex_sft/gelab-env/data_engine/sim2real_envs/amex_sft/` | `.../sft_amex.json` |
| tsyou | `/ext_hdd2/tsyou/gelab-env/data_engine/sim2real_envs/amex_sft/` | `.../sft_amex.json` |
| nhkoh | `/ext_hdd2/nhkoh/gelab-env/data_engine/sim2real_envs/amex_sft/` | `.../sft_amex.json` |

If a member hasn't generated their SFT JSON yet, run:
```bash
cd /ext_hdd2/nhkoh/gelab-env
conda activate gelab
python data_engine/collect_amex_sft.py --env_dir <their_env_directory>
```

## Step 2: Merge All SFT Data

Run the merge script to combine all team members' SFT JSONs into a single training file.
All image paths are absolute, so they work from any working directory as long as the
original directories are accessible.

```bash
conda activate gelab
python data_engine/merge_amex_sft.py \
    --inputs \
        /ext_hdd2/dematsunaga/amex_sft/sft_amex.json \
        "/home/mschoenenberger/gui/gelab_amex_sft/gelab-env/data_engine/sim2real_envs/amex_sft/sft_amex.json" \
        /ext_hdd2/tsyou/gelab-env/data_engine/sim2real_envs/amex_sft/sft_amex.json \
        /ext_hdd2/nhkoh/gelab-env/data_engine/sim2real_envs/amex_sft/sft_amex.json \
    --output datas_amex/sft_amex_combined.json
```

This will:
- Load all SFT JSONs
- Deduplicate by source trajectory ID
- Validate that all image paths exist
- Re-index and save as a single JSON

## Step 3: Train

### Prerequisites

```bash
conda activate gelab
export WANDB_API_KEY="..."
export WANDB_ENTITY="namhokoh-korea-advanced-institute-of-science-and-technology"
export WANDB_PROJECT="gelab"
export HF_TOKEN="..."
export HF_HOME="/path/to/.cache/huggingface"
export XDG_CACHE_HOME="/path/to/.cache"
export TORCH_HOME="/path/to/.cache/torch"
export CUDA_HOME=/path/to/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Option A: 3x A100 80GB (sft_amex.sh)

```bash
export DATASET_PATH="datas_amex/sft_amex_combined.json"
bash gui_scripts/sft_amex.sh
```

Default config: 3 GPUs, ZeRO-2, batch=2, grad_accum=4, lr=1e-5, 1 epoch.

### Option B: 1x H200 141GB

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset "datas_amex/sft_amex_combined.json" \
    --max_length 2048 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --save_steps 500 \
    --save_total_limit 2 \
    --save_only_model true \
    --logging_steps 10 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --output_dir ./checkpoint/gui_exp/sft_amex \
    --max_pixels 200704 \
    --report_to wandb \
    --add_version False
```

No DeepSpeed needed -- single H200 has enough VRAM (141 GB).

### Option C: 1x A100 80GB (tight fit)

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset "datas_amex/sft_amex_combined.json" \
    --max_length 2048 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 24 \
    --gradient_checkpointing true \
    --save_steps 500 \
    --save_total_limit 2 \
    --save_only_model true \
    --logging_steps 10 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --output_dir ./checkpoint/gui_exp/sft_amex \
    --max_pixels 200704 \
    --report_to wandb \
    --add_version False
```

Requires gradient checkpointing and batch_size=1 to fit in 80 GB.

## Estimated Training Times (1 epoch)

| Dataset size | 1x H200 | 3x A100 | 1x A100 |
|-------------|---------|---------|---------|
| ~10K samples | ~6 min | ~12 min | ~30 min |
| ~50K samples | ~30 min | ~50 min | ~2.5 hours |
| ~160K samples | ~1.1 hours | ~2.8 hours | ~8 hours |

## AMEX Unified Action Space

The SFT data uses all 8 AMEX action types:

| Action | Format | Frequency |
|--------|--------|-----------|
| TAP | `tap(start_box='<\|box_start\|>(x,y)<\|box_end\|>')` | 64.1% |
| SWIPE | `swipe(start_box=..., end_box=...)` | 19.7% |
| TASK_COMPLETE | `complete` | 7.3% |
| TYPE | `type(start_box=..., text='...')` | 6.2% |
| PRESS_ENTER | `press_enter()` | 1.7% |
| TASK_IMPOSSIBLE | `impossible` | 0.6% |
| PRESS_BACK | `press_back()` | 0.3% |
| PRESS_HOME | `press_home()` | 0.0% |

## Key Files

| File | Purpose |
|------|---------|
| `data_engine/collect_amex_sft.py` | Generate SFT data from composed trajectories (per-member) |
| `data_engine/merge_amex_sft.py` | Merge multiple SFT JSONs into one training file |
| `data_engine/generate_amex_sft_data.py` | Core SFT sample generation functions |
| `gui_scripts/sft_amex.sh` | Training script (3x A100 config) |

## Branch

All code is on `koh-dev/amex-sft`.
