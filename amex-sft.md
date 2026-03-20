# AMEX SFT Data Generation and Training Guide

## Overview

This guide covers how to generate SFT training data from AMEX trajectories and train a Qwen2.5-VL-7B-Instruct model on it. The pipeline has two stages:

1. **Trajectory Composition**: Run OmniParser (YOLO + OCR) on AMEX screenshots, compose GE-Lab pages with GPT-5-Mini styling, and build UI structure files.
2. **SFT Data Generation**: Convert the composed trajectories into training samples with the AMEX unified action space.

## Prerequisites

```bash
conda activate gelab
export OPENAI_API_KEY="sk-..."  # Required for GPT-5-Mini page styling
export WANDB_API_KEY="..."
export WANDB_ENTITY="namhokoh-korea-advanced-institute-of-science-and-technology"
export WANDB_PROJECT="gelab"
export HF_TOKEN="..."
export HF_HOME="/ext_hdd2/nhkoh/.cache/huggingface"
export XDG_CACHE_HOME="/ext_hdd2/nhkoh/.cache"
export TORCH_HOME="/ext_hdd2/nhkoh/.cache/torch"
export CUDA_HOME=/ext_hdd2/nhkoh/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

OmniParser weights must be at `/ext_hdd2/nhkoh/OmniParser/weights/`.

## Stage 1: Generate Trajectories

### Single Trajectory

```bash
python data_engine/amex_sim2real_compose_action_coord.py \
    --trajectory_id 2024_3_18_17_19_e8ba0101cbc74242b48af70a57dafdf5 \
    --output_dir data_engine/sim2real_envs/amex_sft \
    --gpu 0
```

This produces:
- `pages/` -- Composed page PNGs (1080x2400)
- `extracted_assets/` -- OmniParser-detected icon crops per step
- `generated_code/` -- GPT-generated PIL styling code per page
- `action_coord/` -- Debug overlay images showing action coordinates
- `ui_structure.json` -- Flat page graph with transitions
- `ui_structure_layer.json` -- Tree/hierarchy view
- `trajectory_assets_manifest.json` -- All extracted assets metadata

Runtime: ~15 min per trajectory on 1x A100.

### All Trajectories (Batch)

```bash
python data_engine/amex_sim2real_compose_action_coord.py \
    --output_dir data_engine/sim2real_envs/amex_sft \
    --gpu 0
```

Add `--max_trajectories N` to limit. Estimated: ~20 GPU-days for all 3,046 trajectories on 1x A100.

### Inspect Output

Check the composed pages visually:
```bash
ls data_engine/sim2real_envs/amex_sft/pages/
```

Check the action coordinate overlays:
```bash
ls data_engine/sim2real_envs/amex_sft/action_coord/
```

## Stage 2: Generate SFT Data

After composing trajectories, generate training samples:

```bash
python data_engine/generate_amex_sft_data.py \
    --trajectory_id 2024_3_18_17_19_e8ba0101cbc74242b48af70a57dafdf5 \
    --output_dir data_engine/sim2real_envs/amex_sft \
    --crop_mode deterministic_compose
```

Or run the SFT generation directly on an already-composed environment:

```python
import json, sys
sys.path.insert(0, "data_engine")
from generate_amex_sft_data import (
    generate_trajectory_samples,
    generate_grounding_samples,
    generate_captioning_samples,
)

# Load composed environment
with open("data_engine/sim2real_envs/amex_sft/ui_structure.json") as f:
    ui = json.load(f)
with open("/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno/<trajectory>.json") as f:
    traj = json.load(f)

pages_dir = "data_engine/sim2real_envs/amex_sft/pages"
nav = generate_trajectory_samples(ui, traj, pages_dir, "amex_source")
grounding = generate_grounding_samples(ui, pages_dir, "amex_source", 20)
captioning = generate_captioning_samples(ui, pages_dir, "amex_source", 20)

all_samples = nav + grounding + captioning
with open("sft_amex.json", "w") as f:
    json.dump(all_samples, f, indent=2)
```

### AMEX Unified Action Space

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

Coordinates are normalized to a (0,0) top-left, (1000,1000) bottom-right system.

## Stage 3: Train

```bash
# Set environment variables (see Prerequisites above)
bash gui_scripts/sft_amex.sh
```

Or manually:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
MAX_PIXELS=200704 \
swift sft \
    --model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset "data_engine/sim2real_envs/amex_sft/sft_amex.json" \
    --max_length 2048 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --deepspeed zero2 \
    --output_dir ./checkpoint/gui_exp/sft_amex \
    --max_pixels 200704 \
    --report_to wandb
```

## Data Sources

| Resource | Path |
|----------|------|
| AMEX annotations | `/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno/` |
| AMEX screenshots | `/ext_hdd2/tsyou/AMEX_dataset/AMEX/screenshot/` |
| AMEX element annotations | `/ext_hdd2/tsyou/AMEX_dataset/AMEX/element_anno/` |
| OmniParser weights | `/ext_hdd2/nhkoh/OmniParser/weights/` |
| Pre-extracted icons (ttran) | `/ext_hdd2/ttran/datasets/AMEX_dataset/ui_elements_output_clean/elements/` |

## Branch

All code is on the `koh-dev/amex-sft` branch.

## Key Files

| File | Purpose |
|------|---------|
| `data_engine/amex_sim2real_compose_action_coord.py` | OmniParser detection + GPT compose + UI structure builder |
| `data_engine/generate_amex_sft_data.py` | SFT training data generator (AMEX unified action space) |
| `data_engine/amex_compose_deterministic.py` | Deterministic compose alternative (no GPT needed) |
| `gui_scripts/sft_amex.sh` | Training script with AMEX system prompt |
| `data_engine/env_utils.py` | GE-Lab environment utilities (bbox normalization, graph) |

## Bug Fixes Applied

1. **math module in sandbox**: GPT-generated code sometimes uses `math.sqrt()` etc. Added `math` to the exec namespace in `render_from_code()`.
2. **Fallback compose from asset_path**: When GPT code fails and falls back, `_fallback_compose()` now loads crops from `asset_path` on disk instead of requiring in-memory `crop` objects.
