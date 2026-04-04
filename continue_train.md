# Continue-Train Pipeline (Paper Section 6.2)

Continual training on real-world GUI data to improve grounding on benchmarks like ScreenSpot, MoTIF, Refexp, and VWB.

## Overview

The pipeline has three steps:
1. Download and format 21k real-world samples from 4 datasets (AITW, AITZ, AMEX, Mind2Web)
2. Run SFT continue-training on a base or fine-tuned checkpoint
3. Evaluate on 7 static grounding benchmarks

## Step 1: Prepare Training Data

```bash
conda activate gelab
pip install datasets  # if not installed

python data_engine/prepare_continue_train_data.py \
    --output datas/continue_train_24k.json \
    --image_dir datas/real_world_images \
    --total_samples 24000 \
    --sources aitw aitz amex mind2web
```

**Data sources:**
- **AITW** (6k): Downloaded automatically from `cjfcsjt/AITW_Single` on HuggingFace
- **AITZ** (3.3k): Downloaded automatically from `xwm/AITZ` on HuggingFace (only 3,337 valid samples available after filtering)
- **AMEX** (6k): Loaded from local files at `/ext_hdd2/tsyou/AMEX_dataset/AMEX/`
- **Mind2Web** (6k): Downloaded automatically from `osunlp/Multimodal-Mind2Web` on HuggingFace

Total: ~21k samples (paper uses 24k). Images are saved to `datas/real_world_images/`.

**Runtime:** ~1 hour (dominated by AITW and Mind2Web image saving).

**Action distribution in the output:**
- click: ~72%
- complete: ~12%
- type: ~11%
- scroll: ~5%

All coordinates are in 0-1000 normalized space. Output format is ms-swift compatible (messages + images).

## Step 2: Run Training

```bash
# Base Qwen2.5-VL-7B continue-train (paper Table 5 "Continue-Train" row)
MODEL_STAGE=base bash gui_scripts/continue_train_448.sh

# SFT checkpoint continue-train (paper Table 5 "SFT-Continue-Train" row)
MODEL_STAGE=sft bash gui_scripts/continue_train_448.sh

# SFT from HuggingFace
MODEL_STAGE=sft_hf bash gui_scripts/continue_train_448.sh
```

**Hyperparameters (Paper Appendix A.5):**
| Parameter | Paper | Ours |
|-----------|-------|------|
| Learning rate | 1e-5 | 1e-5 |
| Epochs | 2 | 2 |
| Global batch size | 256 | 258 (3 GPUs x 1 x 86) |
| Max length | 5120 | 5120 |
| Max pixels | 200704 | 1003520 |
| GPUs | 16 | 3 x A100 80GB |
| DeepSpeed | - | ZeRO-3 + gradient checkpointing |

**Note on MAX_PIXELS:** The paper uses 200704 (448x448 equivalent). We use 1003520 (Qwen2.5-VL-7B default) because training at low resolution causes coordinate space mismatch during evaluation — the model learns coordinates relative to the downscaled image, but benchmarks expect coordinates relative to the original resolution.

**Runtime:** ~24 hours on 3x A100 80GB (164 steps, ~9 min/step).

**WandB:** Logs to project `gelab` under entity `namhokoh-korea-advanced-institute-of-science-and-technology`.

## Step 3: Evaluate

```bash
# Base Qwen2.5-VL (no training) — sanity check, should get ~75-85% on ScreenSpot
python eval/evaluate_real_world.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --base_model \
    --num_gpus 1

# Continue-train checkpoint — use --base_model for bbox_2d format
python eval/evaluate_real_world.py \
    --model_path checkpoint/gui_exp/continue_train_448_base/<run>/checkpoint-164 \
    --base_model \
    --num_gpus 1

# Continue-train checkpoint — use --use_system_prompt for click(start_box) format
python eval/evaluate_real_world.py \
    --model_path checkpoint/gui_exp/continue_train_448_base/<run>/checkpoint-164 \
    --use_system_prompt \
    --num_gpus 1
```

**Eval flags:**
| Flag | When to use | Coordinate format |
|------|------------|-------------------|
| `--base_model` | Base Qwen2.5-VL or models that output native bbox_2d | Pixel coords, normalized by image size |
| `--use_system_prompt` | Continue-trained models that output click(start_box) | 0-1000 normalized |
| (neither) | Default, uses grounding prompt without system prompt | Parses <points>, bbox_2d, or (x,y) |

**Benchmarks evaluated (6 of 7 from Paper Table 5):**
- ScreenSpot, ScreenSpot-v2, MoTIF, Refexp, VWB-EG, VWB-AG
- FuncPred requires manual download from AutoGUI repo (not on HuggingFace)
- AndroidWorld requires Android emulator (not included)

## Known Issues

**Coordinate space mismatch:** If `MAX_PIXELS` differs between training and evaluation, the model's coordinate predictions will be in the wrong scale. Always evaluate at the same resolution used during training, or train at full resolution (`MAX_PIXELS=1003520`).

**AITZ sample count:** AITZ only has 3,337 valid samples after filtering (requested 6k), resulting in 21k total instead of 24k. To compensate, increase AITW or AMEX allocation:
```bash
python data_engine/prepare_continue_train_data.py \
    --sources aitw amex mind2web \
    --total_samples 24000
```

## File Locations

| File | Purpose |
|------|---------|
| `data_engine/prepare_continue_train_data.py` | Download and format training data |
| `gui_scripts/continue_train_448.sh` | Training script with paper-aligned hyperparams |
| `eval/evaluate_real_world.py` | Grounding benchmark evaluation |
| `datas/continue_train_24k.json` | Training data (generated by step 1) |
| `datas/real_world_images/` | Training images (generated by step 1) |
| `checkpoint/gui_exp/continue_train_448_base/` | Training checkpoints |
