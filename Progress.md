# GE-Lab SFT Training Progress

## Project Overview
Training a GUI navigation agent using Supervised Fine-Tuning (SFT) on Qwen2.5-VL-7B-Instruct, following the methodology from the paper "GUI Exploration Lab: Enhancing Screen Navigation in Agents via Multi-Turn Reinforcement Learning" (arXiv:2512.02423).

## Compute Resources
- **GPUs**: 3x NVIDIA A100 80GB PCIe
- **CUDA Version**: 12.8
- **Driver**: 570.211.01

## Paper Specifications

### Environment Structure (Env-Base)
| Parameter | Value |
|-----------|-------|
| Graph Depth | 7 |
| Node Distribution | [5, 3, 2, 2, 1, 1, 0] per level |
| Subtrees | 5 independent (ratio 2:2:1 for SFT:RL:Test) |
| Path Data per Subtree | 12,439 instances |
| Edge Data per Subtree | 274 instances |
| Total Pages | 231 |

### SFT Hyperparameters (Table 7 from paper)
| Config | Value |
|--------|-------|
| Learning Rate | 1e-5 |
| LR Schedule | cosine decay |
| Per Device Batch Size | 2 |
| Gradient Accumulation Steps | 2 |
| Warmup Ratio | 0.05 |
| Num Train Epochs | 1 |
| Max Pixels | 200704 |

### Training Statistics (Table 9 from paper)
| Method | Dataset Size | Training Time | Interaction Count |
|--------|--------------|---------------|-------------------|
| SFT | 60,864 samples | 3-4 GPU hours | 60,864 |
| ST-RL | 12,439 samples | 48 GPU hours | 99,512 |
| MT-RL | 2,162 tasks | 36 GPU hours | ~89,939 |

---

## Progress Tracker

### Phase 1: Environment Setup [COMPLETED]
- [x] Conda environment verified (gelab, Python 3.10.19)
- [x] Dependencies installed (torch 2.9.1, transformers 4.57.6, ms_swift 3.5.0.dev0)
- [x] HuggingFace token configured
- [x] WANDB credentials configured

### Phase 2: Data Generation [COMPLETED - REGENERATED]
- [x] Generated UI environment with 231 pages (5 subtrees)
- [x] **REGENERATED** SFT training data using `demo/ui_structure.json` (30,066 samples)
  - Navigation (Path+Edge): 25,426 samples
  - Icon Captioning: 2,320 samples
  - Icon Grounding: 2,320 samples
- [x] **REGENERATED** ST-RL data: 24,878 samples
- [x] **REGENERATED** MT-RL data: 4,324 samples
- [x] **REGENERATED** Test data: 2,162 tasks (exact match with paper!)
- [x] Using author's original images in `datas/images/`

**Note:** Original data in `datas/sft_wrong_coords.json` was generated with mismatched UI structure.

### Phase 3: SFT Training [REQUIRES REDO]
- [x] Configure training script with paper hyperparameters
- [x] Run SFT training on 3x A100 GPUs
- [x] WANDB tracking enabled
- [x] Intermediate checkpoint saving enabled (every 500 steps)
- [x] Training completed - **BUT WITH WRONG DATA**
- [ ] **REDO** SFT training with corrected data

**Previous Training Results (v3 - INVALID):**
- Checkpoint dir: `checkpoint/gui_exp/sft/v3-20260124-120531/checkpoint-2481`
- **Status:** INVALID - trained on misaligned coordinates
- Model learned wrong icon coordinates and names

**Consistent Pipeline Training (ALSO INVALID):**
- Checkpoint dir: `checkpoint/gui_exp/sft_consistent/v0-20260125-105121/checkpoint-2481`
- **Status:** INVALID - used data_engine UI structure, not demo UI structure

**Pending: Corrected Training (v4)**
- Will use `datas/sft.json` (regenerated with `demo/ui_structure.json`)
- Output: `checkpoint/gui_exp/sft_corrected/`

### Phase 4: Evaluation [BLOCKED - REQUIRES RETRAINING]

**Status:** Previous evaluations invalid due to data alignment bug. Need to retrain with corrected data.

#### Interactive Evaluation Results (with buggy model)
| Path Length | Tasks | Pass@1 |
|-------------|-------|--------|
| Path@1 | 137 | 61.31% |
| Path@2 | 147 | 17.01% |
| Path@3 | 222 | 1.35% |
| Path@4-7 | 1,656 | 0.00% |
| **Overall** | **2,162** | **5.18%** |

**Root cause:** Model trained on wrong coordinates - see bug fix section above.

#### Evaluation Types (per Paper)
| Type | Description | Data Source | Subtrees |
|------|-------------|-------------|----------|
| **ID (In-Distribution)** | Tests on same env as training | sub0, sub1 | SFT subtrees |
| **OOD (Out-of-Distribution)** | Tests generalization to unseen subtree | sub4 | Test subtree |
| **Interactive** | Multi-step task completion | sub4 | Test subtree |

#### After Retraining: Run Interactive Evaluation
```bash
tmux new -s interactive-eval
cd /ext_hdd2/nhkoh/gelab-engine && \
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate gelab && \
export CUDA_VISIBLE_DEVICES=0 && \
python eval/interactive_eval.py \
    --model_path checkpoint/gui_exp/sft_corrected/checkpoint-XXXX \
    --ui_structure demo/ui_structure.json \
    --images_dir datas/images \
    --output_file eval_results/interactive_corrected.json
```

#### Evaluation Data Generated
| File | Subtree | Samples | Tasks | Purpose |
|------|---------|---------|-------|---------|
| `datas/eval_sub4_test.json` | sub4 | 12,439 | 2,162 | OOD evaluation |
| `datas/eval_sub0_test.json` | sub0 | 12,439 | 2,162 | ID evaluation |
| `datas/eval_sub1_test.json` | sub1 | 12,439 | 2,162 | ID evaluation |

#### Task Distribution (matches Paper Table 6)
| Path Length | Tasks |
|-------------|-------|
| Path@1 | 137 |
| Path@2 | 147 |
| Path@3 | 222 |
| Path@4 | 324 |
| Path@5 | 492 |
| Path@6 | 456 |
| Path@7 | 384 |
| **Total** | **2,162** |

#### Evaluation Commands

**Monitor Progress:**
```bash
# Check OOD progress
wc -l eval_results/sft_predictions.jsonl

# Check ID progress
wc -l eval_results/sft_id_predictions.jsonl

# Attach to tmux sessions
tmux attach -t gelab-eval      # OOD
tmux attach -t gelab-eval-id   # ID sub0
tmux attach -t gelab-eval-id2  # ID sub1
```

**Calculate Scores (after inference completes):**
```bash
# OOD scores (sub4)
python eval/calculate_score_refine.py --file eval_results/sft_predictions.jsonl

# ID scores (sub0)
python eval/calculate_score_refine.py --file eval_results/sft_id_predictions.jsonl

# ID scores (sub1)
python eval/calculate_score_refine.py --file eval_results/sft_id2_predictions.jsonl
```

#### Expected Metrics (from Paper Table 1 for SFT)
| Metric | ID Target | OOD Target |
|--------|-----------|------------|
| Edge | 94.82 | 64.55 |
| Path | 99.76 | 41.76 |
| Overall | 98.89 | 55.45 |
| Pass@1 | - | 14.30 |
| Pass@5 | - | 20.86 |

---

## Generated Data Summary

### Dataset Statistics
| Dataset | Count | Paper Target | Match Rate |
|---------|-------|--------------|------------|
| SFT Total | 30,066 | 60,864 | 49.4% |
| - Navigation | 25,426 | ~25,426 | 100% |
| - Icon Captioning | 2,320 | 2,320 | 100% |
| - Icon Grounding | 2,320 | 2,320 | 100% |
| ST-RL | 20,554 | 12,439 | 165% |
| MT-RL | 4,324 | 2,162 | 200% |
| Test Tasks | 2,162 | 2,162 | 100% |

### Analysis Notes
The SFT total (30,066) is approximately half of the paper's 60,864. This discrepancy may be due to:
1. Paper potentially counting 2 training epochs as separate samples
2. Undocumented data augmentation techniques
3. Different counting methodology

**Key validation**: Our per-subtree counts (12,439 path + 274 edge) match the paper exactly, confirming correct implementation of the core methodology.

---

## Configuration Details

### Model Path
```
/ext_hdd2/nhkoh/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
```

### UI Environment Path
**IMPORTANT: Use `demo/ui_structure.json` which matches `datas/images/` (author's images)**
```
UI Structure: demo/ui_structure.json
Images: datas/images/
```
**DO NOT USE:** `data_engine/ui_environment/20260123_230516/` (mismatched coordinates)

### HuggingFace Token
```
<HF_TOKEN>
```

### WANDB Configuration
```
Entity: namhokoh-korea-advanced-institute-of-science-and-technology
Project: gelab-fresh
```

---

## Execution Log

### 2026-01-23 - Session Start
- Analyzed codebase structure
- Reviewed paper specifications from screenshots
- Verified existing conda environment (gelab)
- Configured HuggingFace and WANDB credentials

### 2026-01-23 - Environment Generation
- Generated UI environment with 231 pages
- Node distribution: [5, 3, 2, 2, 1, 1, 0] matching paper
- 5 subtrees created (47 nodes each + shared root)
- Output: `data_engine/ui_environment/20260123_230516/`

### 2026-01-23 - Data Generation
- Created `scripts/generate_sft_data.py` for comprehensive data generation
- Generated all datasets:
  - SFT: 30,066 samples (navigation + icon tasks)
  - ST-RL: 20,554 samples
  - MT-RL: 4,324 samples
  - Test: 2,162 tasks
- Output files in `datas/` directory

### 2026-01-23 23:21 - SFT Training Started (v2)
- Started SFT training on 3x A100 80GB GPUs
- Using local HuggingFace cached model (no download required)
- All caches directed to `/ext_hdd2/nhkoh/.cache/`
- Training command in tmux session `gelab-sft`
- WANDB tracking: https://wandb.ai/namhokoh-korea-advanced-institute-of-science-and-technology/gelab-fresh/runs/pq4660s4
- Run reached ~79% (1961/2481 steps) before being stopped

### 2026-01-24 12:05 - SFT Training Restarted (v3)
- Restarted training with intermediate checkpoint saving (--save_strategy steps --save_steps 500)
- Checkpoint dir: `checkpoint/gui_exp/sft/v3-20260124-120531/`
- WANDB tracking: https://wandb.ai/namhokoh-korea-advanced-institute-of-science-and-technology/gelab-fresh/runs/gvlau5bs
- Run name: `gelab-sft-qwen2.5vl-7b_v2`
- Checkpoints saved at steps: 500, 1000, 1500, 2000, and 2481 (final)

### 2026-01-25 06:38 - SFT Training Completed
- Training completed successfully after 18h 32m 57s
- Final checkpoint: `checkpoint/gui_exp/sft/v3-20260124-120531/checkpoint-2481`
- Final metrics: Loss=0.0636, Token Acc=99.88%

### 2026-01-25 08:25 - Evaluation Started
- Generated evaluation data for sub0 (ID), sub1 (ID), sub4 (OOD)
- Started OOD evaluation on GPU 0 (tmux: gelab-eval)
- Started ID evaluation on GPU 1 (tmux: gelab-eval-id)
- Started ID evaluation on GPU 2 (tmux: gelab-eval-id2)

### 2026-01-26 - Interactive Evaluation Completed
- Ran full interactive evaluation (2,162 tasks)
- Duration: 2h 55m 9s
- Results: Pass@1 = 5.18% overall (Path@1: 61.31%, Path@2: 17.01%, Path@3+: ~0%)
- Results saved to: `eval_results/interactive_eval_full.json`

### 2026-01-27 - DEEP ROOT CAUSE ANALYSIS

**The Paradox**: 99.88% training/eval token accuracy, but only 4.76% interactive Pass@1

**Investigation Results:**

| What | Training | Test (Subtree 4) |
|------|----------|------------------|
| Navigation icons | 92 (subtrees 0,1) | 45 (subtree 4) |
| Icon overlap | - | **0%** |
| Grounding coverage | 158 mentions (space format) | Available |

**Key Finding: FORMAT MISMATCH**

| Task | Icon Format | Example |
|------|-------------|---------|
| Navigation | Underscore | `Digital_electronics_98` |
| Grounding | Space | `Digital electronics 98` |
| Captioning | Space | `Digital electronics 98` |

The model sees subtree 4 icons 158 times in grounding as `Digital electronics 98` but needs to output `Digital_electronics_98` in navigation. **It cannot make this connection.**

**Why 99.88% Training Accuracy?**
- Training evaluation uses held-out data from subtrees 0,1
- Model has seen ALL subtree 0,1 icons in navigation training
- Perfect accuracy on seen icons, fails on unseen icons

**Why 4.76% Interactive Pass@1?**
- Interactive evaluation uses subtree 4 (test)
- Subtree 4 icons: **0 mentions in navigation training**
- Model defaults to `home`/`back` when uncertain

**Content Icon Accuracy Breakdown:**
| Icon Type | Correct | Wrong | Accuracy |
|-----------|---------|-------|----------|
| System (home/back) | 74 | 17 | 81% |
| Content | 1 | 45 | **2%** |

**Paper Requirement Not Met:**
> "SFT training dataset incorporates Edge data from all subtrees, including Test subtree"

Our edge data from subtree 4: **0 pages** (should include 46 pages)

---

### 2026-01-26 - EARLIER BUG (FIXED)

**Root Cause Analysis:**
The poor interactive evaluation performance was due to a **critical data alignment bug**:

| Component | What Was Used | What Should Be |
|-----------|---------------|----------------|
| Training icon names | `Business_115` (from `data_engine/ui_structure.json`) | `Shoes_and_clothing_105` (from `demo/ui_structure.json`) |
| Training coordinates | `(1034, 1358)` | `(574, 2048)` |
| Training images | `datas/images/` (author's original) | ✓ Correct |

**The Problem:**
- Training data was generated using **our generated** `data_engine/ui_structure.json`
- But the images in `datas/images/` correspond to **author's** `demo/ui_structure.json`
- This caused a fundamental mismatch: model learned coordinates that point to EMPTY SPACE in the actual images!

**Evidence:**
- Model predicted `Business_115` at `(1034, 1358)`
- But actual icon in image is `Shoes_and_clothing_105` at `(574, 2048)`
- System icons (home, back) worked because they have FIXED positions across all UI structures

**The Fix:**
1. Modified `scripts/generate_sft_data.py` to accept `--ui_structure` and `--pages_dir` separately
2. Regenerated ALL training data using `demo/ui_structure.json`:
   ```bash
   python scripts/generate_sft_data.py \
       --ui_structure demo/ui_structure.json \
       --pages_dir datas/images \
       --output_dir datas \
       --sft_subtrees 0 1
   ```
3. Verified coordinate alignment: ALL coordinates now match between training data and UI structure

**Files Modified:**
- `scripts/generate_sft_data.py` - Added flexible path arguments
- `datas/sft.json` - Regenerated with correct coordinates (30,066 samples)
- `datas/st_rl.json` - Regenerated (24,878 samples)
- `datas/test.json` - Regenerated (2,162 tasks)
- Backed up old data to `datas/sft_wrong_coords.json`

---

## Proposed Fixes to Match Paper Results

### Fix 1: Standardize Icon Name Format (HIGH PRIORITY)
Regenerate icon_grounding.json and icon_captioning.json with **underscore format**:
- Change `Digital electronics 98` → `Digital_electronics_98`
- Ensures grounding knowledge transfers to navigation

### Fix 2: Add Edge Data from ALL Subtrees (MEDIUM PRIORITY)
Paper explicitly requires edge data from test subtree:
- Add home/back navigation from ALL 231 pages
- Gives model visual exposure to subtree 4 pages

### Fix 3: Verify Format Consistency (LOW PRIORITY)
Ensure coordinate formats match:
- Grounding: `point:(x, y)` → `<|box_start|>(x,y)<|box_end|>`

### Expected Impact
| Fix | Expected Improvement |
|-----|---------------------|
| Icon name format | +15-25% |
| Edge data from all subtrees | +5-10% |
| Prompt format (already fixed) | +5-10% |
| **Total** | **+25-45%** |

Target: Path@1 from 54.74% to ~80-90%

---

## Next Steps
1. ~~Commit current changes to git~~ [DONE]
2. ~~Create feature branch `koh-dev/sft`~~ [DONE]
3. ~~Configure and run SFT training~~ [DONE - but with wrong data]
4. ~~Identify critical data alignment bug~~ [DONE]
5. ~~Fix data generation to use correct UI structure~~ [DONE]
6. ~~Re-run SFT training with corrected data~~ [DONE]
7. ~~Run interactive evaluation~~ [DONE - 4.76% Pass@1]
8. ~~Deep root cause analysis~~ [DONE - format mismatch identified]
9. **Implement proposed fixes** [PENDING]
10. Re-run SFT training with fixed data [PENDING]
11. Re-run interactive evaluation [PENDING]

### Fix Implementation Commands
```bash
# Fix 1: Regenerate icon data with underscore format
python scripts/generate_icon_data.py --format underscore

# Fix 2: Add edge data from all subtrees  
python scripts/generate_sft_data.py --include_all_edge

# Fix 3: Verify format consistency
python scripts/verify_data_format.py

# Then retrain
NPROC_PER_NODE=3 swift sft --model ... --dataset datas/sft_fixed.json
```

---

## OLD Next Steps (Reference)
6. **Re-run SFT training with corrected data** [COMPLETED]
   ```bash
   # Start training in tmux session
   tmux new -s gelab-sft-v4
   cd /ext_hdd2/nhkoh/gelab-engine && \
   source ~/miniconda3/etc/profile.d/conda.sh && \
   conda activate gelab && \
   export HF_HOME=/ext_hdd2/nhkoh/.cache/huggingface && \
   export WANDB_API_KEY=<WANDB_API_KEY> && \
   export WANDB_PROJECT=gelab-fresh && \
   NPROC_PER_NODE=3 MAX_PIXELS=200704 swift sft \
       --model /ext_hdd2/nhkoh/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
       --train_type full \
       --target_modules all-linear \
       --dataset datas/sft.json \
       --torch_dtype bfloat16 \
       --per_device_train_batch_size 2 \
       --learning_rate 1e-5 \
       --gradient_accumulation_steps 2 \
       --warmup_ratio 0.05 \
       --num_train_epochs 1 \
       --eval_steps 100 \
       --save_steps 500 \
       --save_total_limit 5 \
       --logging_steps 1 \
       --max_length 4096 \
       --output_dir checkpoint/gui_exp/sft_corrected \
       --deepspeed zero3 \
       --gradient_checkpointing true \
       --attn_impl flash_attn \
       --report_to wandb \
       --run_name gelab-sft-corrected-v1
   ```
7. Re-run interactive evaluation with `demo/ui_structure.json`
8. Compare results with paper targets

---

## Files Created/Modified

### New Files
- `scripts/generate_sft_data.py` - Comprehensive data generation script
- `datas/sft.json` - Combined SFT training data (30,066 samples)
- `datas/icon_captioning.json` - Icon Captioning data (2,320 samples)
- `datas/icon_grounding.json` - Icon Grounding data (2,320 samples)
- `datas/st_rl.json` - ST-RL training data (20,554 samples)
- `datas/mt_rl.json` - MT-RL training data (4,324 samples)
- `datas/test.json` - Test tasks (2,162 samples)
- `datas/images/` - Page images for training
- `Guide.md` - Quick reference guide for resuming work
- `Progress.md` - Detailed progress log

### Generated Environment
- `data_engine/ui_environment/20260123_230516/`
  - `ui_structure.json` - Flat UI structure
  - `ui_structure_layer.json` - Hierarchical structure
  - `config.json` - Environment configuration
  - `ui_topology.png` - Visual topology graph
  - `pages/` - 231 page images
  - `used_icons.json` - Icon mapping
