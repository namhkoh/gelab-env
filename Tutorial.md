# GE-Lab Tutorial

**Paper**: [GUI Exploration Lab: Enhancing Screen Navigation in Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/pdf/2512.02423)

---

## 1. Codebase Structure

```
gelab-env/
├── data_engine/                    # UI environment & dataset generation
│   ├── tree.py                     # Core: synthetic tree-structured UI generator
│   ├── env_utils.py                # Shared utilities (GELabEnvUtils base class)
│   ├── generate_sft_data.py        # SFT dataset generation (path+edge+grounding+caption)
│   ├── generate_st_rl_data.py      # ST-RL dataset generation (path-only, subtrees 2-3)
│   ├── generate_mt_rl_data.py      # MT-RL task generation (balanced by path length)
│   ├── prepare_continue_train_data.py  # Real-world data prep (AITW + Mind2Web -> 24k)
│   └── icons/                      # Icon pool (Animals/, Business/, etc.)
│
├── swift/plugin/                   # Training plugins (modified ms-swift framework)
│   ├── multi_turn.py               # Multi-turn environment for MT-RL rollouts
│   └── orm.py                      # Reward functions (ST-RL 4-component + A2B sparse)
│
├── eval/                           # Evaluation scripts
│   ├── evaluate.py                 # Unified evaluation (static + interactive)
│   ├── evaluate_real_world.py      # Real-world grounding benchmarks (7 datasets)
│   └── generate_eval_splits.py     # Generate ID/OOD Edge/Path test splits
│
├── gui_scripts/                    # Training launch scripts (3x A100 80GB)
│   ├── sft_448.sh                  # SFT training (Paper Table 8)
│   ├── st_rl_448.sh                # Single-Turn RL training (GRPO)
│   ├── mt_rl_448.sh                # Multi-Turn RL training (GRPO + A2B reward)
│   └── continue_train_448.sh       # Continue Train (real-world SFT, Paper Section 6.2)
│
├── datas/                          # Single source of truth for all data
│   ├── config.json                 # Tree parameters, canvas/icon sizes
│   ├── ui_structure.json           # 231 pages with transitions (flat)
│   ├── ui_structure_layer.json     # Hierarchical tree structure (nested subnodes)
│   ├── pages/                      # 231 rendered page images (448x448, gitignored)
│   ├── sft_aligned.json            # SFT training data (30,888 samples)
│   ├── st_rl_path_only.json        # ST-RL training data (24,878 path samples, both subtrees)
│   ├── st_rl_path_sub2.json        # ST-RL training data (12,439 path samples, subtree 2 only)
│   ├── mt_rl_aligned.json          # MT-RL training data (2,200 tasks)
│   ├── test_id_edge.json           # ID edge test (548 samples, subtrees 0-1)
│   ├── test_id_path.json           # ID path test (24,878 all-step samples, subtrees 0-1)
│   ├── test_ood_edge.json          # OOD edge test (274 samples, subtree 4)
│   └── test_ood_path.json          # OOD path test (12,439 all-step samples, subtree 4)
│
├── environment/demo -> datas/      # Symlink for runtime access
│
├── archive/                        # Legacy scripts (42 files, for reference)
│   ├── scripts/                    # Old eval, training, data gen variants
│   └── data/                       # Old datasets & UI environments (gitignored)
│
└── Progress.md                     # Reproduction results and analysis
```

---

## 2. Training Scripts

All training uses the [ms-swift](https://github.com/modelscope/swift) framework with DeepSpeed.
Scripts require `WANDB_API_KEY` and `HF_TOKEN` set in your shell environment.

### 2.1 SFT (Supervised Fine-Tuning)

**Script**: `gui_scripts/sft_448.sh`

Trains the base Qwen2.5-VL-7B-Instruct model on navigation trajectories from subtrees 0-1.
SFT data includes edge transitions from ALL subtrees (including test subtree 4) to provide
fundamental environment knowledge, plus grounding and captioning data for icon comprehension.

| Parameter | Our Value | Paper Value | Notes |
|-----------|-----------|-------------|-------|
| Base Model | Qwen2.5-VL-7B-Instruct | Qwen2.5-VL-7B-Instruct | |
| Dataset | 30,888 samples | ~30,888 | Path + Edge + Grounding + Captioning |
| Learning Rate | 1e-5 | 1e-5 | |
| Epochs | 1 | 1 | |
| Per-device Batch Size | 2 | 2 | |
| Gradient Accumulation | 4 | 2 | Effective batch: 3x2x4=24 (paper: 32) |
| DeepSpeed | ZeRO Stage 2 | — | |
| max_pixels | 200704 (448x448) | 200704 | |
| System Prompt | Paper Appendix A.10 | Paper Appendix A.10 | GUI Navigation Agent prompt |

```bash
export WANDB_API_KEY="your_key"
export HF_TOKEN="your_token"
bash gui_scripts/sft_448.sh
```

**Pre-trained checkpoint**: [`namhokaist/SFT_True_Final`](https://huggingface.co/namhokaist/SFT_True_Final) on HuggingFace Hub.

### 2.2 ST-RL (Single-Turn Reinforcement Learning)

**Script**: `gui_scripts/st_rl_448.sh`

Fine-tunes the SFT checkpoint with GRPO on single-step navigation. Uses subtree 2 only (12,439 samples) to reduce catastrophic forgetting — the paper reports 99,512 total interactions (= 12,439 × 8 generations), consistent with single-subtree training.

| Parameter | Our Value | Paper Value | Notes |
|-----------|-----------|-------------|-------|
| Base Model | SFT checkpoint | SFT checkpoint | |
| Algorithm | GRPO | GRPO | |
| Dataset | 12,439 path-only samples | 12,439* | Single subtree (subtree 2) |
| Learning Rate | 1e-6 | 1e-6 | |
| Epochs | 5 | 5 | |
| Per-device Batch Size | 16 | 8 | Adjusted for 3 GPUs |
| Num Generations | 8 | 8 | |
| Gradient Accumulation | 3 | — | Effective batch: 3x16x3=144 (paper: 128) |
| Temperature | 1.2 | 1.2 | |
| DeepSpeed | ZeRO Stage 3 | — | Required for 80GB GPUs (ZeRO-2 OOMs) |
| Gradient Checkpointing | true | — | Trades compute for memory |
| Reward Functions | 4 equally-weighted (0.25x4) | 4 equally-weighted | r_type + r_coord + r_intent + r_format |

Supports optional `MAX_STEPS` env var for quick validation runs (e.g., `export MAX_STEPS=20`).

```bash
export WANDB_API_KEY="your_key"
export HF_TOKEN="your_token"
bash gui_scripts/st_rl_448.sh
```

**Pre-trained checkpoint**: [`namhokaist/STRL_True_Final`](https://huggingface.co/namhokaist/STRL_True_Final) on HuggingFace Hub.

### 2.3 MT-RL (Multi-Turn Reinforcement Learning)

**Script**: `gui_scripts/mt_rl_448.sh`

Fine-tunes the ST-RL checkpoint with GRPO on multi-turn navigation episodes (up to 8 steps) using subtrees 2-3. Following the paper's three-stage pipeline: SFT → ST-RL → MT-RL.

| Parameter | Our Value | Paper Value | Notes |
|-----------|-----------|-------------|-------|
| Base Model | ST-RL checkpoint | ST-RL checkpoint | Pipeline: SFT → ST-RL → MT-RL |
| Algorithm | GRPO | GRPO | |
| Dataset | 2,200 multi-turn tasks | 2,162 tasks | |
| Learning Rate | 1e-6 | 1e-6 | |
| Epochs | 10 | 5 | Doubled to compensate for smaller effective batch |
| Per-device Batch Size | 1 | 8 | Conservative: multi-turn rollouts have variable memory |
| Num Generations | 3 | 8 | Must divide effective batch (3x1=3). Reduced from 6 to prevent OOM. |
| Gradient Accumulation | 16 | — | Effective batch: 3x1x16=48 (paper: 128) |
| Temperature | 1.2 | 1.2 | |
| Max Completion Length | 1024 | 1024 | |
| Max Turns | 8 | 12 | Configurable via `MT_RL_MAX_TURNS` env var (default 12) |
| Reward Function | A2B (sparse) | A2B (sparse) | +1 if target reached, 0 otherwise |
| Multi-Turn Env | `gelab_multi_turn` | `gelab_multi_turn` | |
| DeepSpeed | ZeRO Stage 3 | ZeRO Stage 3 | |
| Gradient Checkpointing | true | — | Required for memory |

**Pre-trained checkpoint**: [`namhokaist/MTRL_True_Final`](https://huggingface.co/namhokaist/MTRL_True_Final) on HuggingFace Hub.

The multi-turn environment (`swift/plugin/multi_turn.py`) simulates interactive episodes:
1. Model outputs `click(x,y)` -> environment checks if click hits a valid icon bbox
2. If valid click: transitions to target page, updates history, gives model the new screenshot
3. If model reaches target page and outputs `complete`: episode ends with reward=1
4. If max turns exceeded (default 12, configurable via `MT_RL_MAX_TURNS` env var) or invalid navigation: episode ends with reward=0

Supports optional `MAX_STEPS` env var for quick validation runs.

```bash
export WANDB_API_KEY="your_key"
export HF_TOKEN="your_token"
bash gui_scripts/mt_rl_448.sh
```

### 2.4 Hardware Adaptation: 3x A100 80GB vs Paper's 16x A800

The paper uses 16x A800 GPUs. Fitting GRPO full fine-tuning of a 7B VLM on 3x A100 80GB required several adaptations:

| Adaptation | Reason |
|------------|--------|
| ZeRO-3 (not ZeRO-2) for RL | ZeRO-2 OOMs at optimizer step — it keeps full model params + gradients on each GPU. ZeRO-3 partitions everything. |
| Gradient checkpointing | Reduces activation memory by recomputing during backward pass. ~20 GB savings per GPU. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces CUDA memory fragmentation. |
| `num_generations` must divide effective batch | Must evenly divide `per_device_batch_size x num_gpus`. ST-RL achieves paper's 8; MT-RL uses 3 (paper: 8) due to multi-turn memory constraints. |
| Effective batch ~48-144 (paper: 128) | 3 GPUs vs 16 — compensated with gradient accumulation. ST-RL: 144, MT-RL: 48. |
| `save_only_model=true` | Reduces checkpoint size from 109 GB (with optimizer states) to ~15 GB (7x savings). |

**Memory profiles (ST-RL, ZeRO-3 + gradient checkpointing):**
- batch=1, gen=3: 63.6 GB (conservative)
- batch=8, gen=6: 55.5 GB (ZeRO-3 memory is flat across batch sizes)
- batch=16, gen=8: **63.7 GB** (current optimal — matches paper's `num_generations=8`)

**Key lesson learned:** Using 2 subtrees (24,878 samples) for ST-RL with only 3 epochs caused severe catastrophic forgetting — interactive Pass@1 dropped from 15% (SFT) to 7.35% (ST-RL). The paper's 99,512 total interactions (= 12,439 × 8 gen) implies single-subtree training. Switching to 1 subtree with 5 epochs and grad_acc=3 (effective batch 144, close to paper's 128) reduced total gradient updates from 12,439 to 3,420, matching the paper's scale.

### 2.5 Continue Train (Real-World Continual Training — Paper Section 6.2)

**Script**: `gui_scripts/continue_train_448.sh`

Continual SFT training on 24k real-world GUI samples from AITW and Mind2Web. This evaluates whether GE-Lab pre-training transfers to real-world downstream tasks. Applied to all 4 model stages (Base, SFT, ST-RL, MT-RL) to produce Table 5 results.

**Data Preparation**: `data_engine/prepare_continue_train_data.py`

Downloads AITW (mobile) and Mind2Web (web) from HuggingFace, converts to ms-swift SFT format with the real-world system prompt from Appendix A.5. Default: 12k AITW + 12k Mind2Web = 24k samples.

```bash
# Prepare the 24k real-world dataset
python data_engine/prepare_continue_train_data.py \
  --output datas/continue_train_24k.json \
  --image_dir datas/real_world_images \
  --aitw_samples 12000 \
  --mind2web_samples 12000
```

The real-world action space extends the simulator's with: `click`, `TYPE("text")`, `SCROLL(N)`, `WAIT(N)`, `complete`.

| Parameter | Our Value | Paper Value | Notes |
|-----------|-----------|-------------|-------|
| Dataset | 24k (AITW + Mind2Web) | 24k (AITW + AITZ + AMEX + Mind2Web) | AMEX excluded (87GB manual download) |
| Learning Rate | 1e-5 | 1e-5 | |
| Epochs | 2 | 2 | |
| Per-device Batch Size | 2 | 16 | |
| Gradient Accumulation | 43 | 1 | Effective batch: 3x2x43=258 (paper: 256) |
| DeepSpeed | ZeRO Stage 3 | — | ZeRO-2 + grad_ckpt + bf16 causes NaN |
| Gradient Checkpointing | true | — | |
| Max Length | 5120 | 5120 | |
| max_pixels | 200704 (448x448) | 200704 | |

```bash
# Train on each model stage (produces Table 5 columns)
MODEL_STAGE=base    bash gui_scripts/continue_train_448.sh  # Base Qwen2.5-VL
MODEL_STAGE=sft     bash gui_scripts/continue_train_448.sh  # SFT checkpoint
MODEL_STAGE=st_rl   bash gui_scripts/continue_train_448.sh  # ST-RL checkpoint
MODEL_STAGE=mt_rl   bash gui_scripts/continue_train_448.sh  # MT-RL checkpoint
```

**Evaluation**: `eval/evaluate_real_world.py`

Evaluates on 7 static grounding benchmarks (ScreenSpot, ScreenSpot-v2, FuncPred, MoTIF, Refexp, VWB-AG, VWB-EG) using click accuracy (point-in-bbox).

```bash
# Single-GPU evaluation
python eval/evaluate_real_world.py --model_path <checkpoint>

# Multi-GPU evaluation
python eval/evaluate_real_world.py --model_path <checkpoint> --num_gpus 3

# Specific benchmarks only
python eval/evaluate_real_world.py --model_path <checkpoint> --benchmarks screenspot motif
```

---

## 3. Evaluation

All evaluation uses the unified `eval/evaluate.py` script, which combines static (Table 1) and interactive (Table 2) evaluation with strict bbox matching, consistent system prompt, and proper response extraction.

### 3.1 Usage

```bash
# Static evaluation only (Table 1: ID/OOD Edge/Path accuracy)
python eval/evaluate.py --model_path <checkpoint> --mode static

# Interactive evaluation only (Table 2: Pass@1/Pass@5)
python eval/evaluate.py --model_path <checkpoint> --mode interactive

# Full evaluation (both)
python eval/evaluate.py --model_path <checkpoint> --mode all

# With LoRA adapter
python eval/evaluate.py --model_path Qwen/Qwen2.5-VL-7B-Instruct --lora_path <adapter_dir> --mode all

# Multi-GPU evaluation (3x speedup)
python eval/evaluate.py --model_path <checkpoint> --mode static --num_gpus 3

# Multi-GPU with multiple workers per GPU (overlaps CPU/GPU work for interactive eval)
python eval/evaluate.py --model_path <checkpoint> --mode all --num_gpus 3 --workers_per_gpu 2
```

The default `--env_dir` is `datas`, which contains the UI structure, page images, and test splits.

**Note:** All checkpoints (SFT, ST-RL, MT-RL) are full fine-tuned models, NOT LoRA adapters. Use `--model_path` directly — do not use `--lora_path`.

**Multi-GPU details:** When `--num_gpus > 1`, samples/tasks are split round-robin across workers. Each worker loads its own model on the assigned GPU via `torch.multiprocessing` with `spawn`. Use `--workers_per_gpu 2` on A100 80GB (each 7B bf16 model copy uses ~14GB).

### 3.2 OOD Environment Evaluation

To evaluate on OOD environment variants (Env-Base, Env-Image, Env-Name, etc.), use `--env_dir` for the environment and `--test_dir` for the test splits:

```bash
python eval/evaluate.py \
  --model_path namhokaist/gelab-sft-448-seed42 \
  --mode static \
  --env_dir <path_to_ood_environment> \
  --test_dir <path_to_test_splits>
```

`--env_dir` must contain `ui_structure.json`, `ui_structure_layer.json`, and `pages/`. `--test_dir` must contain the test JSON files (`test_id_edge.json`, `test_ood_edge.json`, etc.).

### 3.3 Metrics

**Table 1 — Static Benchmark (ID/OOD):**
- Edge accuracy: single-step transition correctness
- Path accuracy: per-step accuracy across all steps of multi-step navigation paths (matching paper methodology — includes intermediate steps with history context and final `complete` action)
- ID paths are generated per-subtree independently (subtrees 0 and 1 separately) to avoid cross-subtree contamination
- Evaluated on ID (subtrees 0-1) and OOD (subtree 4) splits

**Table 2 — Interactive Benchmark:**
- Pass@1: success rate with greedy decoding (temperature=0)
- Pass@5: success rate with any of 5 attempts (1 greedy + 4 sampled at temperature=0.7)

---

## 4. Data Generation Pipeline

### 4.1 Shared Utilities (`data_engine/env_utils.py`)

All data generation scripts inherit from `GELabEnvUtils`, which provides:
- UI structure loading (`ui_structure.json` + `ui_structure_layer.json`)
- Page-to-subtree mapping (using layer hierarchy)
- Navigation graph + NetworkX graph construction
- Bbox normalization (448px -> 0-1000 range)
- Shortest path finding with action/bbox metadata
- Action string formatting (`click(...)` and `complete`)

### 4.2 Generate a New UI Environment

```bash
python data_engine/tree.py --seed 42
```

This creates a timestamped directory under `data_engine/ui_environment_448/` with:
- `config.json`, `ui_structure.json`, `ui_structure_layer.json`
- `pages/` directory with 231 PNG files

Copy outputs to `datas/` to use as the active environment:
```bash
cp data_engine/ui_environment_448/<TIMESTAMP>/{config,ui_structure,ui_structure_layer}.json datas/
cp -r data_engine/ui_environment_448/<TIMESTAMP>/pages datas/
```

### 4.3 Generate Training Datasets

Each dataset type has its own generator. All read from `datas/` by default:

```bash
# SFT data: 30,888 samples (path + edge + grounding + captioning)
python data_engine/generate_sft_data.py --env_dir datas --output datas/sft_aligned.json

# ST-RL data: 24,878 path-only samples from subtrees 2-3
python data_engine/generate_st_rl_data.py --env_dir datas --output datas/st_rl_path_only.json

# MT-RL data: 2,200 tasks balanced by path length from subtrees 2-3
python data_engine/generate_mt_rl_data.py --env_dir datas --output datas/mt_rl_aligned.json

# Test splits: ID/OOD Edge/Path
python eval/generate_eval_splits.py --env_dir datas --output_dir datas
```

### 4.4 Dataset Sizes (seed=42)

| Dataset | File | Samples | Source |
|---------|------|---------|--------|
| SFT | `sft_aligned.json` | 30,888 | Subtrees 0-1 (path+edge+grounding+caption) |
| ST-RL | `st_rl_path_only.json` | 24,878 | Subtrees 2-3 (path-only) |
| MT-RL | `mt_rl_aligned.json` | 2,200 | Subtrees 2-3 (balanced by path length) |
| Test ID Edge | `test_id_edge.json` | 548 | Subtrees 0-1 (274 per subtree, all transitions incl. back/home) |
| Test ID Path | `test_id_path.json` | 24,878 | Subtrees 0-1 (12,439 per subtree, all path pairs) |
| Test OOD Edge | `test_ood_edge.json` | 274 | Subtree 4 (all transitions) |
| Test OOD Path | `test_ood_path.json` | 12,439 | Subtree 4 (all path pairs) |

---

## 5. How the GE-Lab Simulator Works

### 5.1 Architecture

The simulator is built from 5 subsystems in `data_engine/tree.py`:

```
Icons (PNG files)
    |
    v
UIManager          -- Loads icon pool, allocates icons to pages
    |
    v
TopologyGenerator  -- Builds tree structure via BFS (depth, branching factor)
    |
    v
LayoutGenerator    -- Assigns (x,y) positions on 448x448 canvas (grid layout)
    |
    v
TopologyBuilder    -- Creates NetworkX DiGraph: page->icon->child_page transitions
    |
    v
RenderEngine       -- Renders each page as 448x448 PNG (title, icons, system buttons)
```

### 5.2 Page Structure

Each page contains:
- **Title**: page name at the top (e.g., "page_42")
- **System icons**: back (top-left, returns to parent), home (top-right, returns to root page_0)
- **Content icons**: 1-5 clickable icons arranged in a grid (50x50 pixels each)
- **Transitions**: clicking an icon navigates to the corresponding child page

### 5.3 Tree Configuration

The paper uses a **5-subtree balanced tree**:

```
Root (page_0)
├── Subtree 0 (~46 pages) -> SFT Training
├── Subtree 1 (~46 pages) -> SFT Training
├── Subtree 2 (~46 pages) -> RL Training
├── Subtree 3 (~46 pages) -> RL Training
└── Subtree 4 (~46 pages) -> OOD Testing
```

Key parameters:
- `tree_depth`: 7 levels
- `nodes_per_level`: [5, 3, 2, 2, 1, 1] (branching factor per level)
- `canvas_size`: (448, 448)
- `icon_size`: (50, 50)
- Total pages: 231 (1 root + 230 tree nodes)

### 5.4 Coordinate System

All bounding boxes are normalized to 0-1000 range (canvas-agnostic).

**Prompt format** (training):
```
Instruction: from page_X to page_Y. History: Null
```

**Response format**:
```
Explain: click <icon_name> icon on <page>.\tAction: click(start_box='<|box_start|>(x,y)<|box_end|>')
```

### 5.5 Output Files

The simulator produces:

| File | Contents |
|------|----------|
| `config.json` | Tree parameters, canvas/icon sizes |
| `ui_structure.json` | Flat page definitions with transitions and layout bboxes |
| `ui_structure_layer.json` | Hierarchical tree structure (nested `subnodes`) |
| `pages/page_N.png` | 448x448 rendered page images |

---

## 6. Environment Setup

### 6.1 Prerequisites

- NVIDIA GPU with 80GB+ VRAM (A100 or H100 recommended)
- CUDA 12.x
- Miniconda or Anaconda

### 6.2 Create the Conda Environment

```bash
conda create -n gelab python=3.10 -y
conda activate gelab
```

### 6.3 Install PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 6.4 Install ms-swift (Training Framework)

```bash
cd /path/to/gelab-env
pip install -e .
```

### 6.5 Install Framework Dependencies

```bash
pip install -r requirements/framework.txt
```

Key packages:
- `transformers>=4.33,<4.53`
- `peft>=0.11,<0.16`
- `trl>=0.15,<0.18`
- `accelerate`, `deepspeed`
- `datasets>=3.0,<3.4`
- `pillow`, `numpy<2.0`, `scipy`
- `wandb` (experiment tracking)

### 6.6 Install Optional Packages

```bash
pip install vllm                        # Fast evaluation
pip install flash-attn --no-build-isolation  # Flash Attention
pip install qwen-vl-utils qwen-omni-utils    # Qwen VL utilities
```

### 6.7 Set Environment Variables

Add to your `~/.bashrc` or run before training:

```bash
export WANDB_API_KEY="your_wandb_api_key"
export HF_TOKEN="your_huggingface_token"
export HF_HOME="/path/to/cache/huggingface"
export CUDA_HOME=/path/to/cuda-12.x
```

### 6.8 Verify Installation

```bash
conda activate gelab
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import swift; print(f'ms-swift {swift.__version__}')"
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('Qwen2.5-VL OK')"
```

---

## Quick Reference: Full Pipeline

```bash
# 1. Setup
conda activate gelab
export WANDB_API_KEY="your_key"
export HF_TOKEN="your_token"

# 2. Generate UI environment (with reproducible seed)
python data_engine/tree.py --seed 42

# 3. Copy environment to datas/
cp data_engine/ui_environment_448/<TIMESTAMP>/{config,ui_structure,ui_structure_layer}.json datas/
cp -r data_engine/ui_environment_448/<TIMESTAMP>/pages datas/

# 4. Generate training datasets
python data_engine/generate_sft_data.py
python data_engine/generate_st_rl_data.py
python data_engine/generate_mt_rl_data.py
python eval/generate_eval_splits.py

# 5. SFT training
bash gui_scripts/sft_448.sh

# 6. ST-RL training (after SFT completes)
bash gui_scripts/st_rl_448.sh

# 7. MT-RL training (after ST-RL completes)
bash gui_scripts/mt_rl_448.sh

# 8. Evaluate (single GPU)
python eval/evaluate.py --model_path <checkpoint> --mode all

# 8b. Evaluate (multi-GPU, ~3x faster)
python eval/evaluate.py --model_path <checkpoint> --mode all --num_gpus 3
```

---

## Pre-trained Models

All three pipeline stages are available on HuggingFace. These are full fine-tuned models (not LoRA adapters).

| Stage | HuggingFace Model | Training Data | Base Model |
|-------|-------------------|---------------|------------|
| SFT | [`namhokaist/SFT_True_Final`](https://huggingface.co/namhokaist/SFT_True_Final) | 30,888 samples (subtrees 0-1) | Qwen2.5-VL-7B-Instruct |
| ST-RL | [`namhokaist/STRL_True_Final`](https://huggingface.co/namhokaist/STRL_True_Final) | 12,439 path samples (subtree 2) | SFT_True_Final |
| MT-RL | [`namhokaist/MTRL_True_Final`](https://huggingface.co/namhokaist/MTRL_True_Final) | 2,200 multi-turn tasks (subtrees 2-3) | STRL_True_Final |

### Download and Evaluate

```bash
# Evaluate SFT model (static + interactive)
python eval/evaluate.py --model_path namhokaist/SFT_True_Final --mode all

# Evaluate ST-RL model
python eval/evaluate.py --model_path namhokaist/STRL_True_Final --mode all

# Evaluate MT-RL model
python eval/evaluate.py --model_path namhokaist/MTRL_True_Final --mode all

# Multi-GPU (3x faster)
python eval/evaluate.py --model_path namhokaist/SFT_True_Final --mode all --num_gpus 3
```

### Expected Results (Pipeline B Baselines)

| Model | ID Edge | ID Path | ID Overall | OOD Edge | OOD Path | OOD Overall | Pass@1 | Pass@5 |
|-------|---------|---------|------------|----------|----------|-------------|--------|--------|
| SFT_True_Final | 98.18 | 97.54 | 97.55 | 97.81 | 61.32 | 62.11 | 15.68 | 20.72 |
| STRL_True_Final | 83.03 | 87.49 | 87.39 | 86.50 | 61.30 | 61.84 | 8.83 | 9.81 |
| MTRL_True_Final | 83.94 | 87.60 | 87.52 | 86.86 | 60.53 | 61.09 | 8.74 | 9.99 |

### Load a Pre-trained Model in Python

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_name = "namhokaist/SFT_True_Final"  # or STRL_True_Final, MTRL_True_Final
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")
processor = AutoProcessor.from_pretrained(model_name)
```
