# GE-Lab Tutorial

**Paper**: [GUI Exploration Lab: Enhancing Screen Navigation in Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/pdf/2512.02423)

---

## 1. Codebase Structure

```
gelab-env/
├── data_engine/                    # UI environment generation pipeline
│   ├── tree.py                     # Core: synthetic tree-structured UI generator
│   ├── tree_v1.py                  # Extension: adds TEXT action (search bars)
│   ├── tree_v1_expanded.py         # Extension: full actions (CLICK, TEXT, SCROLL, etc.)
│   ├── generate_dataset_paper.py   # Dataset generation with paper's 2:2:1 subtree split
│   ├── generate_dataset_v2.py      # Generic dataset generation (no partition)
│   ├── build_gelab_real_icons.py   # Sim2Real: real app icon-based page templates
│   ├── extract_real_ui.py          # Sim2Real: extract UI elements from real screenshots
│   ├── render_templates_only.py    # Render page templates without real icons
│   └── icons/                      # Icon pool (Animals/, Business/, etc.)
│
├── swift/plugin/                   # Training plugins (modified ms-swift framework)
│   ├── multi_turn.py               # Multi-turn environment for MT-RL rollouts
│   └── orm.py                      # Reward functions (A2B sparse reward)
│
├── eval/                           # Evaluation scripts
│   ├── evaluate_paper_style.py     # Table 1: ID/OOD Edge/Path accuracy
│   ├── interactive_eval.py         # Table 2: Interactive Pass@1/Pass@5 (single GPU)
│   ├── interactive_eval_multigpu.py# Table 2: Multi-GPU parallel evaluation
│   ├── interactive_benchmark.py    # Table 2: vLLM-accelerated evaluation
│   ├── eval_sft_retrain.py         # Retrained SFT 4-metric evaluation
│   ├── eval_st_rl.py               # ST-RL LoRA model evaluation
│   ├── trajectory_eval.py          # Step-by-step trajectory validation
│   └── inference_correct_prompt.py # Batch inference with correct system prompt
│
├── gui_scripts/                    # Training launch scripts
│   ├── sft_448_3gpu.sh             # SFT training (3x A100)
│   ├── st_rl_448.sh                # Single-Turn RL training
│   ├── mt_rl_448.sh                # Multi-Turn RL training (Paper Table 8)
│   └── ...                         # Additional variants
│
├── environment/demo/               # Runtime UI structure for multi-turn env
│   ├── ui_structure.json           # 231 pages with transitions (flat)
│   └── ui_structure_layer_fixed.json # Hierarchical tree structure
│
├── datas/
│   ├── 448_paper/                  # Paper-aligned training/test datasets
│   ├── 448_paper_new/              # Regenerated eval splits (ID/OOD Edge/Path)
│   └── 448_retrain/                # Retrained model datasets + UI configs
│
└── Progress.md                     # Reproduction results and analysis
```

---

## 2. Training Scripts

All training uses the [ms-swift](https://github.com/modelscope/swift) framework with DeepSpeed ZeRO-3.
Scripts require `WANDB_API_KEY` and `HF_TOKEN` set in your shell environment.

### 2.1 SFT (Supervised Fine-Tuning)

**Script**: `gui_scripts/sft_448_3gpu.sh`

Trains the base Qwen2.5-VL-7B-Instruct model on multi-turn navigation trajectories from subtrees 0-1.

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-VL-7B-Instruct |
| Dataset | ~30,888 multi-turn samples (Edge + Path + Grounding + Captioning) |
| GPUs | 3x A100 80GB |
| Learning Rate | 1e-5 |
| Epochs | 1 |
| DeepSpeed | ZeRO Stage 2 |
| max_pixels | 200704 (448x448) |

```bash
export WANDB_API_KEY="your_key"
export HF_TOKEN="your_token"
bash gui_scripts/sft_448_3gpu.sh
```

### 2.2 ST-RL (Single-Turn Reinforcement Learning)

**Script**: `gui_scripts/st_rl_448.sh`

Fine-tunes the SFT checkpoint with GRPO on single-step navigation using subtrees 2-3.

| Parameter | Value |
|-----------|-------|
| Base Model | SFT checkpoint |
| Algorithm | GRPO |
| Dataset | 3,000 single-step edge samples |
| Learning Rate | 1e-6 |
| Epochs | 10 |
| Num Generations | 3 |
| Temperature | 1.2 |
| DeepSpeed | ZeRO Stage 3 |

```bash
export WANDB_API_KEY="your_key"
export HF_TOKEN="your_token"
bash gui_scripts/st_rl_448.sh
```

### 2.3 MT-RL (Multi-Turn Reinforcement Learning)

**Script**: `gui_scripts/mt_rl_448.sh`

Fine-tunes the SFT checkpoint with GRPO on multi-turn navigation episodes (up to 8 steps) using subtrees 2-3.

| Parameter | Value | Paper Value |
|-----------|-------|-------------|
| Base Model | SFT checkpoint (retrained) | SFT checkpoint |
| Algorithm | GRPO | GRPO |
| Dataset | 2,200 multi-turn tasks | 2,200 tasks |
| Learning Rate | 1e-6 | 1e-6 |
| Epochs | 10 | 10 |
| Num Generations | 3 | 8 |
| Temperature | 1.2 | 1.2 |
| Max Completion Length | 1024 | 1024 |
| Reward Function | A2B (sparse) | A2B (sparse) |
| Multi-Turn Env | `gelab_multi_turn` | `gelab_multi_turn` |
| DeepSpeed | ZeRO Stage 3 | ZeRO Stage 3 |

The multi-turn environment (`swift/plugin/multi_turn.py`) simulates interactive episodes:
1. Model outputs `click(x,y)` -> environment checks if click hits a valid icon bbox
2. If valid click: transitions to target page, updates history, gives model the new screenshot
3. If model reaches target page and outputs `complete`: episode ends with reward=1
4. If 8 steps exceeded or invalid navigation: episode ends with reward=0

```bash
export WANDB_API_KEY="your_key"
export HF_TOKEN="your_token"
bash gui_scripts/mt_rl_448.sh
```

---

## 3. Evaluation Scripts

### 3.1 Table 1: Static Evaluation (ID/OOD Edge/Path)

Evaluates single-step accuracy on Edge (1-hop) and Path (multi-hop) test sets across in-distribution (subtrees 0-1) and out-of-distribution (subtree 4).

**For SFT models:**
```bash
python eval/eval_sft_retrain.py \
    --model_path checkpoint/gui_exp/sft_448_retrain/v0-20260201_054616/checkpoint-850 \
    --test_dir datas/448_paper \
    --verbose
```

**For ST-RL LoRA models:**
```bash
python eval/eval_st_rl.py \
    --base_model checkpoint/gui_exp/sft_448_retrain/v0-20260201_054616/checkpoint-850 \
    --lora_path checkpoint/gui_exp/st_rl_448_lora/v0-XXXXXXXX/checkpoint-576 \
    --test_dir datas/448_paper \
    --verbose
```

**For merged models (evaluate_paper_style.py):**
```bash
python eval/evaluate_paper_style.py \
    --model_path checkpoint/gui_exp/st_rl_448_balanced/merged \
    --eval_all
```

### 3.2 Table 2: Interactive Benchmark (Pass@1 / Pass@5)

Evaluates multi-step interactive navigation where the model controls an agent through the UI environment.

- **Pass@1**: Success with greedy decoding (temperature=0)
- **Pass@5**: Success with any of 5 attempts (1 greedy + 4 sampled at temperature=0.7)

**Single GPU (HuggingFace inference):**
```bash
python eval/interactive_eval.py \
    --model_path checkpoint/gui_exp/st_rl_448_balanced/merged \
    --env_dir data_engine/ui_environment_448/latest \
    --num_tasks 2162 \
    --num_attempts 5 \
    --save_results results/interactive_eval.json
```

**Multi-GPU (parallel workers):**
```bash
python eval/interactive_eval_multigpu.py \
    --model_path checkpoint/gui_exp/st_rl_448_balanced/merged \
    --env_dir data_engine/ui_environment_448/latest \
    --num_tasks 2162 \
    --num_gpus 3 \
    --save_results results/interactive_multigpu.json
```

**vLLM-accelerated (fastest single GPU):**
```bash
python eval/interactive_benchmark.py \
    --model_path checkpoint/gui_exp/st_rl_448_balanced/merged \
    --subtree 4 \
    --num_tasks 500 \
    --num_attempts 5 \
    --output results/interactive_vllm.json
```

### 3.3 Eval Script Summary

| Script | Metrics | Speed | Backend | Multi-GPU |
|--------|---------|-------|---------|-----------|
| eval_sft_retrain.py | Table 1 (4 metrics) | Slow | HF | No |
| eval_st_rl.py | Table 1 (LoRA) | Slow | HF | No |
| evaluate_paper_style.py | Table 1 (merged) | Slow | HF | No |
| interactive_eval.py | Pass@1/5 | Very slow | HF | No |
| interactive_eval_multigpu.py | Pass@1/5 | Fast | HF | Yes |
| interactive_benchmark.py | Pass@1/5 | Fast | vLLM | No |

---

## 4. GUI Emulator: Sim2Real Augmentation

The GE-Lab simulator generates synthetic UI environments for training GUI navigation agents. The sim2real pipeline extends this with real app screenshots.

### Synthetic Environment (Core)

The core simulator (`data_engine/tree.py`) generates a tree-structured UI with:
- **Pages**: 448x448 pixel images with icons arranged in a grid
- **Navigation**: Each icon click transitions to a child page; back/home buttons navigate up
- **Structure**: A balanced tree with configurable depth and branching factor

This produces a fully deterministic, controllable environment where every page, icon, and transition is known -- enabling automated reward computation.

### Sim2Real Augmentation

Two scripts extend the synthetic environment toward realistic UIs:

1. **extract_real_ui.py**: Takes real app screenshots (e.g., from a Pixel device at 1344x2992) with OmniParser-generated bounding box annotations, extracts individual icon crops, and builds a transition graph from recorded interaction sequences.

2. **build_gelab_real_icons.py**: Constructs 6 realistic page templates mimicking an Amazon shopping flow (home screen -> search -> results -> product -> cart). It extracts real app icons (TikTok, Amazon, eBay, etc.) and arranges them in structured mobile-style layouts (336x748).

The sim2real approach preserves the GE-Lab training paradigm (known transitions, automated rewards) while using visually realistic page renders.

---

## 5. How the GeLab Simulator Works

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

### 5.4 Dataset Generation

`generate_dataset_paper.py` partitions the tree into subtrees and generates:

| Dataset | Source Subtrees | Samples | Format |
|---------|----------------|---------|--------|
| SFT | 0, 1 | ~30,888 | Multi-turn trajectories (all steps of all paths) |
| ST-RL | 2, 3 | 3,000 | Single-step edges with bbox coordinates |
| MT-RL | 2, 3 | 2,200 | Multi-turn first-steps, balanced by path length |
| Test ID | 0, 1 | 525 | Edge (90) + Path (435) |
| Test OOD | 4 | 507 | Edge (45) + Path (462) |

**Coordinate system**: All bounding boxes are normalized to 0-1000 range (canvas-agnostic).

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
| `ui_structure_layer.json` | Hierarchical tree structure (nested subnodes) |
| `pages/page_N.png` | 448x448 rendered page images |

---

## 6. Running the Simulator

### 6.1 Generate a New UI Environment

```bash
cd /ext_hdd2/nhkoh/gelab-env
python data_engine/tree.py
```

This creates a new timestamped directory under `data_engine/ui_environment_448/` with:
- `config.json`, `ui_structure.json`, `ui_structure_layer.json`
- `pages/` directory with 231 PNG files

To use the generated environment, create a symlink:
```bash
ln -sfn data_engine/ui_environment_448/<TIMESTAMP> data_engine/ui_environment_448/latest
```

### 6.2 Generate Training Datasets

After generating the environment:

```bash
python data_engine/generate_dataset_paper.py
```

This reads from `environment/demo/ui_structure.json` and produces the paper-aligned datasets (SFT, ST-RL, MT-RL, test splits) into the `datas/` directory.

### 6.3 Copy Environment for Runtime Use

The multi-turn RL environment reads from `environment/demo/` at runtime. Copy your generated environment there:

```bash
cp data_engine/ui_environment_448/latest/ui_structure.json environment/demo/
cp data_engine/ui_environment_448/latest/ui_structure_layer.json environment/demo/ui_structure_layer_fixed.json
cp -r data_engine/ui_environment_448/latest/pages environment/demo/
```

### 6.4 Sim2Real Template Rendering

To render the real-app-style page templates (without real icons):

```bash
python data_engine/render_templates_only.py
```

This produces template PNGs and a horizontal strip showing all 6 pages in sequence.

---

## 7. Environment Setup

### 7.1 Prerequisites

- NVIDIA GPU with 80GB+ VRAM (A100 or H100 recommended)
- CUDA 12.x
- Miniconda or Anaconda

### 7.2 Create the Conda Environment

```bash
conda create -n gelab python=3.10 -y
conda activate gelab
```

### 7.3 Install PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 7.4 Install ms-swift (Training Framework)

```bash
pip install git+https://github.com/modelscope/swift.git
```

Or install from the local repo:

```bash
cd /path/to/gelab-env
pip install -e .
```

### 7.5 Install Framework Dependencies

```bash
pip install -r requirements/framework.txt
```

Key packages this installs:
- `transformers>=4.33,<4.53`
- `peft>=0.11,<0.16`
- `trl>=0.15,<0.18`
- `accelerate`
- `datasets>=3.0,<3.4`
- `pillow`, `numpy<2.0`, `scipy`
- `wandb` (experiment tracking)

### 7.6 Install DeepSpeed

```bash
pip install deepspeed
```

### 7.7 Install vLLM (Optional, for Fast Evaluation)

```bash
pip install vllm
```

### 7.8 Install Flash Attention (Optional, Recommended)

Download the appropriate wheel from [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases) for your CUDA/PyTorch version:

```bash
pip install flash-attn --no-build-isolation
```

### 7.9 Install Qwen VL Utilities

```bash
pip install qwen-vl-utils qwen-omni-utils
```

### 7.10 Set Environment Variables

Add to your `~/.bashrc` or run before training:

```bash
export WANDB_API_KEY="your_wandb_api_key"
export HF_TOKEN="your_huggingface_token"
export HF_HOME="/path/to/cache/huggingface"
export CUDA_HOME=/path/to/cuda-12.x
```

### 7.11 Verify Installation

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

# 2. Generate UI environment
python data_engine/tree.py

# 3. Generate training datasets
python data_engine/generate_dataset_paper.py

# 4. SFT training
bash gui_scripts/sft_448_3gpu.sh

# 5. ST-RL training (after SFT completes)
bash gui_scripts/st_rl_448.sh

# 6. MT-RL training (after SFT completes)
bash gui_scripts/mt_rl_448.sh

# 7. Evaluate (Table 1)
python eval/eval_sft_retrain.py --model_path <checkpoint> --test_dir datas/448_paper

# 8. Evaluate (Table 2 - Interactive)
python eval/interactive_eval_multigpu.py \
    --model_path <checkpoint> \
    --env_dir data_engine/ui_environment_448/latest \
    --num_tasks 2162 --num_gpus 3
```
