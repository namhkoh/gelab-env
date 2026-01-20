# GE-Lab Reproduction Progress

**Paper**: [GUI Exploration Lab: Enhancing Screen Navigation in Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/pdf/2512.02423)

**Branch**: `koh-dev/data-generation`

---

## Completed

### 1. Environment Setup
- [x] Created conda environment `gelab` (Python 3.10)
- [x] Installed ms-swift v3.5.0 with all dependencies
- [x] Configured environment variables:
  - WANDB: `namhokoh-korea-advanced-institute-of-science-and-technology/gelab`
  - HuggingFace token configured
  - Cache directory: `/ext_hdd2/nhkoh/.cache/`

### 2. Synthetic GUI Environment Generation
- [x] Generated UI environment using `tree.py`
- [x] Tree structure: depth 7, nodes_per_level=[5,3,2,2,1,1]
- [x] Output: 231 pages with navigation graph
- [x] Location: `data_engine/ui_environment/20260120_214711/`

### 3. Dataset Generation Script
- [x] Created `data_engine/generate_dataset.py` (paper-aligned)
- [x] Coordinate system: 0-1000 normalized (matching paper)
- [x] Balanced sampling across Path@1-7
- [x] Auto-setup for reward functions (`environment/demo/`)

### 4. Training Datasets Generated
| Dataset | Samples | Purpose |
|---------|---------|---------|
| `datas/sft.json` | 22,496 | Supervised Fine-Tuning |
| `datas/st_rl.json` | 3,000 | Single-Turn RL (GRPO) |
| `datas/mt_rl.json` | 3,000 | Multi-Turn RL (GRPO) |
| `datas/test.json` | 477 | Evaluation |
| `datas/images/` | 231 | Page images |

---

## Next Steps

### 5. Training Pipeline
- [ ] **SFT Stage**: Fine-tune base model on navigation data
  ```bash
  swift sft --model <MODEL_PATH> --dataset datas/sft.json ...
  ```
- [ ] **ST-RL Stage**: Single-turn GRPO with reward functions
  ```bash
  swift rlhf --rlhf_type grpo --reward_funcs web_action_match web_coordinate_match web_intent_match ...
  ```
- [ ] **MT-RL Stage**: Multi-turn GRPO with a2b reward
  ```bash
  swift rlhf --rlhf_type grpo --reward_funcs a2b --multi_turn_func gelab_multi_turn ...
  ```

### 6. Evaluation
- [ ] Run inference on test set
- [ ] Calculate metrics (Step accuracy, Task accuracy per path length)

### 7. Optional Enhancements
- [ ] Generate larger environment (more pages/icons)
- [ ] Ablation studies on training data size
- [ ] Real-world GUI benchmark evaluation

---

## Quick Commands

```bash
# Activate environment
conda activate gelab
export WANDB_API_KEY="<YOUR_WANDB_KEY>"
export HF_TOKEN="<YOUR_HF_TOKEN>"
export HF_HOME="/ext_hdd2/nhkoh/.cache/huggingface"

# Regenerate datasets (if needed)
python data_engine/generate_dataset.py \
    --ui_structure data_engine/ui_environment/20260120_214711/ui_structure.json \
    --pages_dir data_engine/ui_environment/20260120_214711/pages \
    --output_dir datas \
    --project_root /ext_hdd2/nhkoh/gelab-env
```

---

*Last updated: 2026-01-20*
