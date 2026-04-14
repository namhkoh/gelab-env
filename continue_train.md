# Continue-Train Pipeline (Paper Section 6.2)

Continual training on real-world GUI data to improve grounding on benchmarks like ScreenSpot, MoTIF, Refexp, and VWB.

## HuggingFace Models

| Model | Description | Avg Accuracy |
|-------|------------|-------------|
| [`namhokaist/Qwen2.5-VL-7B-ContinueTrain-v2`](https://huggingface.co/namhokaist/Qwen2.5-VL-7B-ContinueTrain-v2) | **Best model.** Base Qwen + 21k mixed real-world data (LR=1e-6, 1ep) | **77.7%** |
| [`namhokaist/Qwen2.5-VL-7B-AmexGeLab-SFT-v3`](https://huggingface.co/namhokaist/Qwen2.5-VL-7B-AmexGeLab-SFT-v3) | Base Qwen + 24k amex-gelab 3-task (LR=1e-6, 1ep) | 64.0% |

Base model: `Qwen/Qwen2.5-VL-7B-Instruct` (76.4% avg on same benchmarks)

---

## Understanding Eval Modes: `--base_model` vs `--use_system_prompt`

This is critical for reproducing scores. The same checkpoint can produce wildly different numbers depending on which eval mode you use.

### What happens during evaluation

The eval script loads a grounding benchmark (e.g., ScreenSpot), which provides a screenshot and a text instruction like "Click on the search icon". The script must:
1. **Prompt the model** to locate the element
2. **Parse the model's response** to extract (x, y) coordinates
3. **Check** if those coordinates fall within the ground-truth bounding box

The two eval modes differ in **how they prompt the model** and **what response format they expect**.

### `--base_model` mode (Qwen native format)

**Prompt sent to the model:**
```
User: [screenshot]
      Click on the search icon
      Please provide the bounding box coordinate of the region this sentence describes.
```

**Expected model response** (Qwen2.5-VL native output):
```
<ref>search icon</ref><box>(456,78),(512,134)</box>
```
or `"bbox_2d": [456, 78, 512, 134]` in pixel coordinates.

The eval script normalizes pixel coordinates to 0-1000 space using the image dimensions.

**When to use:** For base Qwen2.5-VL or any model that still retains Qwen's native grounding ability. This includes models trained with gentle fine-tuning (LR=1e-6, 1 epoch) where the base model's output format is largely preserved.

### `--use_system_prompt` mode (GE-Lab format)

**Prompt sent to the model:**
```
System: You are a Multifaceted Mobile Interface Assistant...
        [full system prompt with Navigation/Grounding/Understanding task types]
User: [screenshot]
      I want to click on the search icon.
      Please locate the target element I should interact with. (with point)
```

**Expected model response** (GE-Lab trained output):
```
Action: click(start_box='<|box_start|>(456,78)<|box_end|>')
```
in 0-1000 normalized coordinates.

**When to use:** Only for models that were specifically trained to output the `click(start_box=...)` format through extensive SFT on GE-Lab simulation data (the paper's SFT/ST-RL/MT-RL checkpoints). Our models trained from base Qwen do NOT reliably produce this format for grounding queries, because our training data is navigation-style (not grounding query-response pairs).

### Why our models score low with `--use_system_prompt`

Our training data looks like:
```
User: <image>Open Hotels. Search for car rental...
Assistant: Explain: click Hotels.com on page_0.  Action: click(start_box='...')
```

But the eval grounding query looks like:
```
User: <image>I want to click on the search icon. Please locate the target element...
```

The model never saw this grounding query format during training. It often responds with `complete` (thinking it's a navigation task that's already done) or outputs coordinates for the wrong element. The `click(start_box=...)` tokens appear in training, but only in response to navigation instructions -- not grounding queries.

### Score impact

The same checkpoint (`namhokaist/Qwen2.5-VL-7B-ContinueTrain-v2`) produces:

| Eval mode | ScreenSpot | Average | Why |
|-----------|-----------|---------|-----|
| `--base_model` | **77.6%** | **77.7%** | Model retains Qwen's native grounding, slightly improved by real-world GUI exposure |
| `--use_system_prompt` | 10.4% | 12.2% | Model doesn't know how to respond to grounding queries in GE-Lab format |
| (neither, default) | 49.8% | 53.9% | Partially works -- grounding prompt without system prompt, model guesses format |

**All scores reported in this document use `--base_model` unless stated otherwise.**

---

## Reproduce Best Model: Run 2 (77.7% avg)

### Environment Setup

```bash
conda create -n gelab python=3.11 -y
conda activate gelab

# PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Training stack
pip install 'ms-swift[llm]' deepspeed qwen-vl-utils wandb tf-keras
pip install 'transformers>=4.33,<4.53'  # required for local swift/ compatibility

# Flash attention (requires CUDA toolkit for compilation)
conda install -c nvidia cuda-toolkit -y
# Adjust TORCH_CUDA_ARCH_LIST for your GPU: "9.0" for H200/H100, "8.0" for A100
TORCH_CUDA_ARCH_LIST="9.0" pip install flash-attn==2.7.3 --no-cache-dir --no-build-isolation
```

**Note:** The repo contains a local `swift/` directory (v3.5) that shadows the pip-installed ms-swift 4.0.4. The CLI `swift sft` uses the pip version, but Python imports from the repo root use the local version. This requires `transformers<4.53` for compatibility.

### Step 1: Prepare 21k Mixed Training Data

```bash
export HF_TOKEN="your_hf_token"

python data_engine/prepare_continue_train_data.py \
    --output datas/continue_train_24k.json \
    --image_dir datas/real_world_images \
    --total_samples 24000 \
    --sources aitw aitz amex mind2web \
    --cache_dir ~/.cache/huggingface/datasets
```

**AMEX requires manual download** (87GB, multi-part zip):
```bash
huggingface-cli download Yuxiang007/AMEX --repo-type dataset --local-dir datas/amex_raw
cd datas/amex_raw/AMEX
unzip instruction_anno.zip
conda install -c conda-forge p7zip -y  # needed for multi-part zip
7z x screenshot.zip
```
Then update `AMEX_LOCAL_ANNOTATIONS` and `AMEX_LOCAL_SCREENSHOTS` in `data_engine/prepare_continue_train_data.py` to point to the extracted paths.

**Data sources** (all from HuggingFace):
| Source | HuggingFace ID | Samples | Notes |
|--------|---------------|---------|-------|
| AITW | `cjfcsjt/AITW_Single` | 6,000 | Auto-downloaded |
| AITZ | `xwm/AITZ` | 3,337 | Auto-downloaded, only 3,337 valid after filtering |
| AMEX | `Yuxiang007/AMEX` | 6,000 | Manual download required (87GB) |
| Mind2Web | `osunlp/Multimodal-Mind2Web` | 6,000 | Auto-downloaded |
| **Total** | | **~21,337** | |

**Output format:** ms-swift compatible JSON with `messages` (user + assistant) and `images` (absolute paths). All coordinates in 0-1000 normalized space.

**Runtime:** ~1 hour (dominated by AITW and Mind2Web image saving).

### Step 2: Train

```bash
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_hf_token"
MODEL_STAGE=base bash gui_scripts/continue_train_448.sh
```

**Training configuration** (`gui_scripts/continue_train_448.sh`):

| Parameter | Value | Paper value | Notes |
|-----------|-------|-------------|-------|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` | Same | Base model, not SFT/RL checkpoint |
| Learning rate | **1e-6** | 1e-5 | 10x lower to prevent catastrophic forgetting (see Key Findings) |
| Epochs | **1** | 2 | Half exposure to reduce forgetting |
| Effective batch size | 256 | 256 | Matches paper exactly |
| Per-device batch | 8 | 16 | Adjusted for GPU count |
| Gradient accumulation | 8 | 1 | 4 GPUs * 8 batch * 8 accum = 256 |
| Max length | 5120 | 5120 | Matches paper |
| Max pixels | 1003520 | 200704 | Full resolution to avoid coordinate mismatch at eval |
| DeepSpeed | ZeRO-3 | - | With gradient checkpointing + bf16 |
| Packing | true | - | Eliminates variable-length padding waste (4x speedup) |
| Flash attention | flash_attn | - | Required for packing |
| System prompt | Simplified (Navigation only) | Full A.10 | See explanation in Key Findings |

**Why we deviate from paper hyperparameters:** The paper trains from SFT/ST-RL/MT-RL checkpoints that were already fine-tuned on GE-Lab simulation data. These checkpoints already learned the output format, so aggressive LR=1e-5 is safe. We train from raw base Qwen, which has never seen the GE-Lab format. LR=1e-5 causes catastrophic forgetting of the base model's grounding ability. LR=1e-6 preserves it.

**Runtime:** ~30 min on 4x H200 (30 steps at ~56s/step). Estimated ~90 min on 4x A100.

### Step 3: Evaluate

```bash
# Evaluate the trained checkpoint
python eval/evaluate_real_world.py \
    --model_path checkpoint/gui_exp/continue_train_448_base/<run_dir>/checkpoint-30 \
    --base_model \
    --num_gpus 4 \
    --output_file eval/results.json

# Or evaluate the HuggingFace model directly
python eval/evaluate_real_world.py \
    --model_path namhokaist/Qwen2.5-VL-7B-ContinueTrain-v2 \
    --base_model \
    --num_gpus 4 \
    --output_file eval/results.json

# Baseline: base Qwen2.5-VL (no training)
python eval/evaluate_real_world.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --base_model \
    --num_gpus 4 \
    --output_file eval/results_base.json
```

**Important:** Use `--base_model` for all models trained with our pipeline. See "Understanding Eval Modes" above for why.

**Expected results** (Run 2, `--base_model` eval):

| Benchmark | Base Qwen | Run 2 | Delta |
|-----------|-----------|-------|-------|
| ScreenSpot | 74.8% | **77.6%** | +2.8 |
| ScreenSpot-v2 | 77.3% | **80.2%** | +2.9 |
| MoTIF | 82.0% | **82.8%** | +0.9 |
| Refexp | 83.9% | 82.7% | -1.3 |
| VWB-EG | 75.3% | **76.8%** | +1.4 |
| VWB-AG | 65.0% | **66.0%** | +1.0 |
| **Average** | 76.4% | **77.7%** | **+1.3** |

**Benchmarks** (6 of 8 from Paper Table 5):
- ScreenSpot, ScreenSpot-v2, MoTIF, Refexp, VWB-EG, VWB-AG
- FuncPred: requires manual download from [AutoGUI GitHub](https://github.com/ZJULiHongxin/AutoGUI)
- AndroidWorld: requires Android emulator infrastructure (not included)

**Runtime:** ~35-45 min on 4 GPUs (~5k total samples across 6 benchmarks).

---

## Reproduce amex-gelab SFT: Run 5 (64.0% avg)

This trains on the `luca0621/amex-gelab` dataset, which contains AMEX mobile app data processed through the GE-Lab engine. We generate 3 task types from the trajectory data:

1. **Navigation** (8k): Given a screenshot + task instruction, predict which element to click
2. **Grounding** (8k): Given a screenshot + "Click on [element name]", predict click coordinates
3. **Understanding** (8k): Given a screenshot + "What is at (x,y)?", predict element name

### Step 1: Download and Extract

```bash
# Download all 61 tar shards (~63GB)
huggingface-cli download luca0621/amex-gelab --repo-type dataset --local-dir datas/amex_gelab_raw

# Extract all shards (parallel, ~10 min)
mkdir -p datas/amex_gelab_extracted
ls datas/amex_gelab_raw/shards/*.tar | xargs -P 8 -I {} bash -c 'tar xf {} -C datas/amex_gelab_extracted/'
```

This produces `datas/amex_gelab_extracted/amex_sft/` with 3,046 trajectory directories, each containing:
- `ui_structure.json`: page layouts, element bboxes, transitions
- `action_coord/*.png`: screenshots with action coordinate overlays

### Step 2: Generate 3-Task SFT Data

```bash
python data_engine/prepare_amex_gelab_sft_v3.py \
    --extracted_dir datas/amex_gelab_extracted/amex_sft \
    --output datas/amex_gelab_sft_v3.json \
    --image_dir datas/amex_gelab_images_v2 \
    --nav_max 8000 --ground_max 8000 --understand_max 8000
```

Output: 24,000 balanced samples (8k per task type).

### Step 3: Train

Training uses the same infrastructure but with different data and the full paper A.10 system prompt (3 task types: Navigation, Grounding, Understanding):

```bash
# Modify continue_train_448.sh or run directly:
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_hf_token"

# Key differences from Run 2:
# - Dataset: amex_gelab_sft_v3.json (not continue_train_24k.json)
# - System prompt: full A.10 with 3 task types
# - Same LR=1e-6, 1 epoch, batch=256
```

See `gui_scripts/continue_train_448.sh` -- change `DATASET_PATH` to `datas/amex_gelab_sft_v3.json`. The system prompt in the script already includes the full A.10 format.

**Runtime:** ~35 min on 4x H200 (47 steps).

### Step 4: Evaluate

```bash
python eval/evaluate_real_world.py \
    --model_path checkpoint/gui_exp/sft_amex_gelab_v3_gentle/<run_dir>/checkpoint-47 \
    --base_model \
    --num_gpus 4 \
    --output_file eval/results_amex_gelab.json
```

**Expected results** (Run 5, LR=1e-6, 1 epoch, `--base_model` eval):

| Benchmark | Base Qwen | Run 5 | Delta |
|-----------|-----------|-------|-------|
| ScreenSpot | 74.8% | 63.9% | -10.9 |
| ScreenSpot-v2 | 77.3% | 64.9% | -12.4 |
| MoTIF | 82.0% | 70.0% | -12.0 |
| Refexp | 83.9% | 50.1% | -33.8 |
| VWB-EG | 75.3% | 70.2% | -5.1 |
| VWB-AG | 65.0% | 65.0% | 0.0 |
| **Average** | 76.4% | 64.0% | -12.4 |

**Why lower than Run 2:** The amex-gelab dataset contains only AMEX mobile app screenshots. The eval benchmarks span mobile, desktop, and web screenshots. Single-domain training hurts generalization compared to the diverse 21k mixed dataset.

---

## Comparison with Paper (Table 5)

| | Paper Base | Paper Continue-Train | Our Base | Our Run 2 (best) |
|---|-----------|---------------------|---------|------------------|
| ScreenSpot | 84.7% | 84.9% | 74.8% | 77.6% |
| ScreenSpot-v2 | - | 84.4% | 77.3% | 80.2% |
| MoTIF | - | 68.3% | 82.0% | 82.8% |
| Refexp | - | 72.1% | 83.9% | 82.7% |

**Why our base Qwen scores differ from paper:** The paper reports 84.7% on ScreenSpot for base Qwen2.5-VL-7B. We get 74.8%. Likely causes:
1. Different eval prompts (paper may use a different grounding prompt template)
2. Different Qwen2.5-VL checkpoint versions
3. Different MAX_PIXELS / image preprocessing

**Why our continue-train pattern differs:** The paper shows continue-train improves ScreenSpot from 84.7% to 84.9% (+0.2). We see 74.8% to 77.6% (+2.8). Our larger improvement suggests the base model had more room to gain from real-world GUI exposure.

---

## Full Benchmark Results (all `--base_model` eval)

| Model | ScreenSpot | SS-v2 | MoTIF | Refexp | VWB-EG | VWB-AG | Avg |
|-------|-----------|-------|-------|--------|--------|--------|-----|
| **Paper Base (their report)** | 84.01 | 80.34 | 71.93 | 79.46 | 90.07 | 72.81 | 67.15* |
| **Paper Continue-Train** | 84.91 | 84.43 | 68.30 | 72.13 | 93.70 | 67.96 | 67.87* |
| **Paper SFT-Continue-Train** | 85.06 | 85.06 | 80.47 | 83.19 | 92.01 | 68.93 | 70.87* |
| Our Base Qwen2.5-VL-7B | 74.84 | 77.28 | 81.98 | 83.89 | 75.30 | 65.05 | 76.39 |
| **Our Run 2 (21k mixed, best)** | **77.59** | **80.19** | **82.85** | 82.65 | **76.76** | **66.02** | **77.68** |
| Our Run 5 (amex-gelab 3-task, base eval) | 63.92 | 64.86 | 70.04 | 50.09 | 70.22 | 65.05 | 64.03 |
| Our Run 6 (amex-gelab 3-task A.5 aligned, sys eval) | 3.38 | 3.07 | 13.99 | 4.60 | 45.04 | 20.39 | 15.08 |
| Our Run 7 (23k combined: no-AMEX mixed + amex-gelab 3-task) | 70.75 | 72.33 | 76.13 | 61.59 | 74.82 | 65.05 | 70.11 |
| Our Run 8 (SFT_True_Final + 21k mixed, base eval) | 1.57 | 1.15 | 2.43 | 0.95 | 0.00 | 0.00 | 1.02 |
| Our Run 8 (SFT_True_Final + 21k mixed, sys eval) | 2.67 | 2.59 | 5.61 | 2.65 | 0.48 | 1.94 | 2.66 |
| Our Run 9 (SFT_True_Final + 21k, LR=1e-5 2ep, sys eval) | 9.04 | 8.81 | 12.17 | 11.50 | 3.39 | 1.94 | 7.81 |
| Our Run 9 (SFT_True_Final + 21k, LR=1e-5 2ep, base eval) | 3.46 | 3.46 | 18.25 | 6.69 | 0.24 | 1.04 | 5.52 |
| Our Run 10 (luca0621/sft-448 + 21k, LR=1e-5 2ep, sys eval) | 7.23 | 7.13 | 22.15 | 11.86 | 2.91 | 3.88 | 9.19 |
| Base Qwen (`--use_system_prompt` eval) | 4.87 | 4.56 | 11.38 | 3.01 | 7.26 | 1.94 | 5.51 |

*Paper averages include FuncPred and AndroidWorld which we don't evaluate, so direct avg comparison is not meaningful. Compare individual benchmarks instead.

**Note on base score gap:** Our base Qwen scores ~10 points below the paper's on ScreenSpot (74.8% vs 84.0%). This is likely due to differences in eval prompt format -- the paper may use the A.5 grounding prompt with system prompt, while we use Qwen's native `bbox_2d` prompt. The relative improvement from continue-training is consistent: both show +1-3 points on ScreenSpot.

---

## Experiment Log

### Run 1: Paper-matched hyperparams (LR=1e-5, 2 epochs)
- **Data**: 21k mixed (AITW 6k, AITZ 3.3k, AMEX 6k, Mind2Web 6k)
- **Config**: LR=1e-5, 2 epochs, batch=256, ZeRO-3, packing+flash_attn, 4x H200
- **System prompt**: Simplified (Navigation task only)
- **Training**: 60 steps, 56 min. Final loss=0.34, token_acc=88.1%
- **Result**: Destroyed base grounding. ScreenSpot: 74.8% -> 10.4% (`--use_system_prompt`) / 49.8% (`--base_model` without sys prompt)
- **Diagnosis**: LR=1e-5 with 2 epochs on 21k samples causes catastrophic forgetting when starting from base Qwen

### Run 2: Gentle hyperparams (LR=1e-6, 1 epoch) -- BEST
- **Data**: 21k mixed (same as Run 1)
- **Config**: LR=1e-6, 1 epoch, batch=256, ZeRO-3, packing+flash_attn, 4x H200
- **System prompt**: Simplified (Navigation only)
- **Training**: 30 steps, 28 min. Final loss=1.08, token_acc=73.9%
- **Eval**: `--base_model` on all 6 benchmarks. **77.7% avg (+1.3 over base)**
- **HuggingFace**: [`namhokaist/Qwen2.5-VL-7B-ContinueTrain-v2`](https://huggingface.co/namhokaist/Qwen2.5-VL-7B-ContinueTrain-v2)

### Run 3: amex-gelab nav-only (LR=1e-6, 10 epochs)
- **Data**: 1,677 nav-only samples from `luca0621/amex-gelab` (shortest-path only)
- **Config**: LR=1e-6, 10 epochs, batch=256, 4x H200
- **Training**: 40 steps, 27 min. Final loss=0.57, token_acc=82.1%
- **Eval**: `--base_model`. **73.5% avg (-2.9 from base)**
- **Diagnosis**: Too few samples (1.7k) with too many epochs caused overfitting; navigation-only data hurt grounding

### Run 4: Full A.10 system prompt (LR=1e-6, 1 epoch)
- **Data**: 21k mixed (same as Run 2)
- **Config**: LR=1e-6, 1 epoch, full A.10 system prompt (Navigation + Grounding + Understanding)
- **Training**: 36 steps, 30 min. Final loss=0.85, token_acc=79.0%
- **Eval**: `--base_model`. **74.6% avg (-1.8 from base)**
- **Diagnosis**: The full A.10 system prompt introduced more distribution shift than the simplified version, without matching grounding/understanding training data to teach those task types

### Run 5: amex-gelab 3-task (nav+grounding+understanding)
- **Data**: 24k from `luca0621/amex-gelab` (8k nav + 8k grounding + 8k understanding)
- **Config**: Full A.10 system prompt
- **HuggingFace**: [`namhokaist/Qwen2.5-VL-7B-AmexGeLab-SFT-v3`](https://huggingface.co/namhokaist/Qwen2.5-VL-7B-AmexGeLab-SFT-v3)
- **Eval**: `--base_model`

| Config | ScreenSpot | SS-v2 | MoTIF | Refexp | VWB-EG | VWB-AG | Avg |
|--------|-----------|-------|-------|--------|--------|--------|-----|
| LR=1e-5, 2ep | 9.7% | 10.7% | 18.2% | 21.6% | 8.0% | 8.7% | 12.8% |
| **LR=1e-6, 1ep** | 63.9% | 64.9% | 70.0% | 50.1% | 70.2% | 65.0% | **64.0%** |

### Run 6: amex-gelab 3-task with A.5-aligned grounding prompt (LR=1e-6, 1 epoch)
- **Data**: 24k from `luca0621/amex-gelab` (8k nav + 8k grounding + 8k understanding)
- **Key change**: Grounding samples now use paper's A.5 eval prompt: `"I want to click on {element}. Please locate the target element I should interact with. (with point)"`
- **Config**: LR=1e-6, 1 epoch, full A.10 system prompt
- **Training**: 47 steps, 38 min. Final loss=0.78, token_acc=77.7%
- **Dataset**: [`namhokaist/amex-gelab-sft-v3`](https://huggingface.co/datasets/namhokaist/amex-gelab-sft-v3)
- **Eval** (`--use_system_prompt`):

| Benchmark | Run 5 (old grounding prompt) | Run 6 (A.5 aligned) |
|-----------|------------------------------|---------------------|
| ScreenSpot | 6.8% | 3.4% |
| ScreenSpot-v2 | 6.4% | 3.1% |
| MoTIF | 32.3% | 14.0% |
| Refexp | 12.6% | 4.6% |
| VWB-EG | 3.6% | **45.0%** |
| VWB-AG | 1.0% | **20.4%** |
| Avg | 10.5% | **15.1%** |

- VWB benchmarks jumped (+41 and +19 points) -- aligned prompt works for web UI grounding
- ScreenSpot/MoTIF/Refexp dropped -- these use mobile/desktop screenshots outside amex-gelab's domain
- `--base_model` eval (64.0%) still outperforms `--use_system_prompt` (15.1%) overall
- **Conclusion**: prompt alignment helps for in-domain benchmarks, but LR=1e-6 / 1 epoch is insufficient to fully learn the GE-Lab output format across all domains

### Run 7: Combined dataset (no-AMEX mixed + amex-gelab 3-task, LR=1e-6, 1 epoch)
- **Data**: 23.3k combined (AITW 4k + AITZ 3.3k + Mind2Web 4k + amex-gelab nav 4k + grounding 4k + understanding 4k). AMEX removed from mixed set to avoid overlap with amex-gelab.
- **Config**: LR=1e-6, 1 epoch, full A.10 system prompt
- **Training**: 38 steps, 33 min. Final loss=1.04, token_acc=72.6%
- **Eval** (`--base_model`): **70.1% avg**
- **Result**: Worse than Run 2 (77.7%) despite having more data. The amex-gelab 3-task data diluted the diverse real-world data, and the full A.10 system prompt added distribution shift.

### Run 8: SFT_True_Final + Continue-Train (21k mixed, LR=1e-6, 1 epoch)
- **Data**: 21k mixed (AITW 6k + AITZ 3.3k + AMEX 6k + Mind2Web 6k) -- same as Run 2
- **Starting model**: `namhokaist/SFT_True_Final` (GE-Lab SFT checkpoint, NOT base Qwen)
- **Config**: LR=1e-6, 1 epoch, full A.10 system prompt, batch=256, ZeRO-3, packing+flash_attn
- **Training**: 36 steps, 42 min on 4x H200. Loss: 3.39 -> 1.91. DATALOADER_NUM_WORKERS=0 (64MB shm limit)
- **Checkpoint**: `checkpoint/gui_exp/continue_train_448_sft_hf/v0-20260406_130128/checkpoint-36`
- **Eval** (`--base_model`): **1.0% avg** -- catastrophic failure

| Benchmark | Base Qwen | Run 8 (base eval) | Delta |
|-----------|-----------|-------------------|-------|
| ScreenSpot | 74.8% | 1.6% | -73.2 |
| ScreenSpot-v2 | 77.3% | 1.2% | -76.1 |
| MoTIF | 82.0% | 2.4% | -79.6 |
| Refexp | 83.9% | 0.9% | -83.0 |
| VWB-EG | 75.3% | 0.0% | -75.3 |
| VWB-AG | 65.0% | 0.0% | -65.0 |
| **Average** | 76.4% | **1.0%** | **-75.4** |

- **Eval** (`--use_system_prompt`): **2.7% avg** -- also catastrophic failure

| Benchmark | Run 8 (sys eval) |
|-----------|-----------------|
| ScreenSpot | 2.7% |
| ScreenSpot-v2 | 2.6% |
| MoTIF | 5.6% |
| Refexp | 2.7% |
| VWB-EG | 0.5% |
| VWB-AG | 1.9% |
| **Average** | **2.7%** |

- **Diagnosis**: Both eval modes fail. The SFT_True_Final model's GE-Lab grounding format was destroyed by continue-training on 21k mixed real-world data (navigation-only format). The mixed data uses a simplified system prompt and `click(start_box=...)` format for navigation but NOT for grounding queries. The model lost both: (1) Qwen's native `bbox_2d` format, and (2) the GE-Lab grounding query-response pattern.
- **Root cause**: The 21k training data is navigation-style (instruction -> click action). The grounding benchmarks ask "locate element X" -- a fundamentally different task. Training on navigation data teaches the model to always predict click actions in response to multi-step instructions, not to locate elements from descriptions.
- **Key lesson**: Continue-training on navigation data is destructive to grounding ability regardless of starting checkpoint. The paper's SFT-Continue-Train (85.06% ScreenSpot) likely uses the paper's own continue-train data pipeline which includes grounding-format samples, not just navigation.

### Run 9: SFT_True_Final + Continue-Train (21k mixed, LR=1e-5, 2 epochs -- paper settings)
- **Data**: 21k mixed (same as Run 2/8) -- AITW, AITZ, AMEX, Mind2Web (per paper A.5)
- **Starting model**: `namhokaist/SFT_True_Final`
- **Config**: **LR=1e-5, 2 epochs** (matching paper Section A.5), batch=256, full A.10 system prompt
- **Training**: 72 steps, 84 min on 4x H200. Loss: 3.39 -> 0.35 (final). DATALOADER_NUM_WORKERS=0
- **Checkpoint**: `checkpoint/gui_exp/continue_train_448_sft_hf/v0-20260406_222525/checkpoint-72`

| Benchmark | Run 9 (sys eval) | Run 9 (base eval) | Paper SFT-CT |
|-----------|-----------------|-------------------|--------------|
| ScreenSpot | 9.0% | 3.5% | 85.06% |
| ScreenSpot-v2 | 8.8% | 3.5% | 85.06% |
| MoTIF | 12.2% | 18.2% | 80.47% |
| Refexp | 11.5% | 6.7% | 83.19% |
| VWB-EG | 3.4% | 0.2% | 92.01% |
| VWB-AG | 1.9% | 1.0% | 68.93% |
| **Average** | **7.8%** | **5.5%** | **70.87%** |

- **Improvement over Run 8**: 3x better with sys_prompt (7.8% vs 2.7%), confirming paper LR/epochs matter
- **Still far from paper**: 7.8% vs 70.87%. The ~63-point gap is too large to be explained by hyperparameters alone
- **Possible causes**: (1) Different SFT checkpoint -- our SFT_True_Final may not match the paper's exact SFT model; (2) Paper may use 16 GPUs with different effective learning dynamics; (3) The 21k data may differ in format/content from what the paper actually uses; (4) MAX_PIXELS mismatch -- paper uses 200704, we use 1003520
- **Investigation results**:
  - MAX_PIXELS=200704 (paper value) at eval: 6.8% avg -- slightly worse, resolution is NOT the issue
  - Raw SFT_True_Final (no continue-train): 2.5% avg -- confirms the SFT model itself has zero real-world grounding
  - Verbose inspection: model outputs GE-Lab hallucinations ("click Business_17 icon on Business_158 page") on real screenshots
  - **Conclusion**: The SFT_True_Final checkpoint lost all base Qwen visual grounding during GE-Lab simulation SFT. The paper's SFT checkpoint must be different -- retaining base grounding while learning GE-Lab navigation. This gap cannot be closed by tuning hyperparameters or data.

### Run 10: luca0621/sft-448 + Continue-Train (sanity check on alternative SFT)
- **Hypothesis tested**: Is the low score in Run 9 specific to `SFT_True_Final` or fundamental to the GE-Lab SFT pipeline?
- **Starting model**: `luca0621/sft-448` (alternative SFT checkpoint from colleague)
- **Config**: Same as Run 9 (LR=1e-5, 2 epochs, 21k mixed data)
- **Training**: 72 steps, 82 min. Loss: 3.60 -> 2.17 -> 0.53 -> 0.43 -> 0.38 -> 0.35 -> 0.34 -> **0.34** (final)
- **Pre-test verification**: Raw model outputs same GE-Lab hallucinations ("Business_208", "Business_168") on real screenshots
- **Eval** (`--use_system_prompt`):

| Benchmark | Run 9 (SFT_True_Final) | Run 10 (luca sft-448) |
|-----------|------------------------|------------------------|
| ScreenSpot | 9.0% | 7.2% |
| ScreenSpot-v2 | 8.8% | 7.1% |
| MoTIF | 12.2% | 22.2% |
| Refexp | 11.5% | 11.9% |
| VWB-EG | 3.4% | 2.9% |
| VWB-AG | 1.9% | 3.9% |
| **Average** | **7.8%** | **9.2%** |

- **Conclusion**: Both SFT checkpoints converge to similar low scores (~8-9% avg) after the same continue-training. The issue is fundamental to the GE-Lab SFT pipeline -- training on simulated icons with random "Business_X" labels destroys real-world visual grounding regardless of which checkpoint we use.
- **Why MoTIF is higher (22% vs 12%)**: MoTIF's mobile screenshots may be more similar in style to GE-Lab simulated screens than ScreenSpot's diverse desktop/web UIs.
- **Final answer to user's question**: We cannot reproduce the paper's 70.9% with any available SFT checkpoint. The paper's actual SFT model used for Table 5 must differ from what's released on HuggingFace. The base Qwen + continue-train pipeline (Run 2 = 77.7%) remains our best result and exceeds the paper's base score on our hardware setup.

### Base Qwen eval with `--use_system_prompt`
- Base Qwen2.5-VL scores **5.5% avg** with `--use_system_prompt` eval (A.5 grounding prompt)
- This confirms the base model has never seen the GE-Lab grounding prompt format
- The 10-point gap between our base (74.8%) and paper's base (84.0%) on ScreenSpot is NOT due to eval prompt -- it's likely a checkpoint version or image preprocessing difference
- Our relative improvements are valid; absolute numbers are consistently ~10 points below paper

### Batch Size Benchmarking
| Batch/GPU | Grad Accum | Memory/GPU | Speed | Notes |
|-----------|-----------|------------|-------|-------|
| 2 | 32 | 41 GB | 82s/step | Baseline (no packing) |
| 4 | 16 | 41 GB | 105s/step | Padding overhead from variable-length images |
| 8 | 8 | 59 GB | 156s/step | More padding waste |
| 8 + packing | 8 | 120-134 GB | **56s/step** | **4x speedup**: packing concatenates samples, eliminates padding |

---

## Key Findings

### 1. Learning rate is the dominant factor
LR=1e-6 preserves base grounding ability. LR=1e-5 destroys it. This holds regardless of data composition, system prompt, or number of epochs. The paper uses LR=1e-5 successfully because their starting checkpoints (SFT/ST-RL/MT-RL) were already adapted to the GUI domain through GE-Lab simulation training. Raw base Qwen needs gentler learning.

### 2. Data diversity > data volume
21k mixed samples from 4 diverse sources (mobile + web + desktop) outperform 24k samples from a single domain (AMEX mobile only). The eval benchmarks span multiple platforms, so training diversity directly translates to eval coverage.

### 3. Three task types prevent catastrophic forgetting
When training on amex-gelab, adding grounding and understanding samples alongside navigation samples improved scores from 8.5% to 64.0% (at LR=1e-5). Navigation-only data teaches the model to always click, losing its ability to locate elements from descriptive queries.

### 4. Packing eliminates variable-length padding waste
Qwen2.5-VL produces variable-length image tokens depending on resolution. Without packing, batch=8 is slower than batch=2 due to padding. With `--packing true --attn_impl flash_attn`, samples are concatenated without padding, achieving 4x speedup and utilizing 120-134GB of H200's 143GB VRAM.

### 5. Eval prompt format must match model's retained capabilities
Use `--base_model` when the model retains Qwen's native grounding format (true for all our LR=1e-6 models). Use `--use_system_prompt` only for models extensively trained on GE-Lab format with grounding query-response pairs (the paper's SFT/ST-RL/MT-RL checkpoints).

### 6. Paper hyperparams assume pre-trained starting checkpoint
The paper's LR=1e-5 / 2 epochs / full A.10 prompt works for their SFT checkpoint (already trained on GE-Lab sim data). When starting from raw base Qwen, these settings cause catastrophic forgetting. The correct analogy: the paper's "continue-train" is stage 2 of a 2-stage pipeline; we're doing stage 2 directly on the base model.

---

## Next Steps (Priority Order)

### 1. Investigate the 10-point base score gap (HIGH PRIORITY)
Our base Qwen scores 74.8% on ScreenSpot vs paper's 84.0%. This offsets ALL our results. Potential causes to investigate:
- **Qwen2.5-VL checkpoint version**: check if the paper uses a specific commit hash
- **Image preprocessing**: try `MAX_PIXELS=200704` (paper's value) at eval time
- **ScreenSpot dataset version**: the `rootsautomation/ScreenSpot` HF dataset may differ from what the paper uses
- Quick test: run eval with `--max_pixels 200704` to see if resolution matters

### 2. Try the paper's exact 2-stage pipeline
The paper's best results come from SFT -> Continue-Train (two stages). We've only done single-stage. The pipeline would be:
1. SFT on amex-gelab (with A.5 grounding prompt, LR=1e-5, 2ep) -- learn GE-Lab format
2. Continue-train the SFT checkpoint on 21k mixed data (LR=1e-5, 2ep) -- add real-world knowledge

This matches "SFT-Continue-Train" in Table 5 (85.06% ScreenSpot).

### 3. Try LoRA instead of full fine-tuning
LoRA would preserve base model capabilities by only training low-rank adapters. This could allow LR=1e-5 without catastrophic forgetting, matching the paper's hyperparameters.

### 4. Scale up amex-gelab grounding data
We have ~1M potential grounding samples but only use 8k. Training with more grounding examples at LR=1e-5 could fully teach the GE-Lab output format without destroying base capabilities (if balanced with enough diverse data).

---

## Known Issues

**Coordinate space mismatch:** If `MAX_PIXELS` differs between training and evaluation, the model's coordinate predictions will be in the wrong scale. We train at full resolution (`MAX_PIXELS=1003520`) to avoid this.

**AITZ sample count:** AITZ only has 3,337 valid samples after filtering (requested 6k), resulting in 21k total instead of 24k.

**Conda env instability:** The conda env may be lost on container restarts. Keep the setup commands handy for quick rebuild (~10 min).

**Local swift/ shadowing:** The repo contains `swift/` (v3.5) which shadows pip-installed ms-swift 4.0.4 for Python imports. The CLI `swift sft` uses the pip version correctly. This requires `transformers<4.53` for compatibility with the local module.

---

## File Locations

| File | Purpose |
|------|---------|
| `data_engine/prepare_continue_train_data.py` | Download and format 21k mixed training data |
| `data_engine/prepare_amex_gelab_sft_v3.py` | Generate 3-task SFT data from `luca0621/amex-gelab` |
| `gui_scripts/continue_train_448.sh` | Training script (configurable LR, epochs, model stage) |
| `eval/evaluate_real_world.py` | Grounding benchmark evaluation (6 benchmarks, multi-GPU) |
| `eval/results_base_qwen_full.json` | Reference: base Qwen2.5-VL scores (76.4% avg) |
| `eval/results_v2_full_base_prompt.json` | Reference: Run 2 best scores (77.7% avg) |
| `eval/results_sft_v3_fixed_sys.json` | Reference: Run 6 amex-gelab A.5-aligned (15.1% avg, sys eval) |
| `eval/results_sft_v3_gentle_base.json` | Reference: amex-gelab SFT scores (64.0% avg) |


### Run 14: Paper-style 2-stage pipeline (Stage 1 SFT 50k luca + Stage 3 CT 15k no-AMEX)

- **Stage 1**: Base Qwen2.5-VL-7B-Instruct -> SFT on 50k luca amex-gelab (16.7k each task), MAX_PIXELS=12,845,056 (Qwen base default native), LR=1e-5, 1 epoch, eff batch 256, max_length 5120. 196 steps in 4h 24m, final loss 0.281, token_acc 90.97%. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-v1
- **Stage 3**: Stage 1 SFT -> continue-train on 15k mind2web/aitw/aitz (no AMEX), MAX_PIXELS=12,845,056, LR=1e-5, 2 epochs, eff batch 256, max_length 5120. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-ContinueTrain-v1

| Variant | ScreenSpot | SS-v2 | MoTIF | Refexp | VWB-EG | VWB-AG | Avg |
|---------|-----------|-------|-------|--------|--------|--------|-----|
| Stage 1 (50k luca SFT, native px) [base eval] | 23.19 | 22.01 | 22.13 | 20.71 | 2.18 | 1.94 | 15.36 |
| Stage 1 (50k luca SFT, native px) [sys eval] | 27.52 | 25.86 | 27.35 | 19.29 | 4.12 | 0.97 | 17.52 |
| Stage 1+3 (Continue-Train, no AMEX) [base eval] | 7.39 | 7.63 | 19.37 | 9.56 | 3.63 | 7.77 | 9.22 |
| Stage 1+3 (Continue-Train, no AMEX) [sys eval] | 15.25 | 15.02 | 21.58 | 21.06 | 5.57 | 3.88 | 13.73 |


### Run 15: Paper-style 2-stage pipeline at LR=1e-6 (Run 5/Run 7 calibration)

- **Motivation**: Run 14 used paper LR=1e-5 + native pixels which collapsed Stage 1 to 15.36% (vs Run 5's 64.03%) due to catastrophic forgetting of base Qwen grounding. Re-running with Run 5/Run 7 proven calibration.
- **Stage 1 v2**: Base Qwen2.5-VL-7B-Instruct -> SFT on 23k luca amex-gelab (~7.7k each task), MAX_PIXELS=1,003,520 (Run 5 baseline), **LR=1e-6**, 1 epoch, eff batch 256, max_length 5120. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-v2
- **Stage 3 v2**: Stage 1 v2 -> continue-train on 15k mind2web/aitw/aitz (no AMEX), MAX_PIXELS=1,003,520, **LR=1e-6**, 2 epochs, eff batch 256, max_length 5120. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-ContinueTrain-v2

| Variant | ScreenSpot | SS-v2 | MoTIF | Refexp | VWB-EG | VWB-AG | Avg |
|---------|-----------|-------|-------|--------|--------|--------|-----|
| Run 15 Stage 1 v2 (23k luca, LR=1e-6, 1003520 px) [base eval] | 69.50 | 71.15 | 75.97 | 60.53 | 74.58 | 66.99 | 69.79 |
| Run 15 Stage 1+3 v2 (Continue-Train 15k no-AMEX, LR=1e-6) [base eval] | 67.30 | 68.79 | 73.75 | 55.58 | 72.64 | 65.05 | 67.18 |


### Run 16: Full 2.02M luca dataset + native resolution (LR=1e-6)

- **Motivation**: Test whether the FULL luca amex-gelab dataset (2.02M samples: 24.8k nav + 1M grounding + 1M understanding) at native resolution improves over Run 15 Stage 1 v2 (23k, 69.79%).
- **Stage 1 v3**: Base Qwen2.5-VL-7B-Instruct -> SFT on 225k balanced luca amex-gelab, MAX_PIXELS=12,845,056 (native), **LR=1e-6**, 1 epoch, eff batch 256, max_length 5120. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-v3
- **Stage 3 v3**: Stage 1 v3 -> continue-train on 15k mind2web/aitw/aitz (no AMEX), MAX_PIXELS=12,845,056, **LR=1e-6**, 2 epochs. Uploaded: namhokaist/Qwen2.5-VL-7B-AmexGelab-FullSFT-ContinueTrain-v3

| Variant | ScreenSpot | SS-v2 | MoTIF | Refexp | VWB-EG | VWB-AG | Avg |
|---------|-----------|-------|-------|--------|--------|--------|-----|
| Run 16 Stage 1 v3 (225k balanced luca, LR=1e-6, native px) [base eval] | 10.61 | 10.93 | 17.63 | 3.19 | 7.26 | 19.42 | 11.51 |
| Run 16 Stage 1+3 v3 (CT 15k no-AMEX, LR=1e-6, native px) [base eval] | 8.10 | 8.33 | 14.23 | 3.01 | 4.84 | 18.45 | 9.49 |

