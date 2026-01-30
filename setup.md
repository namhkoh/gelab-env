# GE-Lab Reproduction Guide

This document provides step-by-step instructions to reproduce the GUI Exploration Lab (GE-Lab) experiments.

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Overview](#2-data-overview)
3. [Training](#3-training)
4. [Evaluation](#4-evaluation)
5. [Model Checkpoints](#5-model-checkpoints)

---

## 1. Environment Setup

### Prerequisites

- NVIDIA GPU with at least 40GB VRAM (A100 recommended)
- CUDA 12.x
- Python 3.10+

### Installation

```bash
# Create conda environment
conda create -n gelab python=3.10 -y
conda activate gelab

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
cd /path/to/ofz
pip install -r requirements.txt

# Install ms-swift for training
pip install ms-swift[llm] -U

# Install additional dependencies
pip install qwen-vl-utils transformers accelerate
```

---

## 2. Data Overview

### Directory Structure

```
ofz/
├── data_engine/
│   └── ui_environment_448/
│       └── latest/
│           ├── pages/              # 231 page images (448x448)
│           ├── ui_structure.json   # Navigation graph metadata
│           └── config.json         # Environment config
├── datas/
│   └── 448_paper/
│       ├── sft_aligned.json        # SFT training data
│       ├── st_rl_aligned.json      # ST-RL training data
│       ├── mt_rl_aligned.json      # MT-RL training data
│       ├── test_id_edge_fixed.json # ID Edge test (correct format)
│       ├── test_ood_edge_fixed.json# OOD Edge test (correct format)
│       ├── test_id_path.json       # ID Path test
│       └── test_ood_path.json      # OOD Path test
```

### Data Splits

| File | Purpose | Samples |
|------|---------|---------|
| `sft_aligned.json` | SFT Training | 30,888 |
| `st_rl_aligned.json` | ST-RL Training | 90 |
| `mt_rl_aligned.json` | MT-RL Training | 2,200 |
| `test_id_edge_fixed.json` | ID Edge Test | 90 |
| `test_ood_edge_fixed.json` | OOD Edge Test | 45 |
| `test_id_path.json` | ID Path Test | 435 |
| `test_ood_path.json` | OOD Path Test | 462 |

### Subtree Organization

The environment has 5 subtrees:
- **Subtrees 0-1**: SFT training (path data)
- **Subtrees 2-3**: RL training
- **Subtree 4**: OOD testing (held out)

Note: Edge data from ALL subtrees (including subtree 4) is included in SFT training per the paper design.

---

## 3. Training

### 3.1 SFT Training

```bash
cd /path/to/ofz

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --train_type full \
    --dataset datas/448_paper/sft_aligned.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir checkpoint/gui_exp/sft_448/my_run \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --deepspeed zero2
```

Expected training time: ~90 minutes on 8x A100 GPUs.

### 3.2 ST-RL Training (Optional)

Requires SFT checkpoint as base model.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift rlhf \
    --rlhf_type grpo \
    --model /path/to/sft/checkpoint \
    --dataset datas/448_paper/st_rl_path_only.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-6 \
    --num_generations 8 \
    --temperature 1.2 \
    --output_dir checkpoint/gui_exp/st_rl/my_run
```

---

## 4. Evaluation

### 4.1 Edge Evaluation (Single-Step)

```python
import json
import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Load model from Hugging Face
model_path = "namhokaist/gelab-sft-448"  # or local path
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)
model.eval()

def evaluate_edge(test_file):
    with open(test_file) as f:
        test_data = json.load(f)
    
    correct = 0
    total = 0
    
    for sample in tqdm(test_data):
        image_path = sample['images'][0]
        user_msg = sample['messages'][0]['content']
        bbox = sample['bbox_norm']
        icon_name = sample['icon_name']
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_msg.replace("<image>", "")}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Check correctness
        icon_match = re.search(r'click\s+(.*?)\s+icon', pred)
        coord_match = re.search(r'\((\d+),(\d+)\)', pred)
        
        if icon_match and coord_match:
            pred_icon = icon_match.group(1).strip()
            pred_x = int(coord_match.group(1))
            pred_y = int(coord_match.group(2))
            
            x_min, y_min, x_max, y_max = bbox
            
            icon_correct = pred_icon == icon_name
            coord_correct = x_min <= pred_x <= x_max and y_min <= pred_y <= y_max
            
            if icon_correct and coord_correct:
                correct += 1
        
        total += 1
    
    return correct / total * 100

# Run evaluation
id_edge_acc = evaluate_edge('datas/448_paper/test_id_edge_fixed.json')
ood_edge_acc = evaluate_edge('datas/448_paper/test_ood_edge_fixed.json')

print(f"ID Edge: {id_edge_acc:.2f}%")
print(f"OOD Edge: {ood_edge_acc:.2f}%")
```

### 4.2 Expected Results

| Metric | Our SFT | Paper SFT |
|--------|---------|-----------|
| ID Edge | 87.78% | 94.82% |
| OOD Edge | 66.67% | 64.55% |

---

## 5. Model Checkpoints

### Hugging Face (Recommended)

The best SFT model is available on Hugging Face:

**https://huggingface.co/namhokaist/gelab-sft-448**

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("namhokaist/gelab-sft-448")
processor = AutoProcessor.from_pretrained("namhokaist/gelab-sft-448")
```

### Local Path (if training yourself)

```
/root/.cursor/worktrees/gelab-env__SSH__vast_/ofz/checkpoint/gui_exp/sft_448/v1-20260130-013131/v0-20260130-013206/checkpoint-956
```

### Checkpoint Contents

```
checkpoint-956/
├── config.json
├── generation_config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json
```

### Compressed Archive

A compressed version is available at:
```
checkpoint/gui_exp/sft_448/v1-20260130-013131/v0-20260130-013206/checkpoint-956.tar.gz
```

Size: ~10GB

---

## Quick Reference

### Training Data
```
datas/448_paper/sft_aligned.json
```

### Evaluation Data (use these for proper evaluation)
```
datas/448_paper/test_id_edge_fixed.json   # ID Edge (90 samples)
datas/448_paper/test_ood_edge_fixed.json  # OOD Edge (45 samples)
datas/448_paper/test_id_path.json         # ID Path (435 samples)
datas/448_paper/test_ood_path.json        # OOD Path (462 samples)
```

### UI Environment Images
```
data_engine/ui_environment_448/latest/pages/   # 231 page images
```

### Best Model Checkpoint
```
checkpoint/gui_exp/sft_448/v1-20260130-013131/v0-20260130-013206/checkpoint-956
```

---

## Important Notes

1. **Image Size**: All images must be 448x448 pixels (matching `max_pixels=200704` in Qwen2.5-VL).

2. **Test Format**: Use `*_fixed.json` files for Edge evaluation. The original files had incorrect format that inflated accuracy to 100%.

3. **Edge vs Path**:
   - Edge = single-step navigation (click one icon)
   - Path = multi-step navigation (sequence of clicks)

4. **ID vs OOD**:
   - ID = subtrees 0-1 (seen during path training)
   - OOD = subtree 4 (held out, never seen paths)
   - Note: Edge data from ALL subtrees is in SFT training
