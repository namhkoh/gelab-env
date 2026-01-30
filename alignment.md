# GE-Lab Paper vs Codebase Alignment Analysis

**Document Purpose**: Systematic comparison between what is described in the GE-Lab paper and what is available in the current codebase.

**Analysis Confidence**: 0.9

---

## 1. Executive Summary

The codebase provides the **infrastructure** for GE-Lab training but is **missing the full datasets** described in the paper. The core components (swift training framework, data_engine for environment generation, evaluation scripts) are present, but the actual training data needs to be generated or obtained separately.

---

## 2. Environment Configuration

### Paper Description (Env-Base)
| Parameter | Paper Value |
|-----------|-------------|
| Tree Depth | 7 levels |
| Branching Structure | [5, 3, 2, 2, 1, 1, 0] per level |
| Total Pages | ~275 nodes |
| Subtree Allocation | 2 SFT + 2 RL + 1 Test |
| Canvas Size | 1179 x 2556 pixels |
| Icon Size | 200 x 200 pixels |

### Codebase Status
- **tree.py**: Correctly implements [5, 3, 2, 2, 1, 1] node distribution (line 939)
- **Canvas**: 1179 x 2556 (LayoutGenerator, line 625)
- **Icon**: 200 x 200 (line 626-627)
- **Status**: ALIGNED - Environment generator matches paper specifications

---

## 3. Dataset Comparison

### Paper Dataset Sizes

| Dataset Type | Paper Size | Codebase Size | Gap |
|--------------|-----------|---------------|-----|
| SFT Training | 60,864 samples | 2 samples | **99.997% missing** |
| ST-RL Training | 12,439 samples | 2 samples | **99.98% missing** |
| MT-RL Training | 2,162 tasks | 2 samples | **99.9% missing** |
| Icon Captioning | 2,320 instances | 0 | **100% missing** |
| Icon Grounding | 2,320 instances | 0 | **100% missing** |
| Interactive Benchmark | 2,162 tasks | 1 sample | **99.95% missing** |
| Test Images | ~275 pages | 4 images | **98.5% missing** |

### Paper Dataset Composition (SFT)
According to Appendix A.1:
- **Path data**: 12,439 per subtree x 2 subtrees = 24,878 samples
- **Edge data**: 274 per subtree x 5 subtrees = 1,370 samples
- **Icon Captioning**: 2,320 samples
- **Icon Grounding**: 2,320 samples
- **Additional grounding/understanding data** for Env-Base icons

### Available Data Files
```
datas/
├── sft.json         # 2 samples (demonstration only)
├── st_rl.json       # 2 samples (demonstration only)  
├── mt_rl.json       # 2 samples (demonstration only)
├── test.json        # 1 sample (demonstration only)
└── images/
    ├── page_0.png
    ├── page_1.png
    ├── page_3.png
    └── page_161.png
```

---

## 4. Model and Training Parameters

### Foundation Model
| Aspect | Paper | Codebase | Status |
|--------|-------|----------|--------|
| Model | Qwen2.5-VL-7B-Instruct | Configurable via MODEL_PATH | ALIGNED |
| Training Framework | ms-swift | swift/ module | ALIGNED |

### SFT Hyperparameters
| Parameter | Paper (Table 7) | Codebase (sft.sh) | Status |
|-----------|-----------------|-------------------|--------|
| Learning Rate | 1e-5 | 1e-5 | ALIGNED |
| LR Schedule | Cosine decay | Cosine (implicit) | ALIGNED |
| Batch Size | 2 | 2 | ALIGNED |
| Gradient Accum | 2 | 2 | ALIGNED |
| Warmup Ratio | 0.05 | 0.05 | ALIGNED |
| Epochs | 1 | 1 | ALIGNED |
| Max Pixels | 200704 | 200704 | ALIGNED |

### RL Hyperparameters
| Parameter | Paper (Table 8) | Expected | Status |
|-----------|-----------------|----------|--------|
| Learning Rate | 1e-6 | Needs verification | TBD |
| Batch Size | 8 | Needs verification | TBD |
| Epochs | 5 | Needs verification | TBD |
| Num Generations | 8 | Needs verification | TBD |
| Temperature | 1.2 | Needs verification | TBD |

---

## 5. Reward Functions

### Paper Reward Components
1. **Action Type Reward** (r_type): Correct action type (click/complete)
2. **Coordinate Accuracy Reward** (r_coord): Click within target bounding box
3. **Intent Matching Reward** (r_intent): Icon name match in explanation
4. **Format Reward** (r_format): Output format compliance

### Codebase Implementation
From `gui_scripts/single_turn_rl.sh` (inferred):
```bash
--reward_funcs web_action_match web_coordinate_match web_intent_match
```

**Status**: Partially aligned - needs to verify swift plugin implementation

---

## 6. Evaluation Benchmarks

### Paper Benchmarks
| Benchmark | Type | Paper Status | Codebase Status |
|-----------|------|--------------|-----------------|
| GE-Lab Static (ID) | Internal | Used | Needs generation |
| GE-Lab Static (OOD) | Internal | Used | Needs generation |
| GE-Lab Interactive | Internal | Used | Needs generation |
| ScreenSpot | External | Evaluated | NOT INCLUDED |
| ScreenSpot-v2 | External | Evaluated | NOT INCLUDED |
| FuncPred | External | Evaluated | NOT INCLUDED |
| MoTIF | External | Evaluated | NOT INCLUDED |
| Refexp | External | Evaluated | NOT INCLUDED |
| VWB AG/EG | External | Evaluated | NOT INCLUDED |
| AndroidWorld | External | Evaluated | NOT INCLUDED |

### Evaluation Scripts
- `eval/inference_qwen2p5_mixed_vllm.py`: Present and functional
- `eval/calculate_score_refine.py`: Present and functional
- **Status**: Scripts ready, but need test data

---

## 7. Key Gaps Summary

### Critical (Must Resolve for SFT Training)
1. **Full training dataset generation** - Need to run data_engine to create complete environment and generate trajectory data
2. **Path data synthesis** - Need script to generate shortest/redundant path trajectories
3. **Edge data generation** - Need to extract single-step transitions
4. **Icon grounding/captioning data** - Need to generate from UI structure

### Important (For Complete Reproduction)
5. **OOD Environment Variants** (Env-Image, Env-Name, Env-Position, Env-Noise)
6. **Real-world evaluation datasets** (AITW, AITZ, AMEX, Mind2Web)
7. **External benchmark datasets** (ScreenSpot, AndroidWorld, etc.)

### Nice-to-Have
8. **Multi-node training scripts** optimization
9. **Curriculum learning** implementation for MT-RL

---

## 8. Path Forward

### Phase 1: Environment Generation
1. Run `python data_engine/tree.py` to generate full Env-Base
2. This creates: ui_structure.json, ui_structure_layer.json, pages/, ui_topology.png

### Phase 2: Dataset Synthesis
1. Create script to generate path data from navigation graph
2. Create script to generate edge data (single-step transitions)
3. Create script to generate icon captioning/grounding data
4. Split data by subtree according to paper allocation

### Phase 3: Training
1. SFT on generated data
2. ST-RL with GRPO
3. MT-RL with sparse rewards

### Phase 4: Evaluation
1. Generate static benchmark from test subtree
2. Generate interactive benchmark tasks
3. Optionally integrate external benchmarks

---

## 9. Environment Setup Requirements

### GPU Infrastructure
- **Paper**: 16 NVIDIA A800 GPUs
- **Available**: 8 NVIDIA H200 GPUs (143GB each)
- **CUDA Version**: 13.0
- **Status**: Hardware is MORE capable than paper setup

### Software Dependencies
- Python 3.8+
- PyTorch with CUDA 13.0 support
- transformers >= 4.33
- ms-swift framework
- vllm (for inference)
- qwen-vl-utils

---

## 10. Confidence Levels

| Component | Confidence | Notes |
|-----------|------------|-------|
| Environment Generator | 0.95 | Code matches paper specs |
| Training Framework | 0.90 | ms-swift implementation present |
| SFT Parameters | 0.95 | Direct match with paper |
| RL Parameters | 0.80 | Need to verify RL scripts |
| Reward Functions | 0.75 | Need to check swift plugin |
| Dataset Generation | 0.70 | Scripts exist but need pipeline |
| Evaluation Pipeline | 0.85 | Scripts present, need data |

---

*Document generated: 2026-01-29*
*Analyst: AI Assistant*
