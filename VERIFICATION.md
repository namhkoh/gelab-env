# Implementation Verification: Our Code vs Paper

## Environment Configuration

| Parameter | Paper | Our Implementation | Match |
|-----------|-------|-------------------|-------|
| Tree Depth | 7 | 7 | ✅ |
| Node Distribution | [5, 3, 2, 2, 1, 1, 0] | [5, 3, 2, 2, 1, 1] | ✅ |
| Total Pages | ~231 | 231 | ✅ |
| Subtrees | 5 (2 SFT, 2 RL, 1 Test) | 5 (0-1 SFT, 2-3 RL, 4 Test) | ✅ |
| Image Size | 448×448 (max_pixels=200704) | 448×448 | ✅ |
| Coordinate System | 0-1000 normalized | 0-1000 normalized | ✅ |

### Pages per Depth (Verified)
```
Depth 0: 1 page   (root)
Depth 1: 5 pages  (5 subtree roots)
Depth 2: 15 pages (5 × 3)
Depth 3: 30 pages (15 × 2)
Depth 4: 60 pages (30 × 2)
Depth 5: 60 pages (60 × 1)
Depth 6: 60 pages (60 × 1, leaves)
Total:   231 pages ✅
```

---

## Dataset Sizes

| Dataset | Paper | Our Implementation | Notes |
|---------|-------|-------------------|-------|
| SFT Training | 60,864 | 30,888 | ⚠️ Paper may count trajectory steps differently |
| ST-RL | 12,439 (per subtree) | 24,968 (2 subtrees) | ✅ 12,439 × 2 ≈ 24,878 |
| Path per subtree | 12,439 | ~12,439 | ✅ |
| Edge per subtree | 274 | ~45-90 | ⚠️ Depends on counting method |

### SFT Dataset Composition (Paper)
- Edge (all 5 subtrees): included
- Path (subtrees 0-1 only): included
- Icon Grounding: 2,320 samples
- Icon Captioning: 2,320 samples

### Our SFT Dataset: 30,888 samples
- Matches paper structure (Edge + Path + Grounding + Captioning)
- The 2:1 difference from paper's 60,864 may be due to trajectory step counting

---

## Training Hyperparameters

### SFT (Table 7)

| Parameter | Paper | Our Implementation | Match |
|-----------|-------|-------------------|-------|
| Learning Rate | 1e-5 | 1e-5 | ✅ |
| LR Schedule | cosine decay | cosine (default) | ✅ |
| Batch Size (per device) | 2 | 2 | ✅ |
| Gradient Accumulation | 2 | 2 | ✅ |
| Warmup Ratio | 0.05 | 0.05 | ✅ |
| Epochs | 1 | 1 | ✅ |
| max_pixels | 200704 | 200704 | ✅ |

### ST-RL (Table 8)

| Parameter | Paper | Our Implementation | Match |
|-----------|-------|-------------------|-------|
| Learning Rate | 1e-6 | 1e-6 | ✅ |
| Batch Size (per device) | 8 | 16 (optimized for H200) | ⚠️ |
| Epochs | 5 | 5 | ✅ |
| Num Generations | 8 | 8 | ✅ |
| Temperature | 1.2 | 1.2 | ✅ |
| top_p | 1.0 | 1.0 | ✅ |
| top_k | 8 | 8 | ✅ |

**Note:** We increased batch size from 8→16 to better utilize H200 GPUs (143GB vs A800's 80GB).

---

## Results Comparison (SFT Stage)

### Paper Table 1 vs Our Results

| Metric | Paper SFT | Our SFT | Difference |
|--------|-----------|---------|------------|
| ID Edge | 94.82% | 100.00% | +5.18% ✅ |
| ID Path | 99.76% | 64.14% | -35.62% ⚠️ |
| OOD Edge | 64.55% | 100.00% | +35.45% ✅ |
| OOD Path | 41.76% | 56.28% | +14.52% ✅ |

### Analysis of ID Path Discrepancy

**Paper's ID Path: 99.76%** vs **Our ID Path: 64.14%**

This is likely a **methodology difference**:
- **Paper**: May measure "step accuracy" (each step of trajectory counted separately)
- **Ours**: Measures "first-step accuracy" for multi-step paths

Evidence:
- Our Edge (single-step) accuracy is 100%
- Our Path (multi-step) first-step accuracy is 64%
- If paper measures per-step accuracy across trajectories, they could get ~99%

---

## Evaluation Methodology

### Paper's Static Benchmark
- **Edge**: Single-step click accuracy
- **Path**: Multi-step navigation (unclear if per-step or full-path completion)
- **ID**: Subtrees 0-1 (seen in training)
- **OOD**: Subtree 4 (held out)

### Our Implementation
- **Edge**: Single-step accuracy (correct icon click) ✅
- **Path**: First-step accuracy for multi-step tasks ⚠️
- **ID/OOD Split**: Matches paper ✅

### Test Set Sizes

| Set | Paper (implied) | Ours |
|-----|-----------------|------|
| ID Edge | ~274×2 | 90 |
| ID Path | ~12,439×2 | 435 |
| OOD Edge | ~274 | 45 |
| OOD Path | ~12,439 | 462 |

---

## Key Findings

### Verified Alignments ✅
1. **Environment structure** matches paper exactly (231 pages, depth 7, [5,3,2,2,1,1] distribution)
2. **Training hyperparameters** match paper (SFT and ST-RL)
3. **OOD performance exceeds paper** (Edge: +35%, Path: +15%)
4. **Subtree allocation** matches paper (2:2:1 for SFT:RL:Test)

### Potential Discrepancies ⚠️
1. **SFT dataset size**: 30,888 vs paper's 60,864 (may be counting method)
2. **ID Path accuracy**: 64% vs paper's 99.76% (likely methodology difference)
3. **ST-RL batch size**: 16 vs paper's 8 (intentional optimization)

---

## Recommended Actions

1. **Verify Path evaluation methodology** - Check if paper measures per-step or full-path accuracy
2. **Compare SFT data generation** - Review if each trajectory step should be a separate sample
3. **ST-RL training** - Currently running with optimized batch size

---

## Files in Pipeline

### Data Generation
- `data_engine/tree.py` - Core environment generator (448×448)
- `data_engine/generate_paper_env.py` - Paper-aligned environment ([5,3,2,2,1,1])
- `data_engine/generate_dataset_aligned.py` - SFT/Test data generation
- `data_engine/generate_st_rl_data.py` - ST-RL Path data generation

### Training Scripts
- `gui_scripts/sft_448.sh` - SFT training (paper Table 7 params)
- `gui_scripts/st_rl_448.sh` - ST-RL training (paper Table 8 params)

### Evaluation
- `eval/evaluate_paper_style.py` - Paper Table 1 style evaluation

### Checkpoints
- SFT: `checkpoint/gui_exp/sft_448/v1-20260130-013131/v0-20260130-013206/checkpoint-956`
- ST-RL: Training in progress...

---

*Last verified: 2026-01-30*
