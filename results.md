# GeLab SFT Evaluation Results

**Date**: 2026-01-27  
**Model**: Qwen2.5-VL-7B-Instruct (SFT fine-tuned)  

---

## Current Results vs Paper

### Interactive Evaluation (Pass@1)

| Path Length | Paper SFT | Our SFT | Gap |
|-------------|-----------|---------|-----|
| Path@1 | **99.71%** | 54.74% | -44.97% |
| Path@2 | 51.16% | 12.24% | -38.92% |
| Path@3 | 19.55% | 3.60% | -15.95% |
| Path@4 | 8.52% | 0.62% | -7.90% |
| Path@5-7 | 0.31-3.13% | 0.00% | - |
| **Overall** | ~15% | **4.76%** | -10% |

### Static Evaluation (OOD)

| Metric | Paper SFT | Our SFT | Gap |
|--------|-----------|---------|-----|
| Overall | 55.46% | **53.51%** | -1.95% |
| Path | 41.76% | **38.52%** | -3.24% |
| Edge | 64.55% | **63.28%** | -1.27% |

**Static eval is close (~2% gap). Interactive eval has major gap (~45% on Path@1).**

---

## Root Cause Analysis

### Issue 1: Prompt Format Mismatch (FIXED)

| | Training | Evaluation (old) | Evaluation (fixed) |
|--|----------|------------------|-------------------|
| First step | `History: Null` | (none) | `History: Null` |

**Status**: Fixed in `interactive_eval.py`

### Issue 2: Icon Name Format Mismatch (CRITICAL)

| Task Type | Icon Format | Example |
|-----------|-------------|---------|
| Navigation | Underscore | `Digital_electronics_98` |
| Grounding | Space | `Digital electronics 98` |
| Captioning | Space | `Digital electronics 98` |

**Impact**: Model learns `Digital electronics 98` in grounding but needs to output `Digital_electronics_98` in navigation. Cannot transfer learning between tasks.

### Issue 3: Zero Icon Overlap Between Training and Test (CRITICAL)

```
Training navigation icons: 92 unique
Test subtree (4) icons: 45 unique
Overlap: 0 icons (0%)
```

The model has NEVER seen test subtree icons during navigation training.

### Issue 4: Missing Edge Data from Test Subtree

**Paper states**: "SFT training dataset incorporates Edge data from all subtrees, including Test subtree"

**Our data**:
- Edge samples: 5,268 total
- Subtree 4 pages in edge: 0/46 (0%)

This violates the paper's data requirements.

### Issue 5: Content Icon Prediction Accuracy

| Icon Type | Correct | Wrong | Accuracy |
|-----------|---------|-------|----------|
| System (home/back) | 74 | 17 | **81%** |
| Content | 1 | 45 | **2%** |

Model defaults to `home`/`back` when it doesn't recognize content icons.

---

## Data Analysis

### What We Have

| File | Samples | Source |
|------|---------|--------|
| `sft.json` | 30,066 | Regenerated from demo UI |
| `icon_grounding.json` | 2,320 | Generated |
| `icon_captioning.json` | 2,320 | Generated |
| `eval_sub4_test.json` | 12,439 | Regenerated from demo UI |

### Format Comparison

```
NAVIGATION:
Input:  "Instruction: from page_X to page_Y. History: Null"
Output: "Explain:click Digital_electronics_98 icon on page_18. Action: click(start_box='<|box_start|>(486,531)<|box_end|>')"

GROUNDING:
Input:  "Click on Digital electronics 98 in the image."
Output: "action: CLICK point:(486, 351)"

CAPTIONING:
Input:  "What is the icon at point (486, 531) in the image?"
Output: "Digital electronics 98"
```

**Problems**:
1. Icon names: underscore vs space
2. Coordinate format: `<|box_start|>` vs `point:`
3. Action format: `click(start_box=...)` vs `action: CLICK`

---

## Path to 10% Discrepancy

### Priority 1: Fix Icon Name Format (HIGH IMPACT)

Regenerate `icon_grounding.json` and `icon_captioning.json` with underscore format:
- Change `Digital electronics 98` → `Digital_electronics_98`

### Priority 2: Add Edge Data from All Subtrees (MEDIUM IMPACT)

Regenerate training data to include edge samples from subtree 4:
- Home/back navigation from all 231 pages
- This gives visual exposure to test pages

### Priority 3: Re-run Evaluation with Fixes (REQUIRED)

1. "History: Null" fix already applied
2. After data regeneration, retrain
3. Re-evaluate

### Priority 4: Verify Coordinate Systems (LOW IMPACT)

Ensure grounding coordinates match navigation coordinates:
- Grounding uses image-relative coordinates
- Navigation uses absolute coordinates
- May need normalization

---

## Recommended Action Plan

### Phase 1: Data Fix (Immediate)

```bash
# 1. Regenerate icon grounding/captioning with underscore names
python scripts/generate_icon_data.py --format underscore

# 2. Add edge data from all subtrees
python scripts/generate_sft_data.py --include_all_edge

# 3. Verify format consistency
python scripts/verify_data_format.py
```

### Phase 2: Retrain (After Data Fix)

```bash
# Same training config as before
swift sft --model Qwen2.5-VL-7B-Instruct --dataset datas/sft.json ...
```

### Phase 3: Re-evaluate

```bash
# With History: Null fix
python eval/interactive_eval.py --model_path NEW_CHECKPOINT ...
```

---

## Expected Improvement

| Fix | Expected Impact |
|-----|-----------------|
| History: Null prompt | +5-10% |
| Icon name format | +15-25% |
| Edge data from all subtrees | +5-10% |
| **Total estimated** | **+25-45%** |

With fixes, we expect Path@1 to improve from 54.74% to approximately 80-90%.

---

## Summary

### Why We Differ from Paper

1. **Format inconsistency**: Model can't connect icon grounding knowledge to navigation
2. **Missing edge data**: Model hasn't seen test subtree pages in navigation context
3. **Zero icon overlap**: Test icons completely unseen during navigation training

### Key Insight

Our **static evaluation matches paper closely** (2% gap), proving the model learned navigation. The gap is specifically in **visual icon recognition** for unseen icons, which requires proper knowledge transfer from grounding/captioning tasks.

### Next Steps

1. Fix icon name format in grounding/captioning data
2. Add edge data from all subtrees
3. Retrain with consistent data
4. Re-evaluate with fixed prompt

**Confidence**: 0.8 that these fixes will reduce the gap to within 10-15% of paper results.
