# AMEX SFT Evaluation Log

## Run 1: SFT-AMEX-FT (2026-03-23)

**Model:** `checkpoint/gui_exp/sft_amex/v1-20260323_071511/checkpoint-1537`
**Base:** Qwen2.5-VL-7B-Instruct, full fine-tune, 1 epoch
**Training data:** 37,260 AMEX SFT samples (1,217 trajectories)
**HuggingFace:** https://huggingface.co/namhokaist/sft-amex-ft

| Benchmark | Accuracy | Correct | Total |
|-----------|----------|---------|-------|
| ScreenSpot | 33.18% | 422 | 1272 |
| ScreenSpot-v2 | 33.73% | 429 | 1272 |
| FuncPred | - | - | (failed to load) |
| MoTIF | 52.09% | 659 | 1265 |
| Refexp | - | - | (not run) |
| VWB-EG | 6.05% | 25 | 413 |
| VWB-AG | 4.85% | 5 | 103 |
| Average | 25.98% | | |

**Analysis:** Significant degradation vs base model (84% -> 33% ScreenSpot). Likely causes:
1. Catastrophic forgetting from pure AMEX SFT without grounding data preservation
2. Action format mismatch (trained with `tap()`, eval expects `click()`)
3. No real-world grounding data mixed into training

## Run 2: Base Qwen2.5-VL-7B-Instruct -- bbox_2d prompt (2026-03-23)

**Model:** `Qwen/Qwen2.5-VL-7B-Instruct` (no fine-tuning)
**Eval script:** `eval/evaluate_real_world.py --base_model` (bbox_2d prompt, no system prompt, full resolution)

| Benchmark | Paper | Ours | Correct | Total |
|-----------|-------|------|---------|-------|
| ScreenSpot | 84.01% | 74.92% | 953 | 1272 |
| ScreenSpot-v2 | 80.34% | 77.44% | 985 | 1272 |
| MoTIF | 71.93% | 82.29% | 1041 | 1265 |
| VWB-EG | 90.07% | 74.58% | 308 | 413 |
| VWB-AG | 72.81% | 66.02% | 68 | 103 |

**Note:** Gap due to using bbox_2d prompt instead of computer_use guided generation.

## Run 3: Base Qwen2.5-VL-7B-Instruct -- guided generation (2026-03-23)

**Model:** `Qwen/Qwen2.5-VL-7B-Instruct` (no fine-tuning)
**Eval script:** `eval/eval_screenspot.py --num_gpus 3` (computer_use tool-calling with guided generation, max_pixels=unlimited)
**Method:** Same approach as ScreenSpot-Pro eval framework

| Benchmark | Paper | Ours | Correct | Total | Wrong Format |
|-----------|-------|------|---------|-------|--------------|
| ScreenSpot | 84.01% | **85.61%** | 1089 | 1272 | 0 |

**Result:** Successfully reproduced and slightly exceeded the paper's baseline (85.61% vs 84.01%).
The guided generation approach with computer_use tool-calling is the correct eval method for Qwen2.5-VL.
