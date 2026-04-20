# Augmentation validation — results (Path B)

**Protocol**: Every run is a one-stage gentle fine-tune from `Qwen/Qwen2.5-VL-7B-Instruct`, LR=1e-6, 1 epoch, max_pixels=1,003,520, max_length=5120, eff_batch=256, zero3+grad_ckpt+flash_attn. Identical compute across runs; **only the training data changes**.

**Eval harness**: `eval/evaluate_real_world.py --base_model` on 6 grounding benchmarks (ScreenSpot / ScreenSpot-v2 / MoTIF / Refexp / VWB-AG / VWB-EG). Resolution = native; prompt mode = Qwen-native bbox format (required because our LR=1e-6 preserves native grounding).

## Reference points (no retrain)

| Model | Source | Avg |
|---|---|---:|
| `Qwen/Qwen2.5-VL-7B-Instruct` | zero-shot base | 76.4 |
| `namhokaist/Qwen2.5-VL-7B-AmexGeLab-SFT-v3` | 24k non-aug full-SFT (released) | 64.0 |
| `namhokaist/Qwen2.5-VL-7B-ContinueTrain-v2` | **21k real-world**, LR=1e-6 — reference | **77.7** |

## Leaderboard

| Run | Samples | ScreenSpot | SS-v2 | FuncPred | MoTIF | Refexp | VWB-AG | VWB-EG | **Avg** | HF model |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
<!-- LEADERBOARD:BEGIN -->
| **t1a_aug_21k** (21,337) | 21337 | 48.19 | 48.66 | — | 52.25 | 30.27 | 63.11 | 53.03 | **49.25** | [namhokaist/qwen25vl-7b-augexp-t1a_aug_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-t1a_aug_21k) |
| **t1b_orig_21k** (21,337) | 21337 | 49.61 | 50.94 | — | 53.99 | 31.33 | 62.14 | 55.93 | **50.66** | [namhokaist/qwen25vl-7b-augexp-t1b_orig_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-t1b_orig_21k) |
| **t1c_mix_21k** (21,337) | 21337 | 47.48 | 48.11 | — | 52.81 | 29.73 | 63.11 | 51.33 | **48.76** | [namhokaist/qwen25vl-7b-augexp-t1c_mix_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-t1c_mix_21k) |
| **t2a_aug_nav_21k** (21,000) | 21000 | 57.70 | 58.88 | — | 61.50 | 40.00 | 65.05 | 63.92 | **57.84** | [namhokaist/qwen25vl-7b-augexp-t2a_aug_nav_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-t2a_aug_nav_21k) |
| **t2b_orig_nav_21k** (21,000) | 21000 | 61.79 | 62.42 | — | 67.35 | 46.73 | 65.05 | 67.55 | **61.82** | [namhokaist/qwen25vl-7b-augexp-t2b_orig_nav_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-t2b_orig_nav_21k) |
| **lora_mix_21k_lr5e5** (21,337) | 21337 | 39.78 | 38.60 | — | 50.83 | 25.31 | 55.34 | 38.26 | **41.35** | [namhokaist/qwen25vl-7b-augexp-lora_mix_21k_lr5e5](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-lora_mix_21k_lr5e5) |
| **lora_mix_21k_lr1e4** (21,337) | 21337 | 1.73 | 1.97 | — | 6.48 | 1.59 | 1.94 | 1.21 | **2.49** | [namhokaist/qwen25vl-7b-augexp-lora_mix_21k_lr1e4](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-lora_mix_21k_lr1e4) |
| **lora_mix_80k_lr5e5** (80,000) | 80000 | 1.42 | 1.97 | — | 11.78 | 1.95 | 7.77 | 2.42 | **4.55** | [namhokaist/qwen25vl-7b-augexp-lora_mix_80k_lr5e5](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-lora_mix_80k_lr5e5) |
| **lora_mix_80k_lr1e4** (80,000) | 80000 | 0.55 | 0.63 | — | 4.19 | 0.53 | 1.94 | 0.24 | **1.35** | [namhokaist/qwen25vl-7b-augexp-lora_mix_80k_lr1e4](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-lora_mix_80k_lr1e4) |
| **r0_real_only_21k** (21,337) | 21337 | 62.97 | 64.31 | — | 69.57 | 54.87 | 64.08 | 63.92 | **63.29** | [namhokaist/qwen25vl-7b-augexp-r0_real_only_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r0_real_only_21k) |
| **r1_real_aug_8020_21k** (21,337) | 21337 | 64.15 | 66.04 | — | 71.62 | 56.64 | 63.11 | 66.10 | **64.61** | [namhokaist/qwen25vl-7b-augexp-r1_real_aug_8020_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r1_real_aug_8020_21k) |
| **r2_real_aug_5050_21k** (21,337) | 21337 | 50.31 | 51.10 | — | 58.74 | 34.69 | 60.19 | 54.48 | **51.59** | [namhokaist/qwen25vl-7b-augexp-r2_real_aug_5050_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r2_real_aug_5050_21k) |
| **r_real_aug_9010_21k** (21,337) | 21337 | 61.08 | 62.97 | — | 67.43 | 50.97 | 63.11 | 61.26 | **61.14** | [namhokaist/qwen25vl-7b-augexp-r_real_aug_9010_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r_real_aug_9010_21k) |
| **r_real_aug_8515_21k** (21,337) | 21337 | 58.33 | 59.51 | — | 66.01 | 46.19 | 61.17 | 60.05 | **58.54** | [namhokaist/qwen25vl-7b-augexp-r_real_aug_8515_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r_real_aug_8515_21k) |
| **r_real_aug_7030_21k** (21,337) | 21337 | 54.87 | 55.19 | — | 62.21 | 42.30 | 63.11 | 57.63 | **55.89** | [namhokaist/qwen25vl-7b-augexp-r_real_aug_7030_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r_real_aug_7030_21k) |
| **s1_success_nav_orig** (23,849) | 23849 | 59.28 | 61.08 | — | 65.93 | 44.25 | 62.14 | 65.38 | **59.67** | [namhokaist/qwen25vl-7b-augexp-s1_success_nav_orig](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-s1_success_nav_orig) |
| **s3_success_nav_aug_21k** (21,000) | 21000 | 58.33 | 59.04 | — | 61.82 | 40.71 | 66.02 | 64.16 | **58.35** | [namhokaist/qwen25vl-7b-augexp-s3_success_nav_aug_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-s3_success_nav_aug_21k) |
| **s2_success_nav_aug** (38,533) | 38533 | 41.27 | 41.82 | — | 41.42 | 20.53 | 63.11 | 42.86 | **41.84** | [namhokaist/qwen25vl-7b-augexp-s2_success_nav_aug](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-s2_success_nav_aug) |
| **c1_combined_nav_21k** (21,000) | 21000 | 58.49 | 59.36 | — | 64.51 | 43.01 | 66.02 | 64.41 | **59.30** | [namhokaist/qwen25vl-7b-augexp-c1_combined_nav_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-c1_combined_nav_21k) |
<!-- LEADERBOARD:END -->

## Per-run details

### `t1a_aug_21k` — Aug 21k — stratified 25/37.5/37.5 nav/G/U (Path B)

- Recorded: 2026-04-18 12:19 UTC
- Samples: 21,337
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_t1a_aug_21k/v0-20260418_104512/checkpoint-83`
- HF model: [namhokaist/qwen25vl-7b-augexp-t1a_aug_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-t1a_aug_21k)
- Notes: Path B: 1-stage gentle FT from base Qwen (LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256)

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 48.19 | 613 / 1272 |
| ScreenSpot-v2 | 48.66 | 619 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 52.25 | 661 / 1265 |
| Refexp | 30.27 | 171 / 565 |
| VWB-AG | 63.11 | 65 / 103 |
| VWB-EG | 53.03 | 219 / 413 |
| **Average** | **49.25** | |


### `t1b_orig_21k` — Orig 21k — same ratio, non-augmented AMEX (Path B)

- Recorded: 2026-04-18 13:53 UTC
- Samples: 21,337
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_t1b_orig_21k/v0-20260418_121938/checkpoint-83`
- HF model: [namhokaist/qwen25vl-7b-augexp-t1b_orig_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-t1b_orig_21k)
- Notes: Path B: 1-stage gentle FT from base Qwen (LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256)

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 49.61 | 631 / 1272 |
| ScreenSpot-v2 | 50.94 | 648 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 53.99 | 683 / 1265 |
| Refexp | 31.33 | 177 / 565 |
| VWB-AG | 62.14 | 64 / 103 |
| VWB-EG | 55.93 | 231 / 413 |
| **Average** | **50.66** | |


### `t1c_mix_21k` — Mix 21k — 50-50 aug+orig per task bucket (Path B)

- Recorded: 2026-04-18 15:28 UTC
- Samples: 21,337
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_t1c_mix_21k/v0-20260418_135347/checkpoint-83`
- HF model: [namhokaist/qwen25vl-7b-augexp-t1c_mix_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-t1c_mix_21k)
- Notes: Path B: 1-stage gentle FT from base Qwen (LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256)

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 47.48 | 604 / 1272 |
| ScreenSpot-v2 | 48.11 | 612 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 52.81 | 668 / 1265 |
| Refexp | 29.73 | 168 / 565 |
| VWB-AG | 63.11 | 65 / 103 |
| VWB-EG | 51.33 | 212 / 413 |
| **Average** | **48.76** | |


### `t2a_aug_nav_21k` — Aug-nav 21k — nav-only from augmented pool (Path B)

- Recorded: 2026-04-18 17:01 UTC
- Samples: 21,000
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_t2a_aug_nav_21k/v0-20260418_152852/checkpoint-82`
- HF model: [namhokaist/qwen25vl-7b-augexp-t2a_aug_nav_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-t2a_aug_nav_21k)
- Notes: Path B: 1-stage gentle FT from base Qwen (LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256)

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 57.70 | 734 / 1272 |
| ScreenSpot-v2 | 58.88 | 749 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 61.50 | 778 / 1265 |
| Refexp | 40.00 | 226 / 565 |
| VWB-AG | 65.05 | 67 / 103 |
| VWB-EG | 63.92 | 264 / 413 |
| **Average** | **57.84** | |


### `t2b_orig_nav_21k` — Orig-nav 21k — nav-only from original pool (Path B)

- Recorded: 2026-04-18 18:37 UTC
- Samples: 21,000
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_t2b_orig_nav_21k/v0-20260418_170150/checkpoint-82`
- HF model: [namhokaist/qwen25vl-7b-augexp-t2b_orig_nav_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-t2b_orig_nav_21k)
- Notes: Path B: 1-stage gentle FT from base Qwen (LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256)

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 61.79 | 786 / 1272 |
| ScreenSpot-v2 | 62.42 | 794 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 67.35 | 852 / 1265 |
| Refexp | 46.73 | 264 / 565 |
| VWB-AG | 65.05 | 67 / 103 |
| VWB-EG | 67.55 | 279 / 413 |
| **Average** | **61.82** | |


### `lora_mix_21k_lr5e5` — LoRA mix 21k, LR=5e-5 (rank=16, alpha=32)

- Recorded: 2026-04-19 02:38 UTC
- Samples: 21,337
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_lora_mix_21k_lr5e5/v0-20260419_012018/checkpoint-83`
- HF model: [namhokaist/qwen25vl-7b-augexp-lora_mix_21k_lr5e5](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-lora_mix_21k_lr5e5)
- Notes: Path B: lora FT, LR=5e-5, 1 ep, max_pixels=1M, eff_batch=256, rank=16 alpha=32

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 39.78 | 506 / 1272 |
| ScreenSpot-v2 | 38.60 | 491 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 50.83 | 643 / 1265 |
| Refexp | 25.31 | 143 / 565 |
| VWB-AG | 55.34 | 57 / 103 |
| VWB-EG | 38.26 | 158 / 413 |
| **Average** | **41.35** | |


### `lora_mix_21k_lr1e4` — LoRA mix 21k, LR=1e-4 (rank=16, alpha=32)

- Recorded: 2026-04-19 03:54 UTC
- Samples: 21,337
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_lora_mix_21k_lr1e4/v0-20260419_023802/checkpoint-83`
- HF model: [namhokaist/qwen25vl-7b-augexp-lora_mix_21k_lr1e4](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-lora_mix_21k_lr1e4)
- Notes: Path B: lora FT, LR=1e-4, 1 ep, max_pixels=1M, eff_batch=256, rank=16 alpha=32

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 1.73 | 22 / 1272 |
| ScreenSpot-v2 | 1.97 | 25 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 6.48 | 82 / 1265 |
| Refexp | 1.59 | 9 / 565 |
| VWB-AG | 1.94 | 2 / 103 |
| VWB-EG | 1.21 | 5 / 413 |
| **Average** | **2.49** | |


### `lora_mix_80k_lr5e5` — LoRA mix 80k, LR=5e-5 (rank=16, alpha=32)

- Recorded: 2026-04-19 06:40 UTC
- Samples: 80,000
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_lora_mix_80k_lr5e5/v0-20260419_035459/checkpoint-310`
- HF model: [namhokaist/qwen25vl-7b-augexp-lora_mix_80k_lr5e5](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-lora_mix_80k_lr5e5)
- Notes: Path B: lora FT, LR=5e-5, 1 ep, max_pixels=1M, eff_batch=256, rank=16 alpha=32

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 1.42 | 18 / 1272 |
| ScreenSpot-v2 | 1.97 | 25 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 11.78 | 149 / 1265 |
| Refexp | 1.95 | 11 / 565 |
| VWB-AG | 7.77 | 8 / 103 |
| VWB-EG | 2.42 | 10 / 413 |
| **Average** | **4.55** | |


### `lora_mix_80k_lr1e4` — LoRA mix 80k, LR=1e-4 (rank=16, alpha=32)

- Recorded: 2026-04-19 09:27 UTC
- Samples: 80,000
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_lora_mix_80k_lr1e4/v0-20260419_064100/checkpoint-310`
- HF model: [namhokaist/qwen25vl-7b-augexp-lora_mix_80k_lr1e4](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-lora_mix_80k_lr1e4)
- Notes: Path B: lora FT, LR=1e-4, 1 ep, max_pixels=1M, eff_batch=256, rank=16 alpha=32

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 0.55 | 7 / 1272 |
| ScreenSpot-v2 | 0.63 | 8 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 4.19 | 53 / 1265 |
| Refexp | 0.53 | 3 / 565 |
| VWB-AG | 1.94 | 2 / 103 |
| VWB-EG | 0.24 | 1 / 413 |
| **Average** | **1.35** | |


### `r0_real_only_21k` — R0: 21k real-world (reproduces ContinueTrain-v2 recipe)

- Recorded: 2026-04-19 11:01 UTC
- Samples: 21,337
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_r0_real_only_21k/v0-20260419_092745/checkpoint-83`
- HF model: [namhokaist/qwen25vl-7b-augexp-r0_real_only_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r0_real_only_21k)
- Notes: Path B: full FT, LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 62.97 | 801 / 1272 |
| ScreenSpot-v2 | 64.31 | 818 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 69.57 | 880 / 1265 |
| Refexp | 54.87 | 310 / 565 |
| VWB-AG | 64.08 | 66 / 103 |
| VWB-EG | 63.92 | 264 / 413 |
| **Average** | **63.29** | |


### `r1_real_aug_8020_21k` — R1: 80% real + 20% aug (additive augmentation test)

- Recorded: 2026-04-19 12:35 UTC
- Samples: 21,337
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_r1_real_aug_8020_21k/v0-20260419_110115/checkpoint-83`
- HF model: [namhokaist/qwen25vl-7b-augexp-r1_real_aug_8020_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r1_real_aug_8020_21k)
- Notes: Path B: full FT, LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 64.15 | 816 / 1272 |
| ScreenSpot-v2 | 66.04 | 840 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 71.62 | 906 / 1265 |
| Refexp | 56.64 | 320 / 565 |
| VWB-AG | 63.11 | 65 / 103 |
| VWB-EG | 66.10 | 273 / 413 |
| **Average** | **64.61** | |


### `r2_real_aug_5050_21k` — R2: 50% real + 50% aug (heavy augmentation test)

- Recorded: 2026-04-19 14:28 UTC
- Samples: 21,337
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_r2_real_aug_5050_21k/v0-20260419_125514/checkpoint-83`
- HF model: [namhokaist/qwen25vl-7b-augexp-r2_real_aug_5050_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r2_real_aug_5050_21k)
- Notes: Path B: full FT, LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 50.31 | 640 / 1272 |
| ScreenSpot-v2 | 51.10 | 650 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 58.74 | 743 / 1265 |
| Refexp | 34.69 | 196 / 565 |
| VWB-AG | 60.19 | 62 / 103 |
| VWB-EG | 54.48 | 225 / 413 |
| **Average** | **51.59** | |


### `r_real_aug_9010_21k` — R3: 90% real + 10% aug (narrow sweet-spot test)

- Recorded: 2026-04-19 23:41 UTC
- Samples: 21,337
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_r_real_aug_9010_21k/v0-20260419_220708/checkpoint-83`
- HF model: [namhokaist/qwen25vl-7b-augexp-r_real_aug_9010_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r_real_aug_9010_21k)
- Notes: Path B: full FT, LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 61.08 | 777 / 1272 |
| ScreenSpot-v2 | 62.97 | 801 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 67.43 | 853 / 1265 |
| Refexp | 50.97 | 288 / 565 |
| VWB-AG | 63.11 | 65 / 103 |
| VWB-EG | 61.26 | 253 / 413 |
| **Average** | **61.14** | |


### `r_real_aug_8515_21k` — R4: 85% real + 15% aug (narrow sweet-spot test)

- Recorded: 2026-04-20 01:15 UTC
- Samples: 21,337
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_r_real_aug_8515_21k/v0-20260419_234113/checkpoint-83`
- HF model: [namhokaist/qwen25vl-7b-augexp-r_real_aug_8515_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r_real_aug_8515_21k)
- Notes: Path B: full FT, LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 58.33 | 742 / 1272 |
| ScreenSpot-v2 | 59.51 | 757 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 66.01 | 835 / 1265 |
| Refexp | 46.19 | 261 / 565 |
| VWB-AG | 61.17 | 63 / 103 |
| VWB-EG | 60.05 | 248 / 413 |
| **Average** | **58.54** | |


### `r_real_aug_7030_21k` — R5: 70% real + 30% aug (mid-ratio test)

- Recorded: 2026-04-20 02:49 UTC
- Samples: 21,337
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_r_real_aug_7030_21k/v0-20260420_011555/checkpoint-83`
- HF model: [namhokaist/qwen25vl-7b-augexp-r_real_aug_7030_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-r_real_aug_7030_21k)
- Notes: Path B: full FT, LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 54.87 | 698 / 1272 |
| ScreenSpot-v2 | 55.19 | 702 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 62.21 | 787 / 1265 |
| Refexp | 42.30 | 239 / 565 |
| VWB-AG | 63.11 | 65 / 103 |
| VWB-EG | 57.63 | 238 / 413 |
| **Average** | **55.89** | |


### `s1_success_nav_orig` — S1: success-only nav, orig (23.8k, non-augmented)

- Recorded: 2026-04-20 05:57 UTC
- Samples: 23,849
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_s1_success_nav_orig/v0-20260420_041745/checkpoint-93`
- HF model: [namhokaist/qwen25vl-7b-augexp-s1_success_nav_orig](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-s1_success_nav_orig)
- Notes: Path B: full FT, LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 59.28 | 754 / 1272 |
| ScreenSpot-v2 | 61.08 | 777 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 65.93 | 834 / 1265 |
| Refexp | 44.25 | 250 / 565 |
| VWB-AG | 62.14 | 64 / 103 |
| VWB-EG | 65.38 | 270 / 413 |
| **Average** | **59.67** | |


### `s3_success_nav_aug_21k` — S3: success-only nav, aug subsampled to 21k (matches T2.A budget)

- Recorded: 2026-04-20 07:32 UTC
- Samples: 21,000
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_s3_success_nav_aug_21k/v0-20260420_055731/checkpoint-82`
- HF model: [namhokaist/qwen25vl-7b-augexp-s3_success_nav_aug_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-s3_success_nav_aug_21k)
- Notes: Path B: full FT, LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 58.33 | 742 / 1272 |
| ScreenSpot-v2 | 59.04 | 751 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 61.82 | 782 / 1265 |
| Refexp | 40.71 | 230 / 565 |
| VWB-AG | 66.02 | 68 / 103 |
| VWB-EG | 64.16 | 265 / 413 |
| **Average** | **58.35** | |


### `s2_success_nav_aug` — S2: success-only nav, aug (38.5k, full augmented set)

- Recorded: 2026-04-20 09:46 UTC
- Samples: 38,533
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_s2_success_nav_aug/v0-20260420_073227/checkpoint-150`
- HF model: [namhokaist/qwen25vl-7b-augexp-s2_success_nav_aug](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-s2_success_nav_aug)
- Notes: Path B: full FT, LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 41.27 | 525 / 1272 |
| ScreenSpot-v2 | 41.82 | 532 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 41.42 | 524 / 1265 |
| Refexp | 20.53 | 116 / 565 |
| VWB-AG | 63.11 | 65 / 103 |
| VWB-EG | 42.86 | 177 / 413 |
| **Average** | **41.84** | |


### `c1_combined_nav_21k` — C1: combined nav 21k (10.5k orig + 10.5k aug, all trajectories)

- Recorded: 2026-04-20 11:45 UTC
- Samples: 21,000
- Checkpoint: `/workspace/gelab-env/checkpoint/gui_exp/aug_c1_combined_nav_21k/v0-20260420_101119/checkpoint-82`
- HF model: [namhokaist/qwen25vl-7b-augexp-c1_combined_nav_21k](https://huggingface.co/namhokaist/qwen25vl-7b-augexp-c1_combined_nav_21k)
- Notes: Path B: full FT, LR=1e-6, 1 ep, max_pixels=1M, eff_batch=256

| Benchmark | Accuracy | Correct / Total |
|---|---:|---:|
| ScreenSpot | 58.49 | 744 / 1272 |
| ScreenSpot-v2 | 59.36 | 755 / 1272 |
| FuncPred | — | ? / ? |
| MoTIF | 64.51 | 816 / 1265 |
| Refexp | 43.01 | 243 / 565 |
| VWB-AG | 66.02 | 68 / 103 |
| VWB-EG | 64.41 | 266 / 413 |
| **Average** | **59.30** | |

