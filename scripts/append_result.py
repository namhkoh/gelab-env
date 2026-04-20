"""Append a row for a trained model to results.md based on its eval JSON.

results.md is created if absent, with a header + reference rows. Each run
appends a detailed section and updates the leaderboard table. Existing rows
and detail sections for the same run name are deduplicated.

Usage:
    python scripts/append_result.py \
        --run t1a_aug_21k \
        --eval eval_results/results_t1a_aug_21k.json \
        --checkpoint checkpoint/gui_exp/.../checkpoint-83 \
        --hf-repo namhokaist/qwen25vl-7b-augexp-t1a_aug_21k \
        --samples 21337 \
        --results-md /workspace/gelab-env/results.md
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
from typing import Optional

BENCHMARK_ORDER = [
    "ScreenSpot",
    "ScreenSpot-v2",
    "FuncPred",
    "MoTIF",
    "Refexp",
    "VWB-AG",
    "VWB-EG",
]

RUN_DESCRIPTIONS = {
    "t1a_aug_21k":       "Aug 21k — stratified 25/37.5/37.5 nav/G/U (Path B, full-FT LR=1e-6)",
    "t1b_orig_21k":      "Orig 21k — same ratio, non-augmented AMEX (Path B, full-FT LR=1e-6)",
    "t1c_mix_21k":       "Mix 21k — 50-50 aug+orig per task (Path B, full-FT LR=1e-6)",
    "t2a_aug_nav_21k":   "Aug-nav 21k — nav-only from augmented (Path B, full-FT LR=1e-6)",
    "t2b_orig_nav_21k":  "Orig-nav 21k — nav-only from original (Path B, full-FT LR=1e-6)",
    "lora_mix_21k_lr5e5": "LoRA mix 21k, LR=5e-5 (rank=16, alpha=32)",
    "lora_mix_21k_lr1e4": "LoRA mix 21k, LR=1e-4 (rank=16, alpha=32)",
    "lora_mix_80k_lr5e5": "LoRA mix 80k, LR=5e-5 (rank=16, alpha=32)",
    "lora_mix_80k_lr1e4": "LoRA mix 80k, LR=1e-4 (rank=16, alpha=32)",
    "r0_real_only_21k":        "R0: 21k real-world (reproduces ContinueTrain-v2 recipe)",
    "r1_real_aug_8020_21k":    "R1: 80% real + 20% aug (additive augmentation test)",
    "r2_real_aug_5050_21k":    "R2: 50% real + 50% aug (heavy augmentation test)",
    "r_real_aug_9010_21k":     "R3: 90% real + 10% aug (narrow sweet-spot test)",
    "r_real_aug_8515_21k":     "R4: 85% real + 15% aug (narrow sweet-spot test)",
    "r_real_aug_7030_21k":     "R5: 70% real + 30% aug (mid-ratio test)",
    "s1_success_nav_orig":     "S1: success-only nav, orig (23.8k, non-augmented)",
    "s2_success_nav_aug":      "S2: success-only nav, aug (38.5k, full augmented set)",
    "s3_success_nav_aug_21k":  "S3: success-only nav, aug subsampled to 21k (matches T2.A budget)",
    "c1_combined_nav_21k":         "C1: combined nav 21k (10.5k orig + 10.5k aug, all trajectories)",
    "c2_combined_success_nav_21k": "C2: combined success-nav 21k (10.5k orig-success + 10.5k aug-success)",
}


def pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if isinstance(x, dict):
        x = x.get("accuracy")
    if x is None:
        return "—"
    return f"{x * 100:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", required=True, help="path to results_<run>.json")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--wandb", default="")
    ap.add_argument("--results-md", default="/workspace/gelab-env/results.md")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    if not os.path.exists(args.eval):
        print(f"ERROR: eval JSON not found: {args.eval}")
        return 2

    with open(args.eval) as f:
        ev = json.load(f)
    benches = ev.get("benchmarks", {})
    avg = ev.get("average_accuracy")

    if not os.path.exists(args.results_md):
        print(f"ERROR: results.md not found at {args.results_md}")
        return 2

    with open(args.results_md) as f:
        md = f.read()

    # ---- Leaderboard row
    cells = [pct(benches.get(b)) for b in BENCHMARK_ORDER]
    avg_s = pct(avg)
    hf_link = f"[{args.hf_repo}](https://huggingface.co/{args.hf_repo})" if args.hf_repo else "—"
    desc = RUN_DESCRIPTIONS.get(args.run, args.run)
    leaderboard_row = (
        f"| **{args.run}** ({args.samples:,}) | {args.samples} "
        f"| {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} | {cells[5]} | {cells[6]} "
        f"| **{avg_s}** | {hf_link} |\n"
    )

    beg = "<!-- LEADERBOARD:BEGIN -->\n"
    end = "<!-- LEADERBOARD:END -->"
    if beg in md and end in md:
        head, rest = md.split(beg, 1)
        rows, tail = rest.split(end, 1)
        rows = "".join(
            line for line in rows.splitlines(keepends=True)
            if f"**{args.run}**" not in line
        )
        md = head + beg + rows + leaderboard_row + end + tail

    # ---- Per-run detail section: dedupe then append
    pattern = re.compile(
        r"\n### `" + re.escape(args.run) + r"` .*?(?=\n### `|\Z)",
        flags=re.DOTALL,
    )
    md = pattern.sub("", md)

    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    details = [
        f"\n### `{args.run}` — {desc}\n",
        f"- Recorded: {ts}",
        f"- Samples: {args.samples:,}" if args.samples else "- Samples: —",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- HF model: {hf_link}",
    ]
    if args.wandb:
        details.append(f"- wandb: {args.wandb}")
    if args.notes:
        details.append(f"- Notes: {args.notes}")
    details.append("")
    details.append("| Benchmark | Accuracy | Correct / Total |")
    details.append("|---|---:|---:|")
    for b in BENCHMARK_ORDER:
        cell = benches.get(b, {})
        if isinstance(cell, dict):
            acc = cell.get("accuracy")
            n = f"{cell.get('correct', '?')} / {cell.get('total', '?')}"
        else:
            acc = cell
            n = "—"
        details.append(f"| {b} | {pct(acc)} | {n} |")
    details.append(f"| **Average** | **{avg_s}** | |")
    details.append("")

    md += "\n".join(details) + "\n"

    with open(args.results_md, "w") as f:
        f.write(md)
    print(f"[results] wrote {args.results_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
