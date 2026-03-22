"""
Merge multiple SFT JSON files from different team members into a single training file.

Deduplicates by source trajectory ID, validates image paths exist,
and re-indexes all samples.

Usage:
    python data_engine/merge_amex_sft.py \
        --inputs path/to/sft1.json path/to/sft2.json path/to/sft3.json \
        --output datas_amex/sft_amex_combined.json
"""

import argparse
import json
import os
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Merge multiple AMEX SFT JSON files")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths to SFT JSON files to merge")
    parser.add_argument("--output", default="datas_amex/sft_amex_combined.json",
                        help="Output path for merged SFT JSON")
    parser.add_argument("--skip_missing_images", action="store_true",
                        help="Skip samples with missing image files instead of erroring")
    args = parser.parse_args()

    all_samples = []
    seen_sources = set()
    stats_per_file = []

    for input_path in args.inputs:
        if not os.path.exists(input_path):
            print(f"WARNING: {input_path} not found, skipping")
            continue

        with open(input_path, "r") as f:
            samples = json.load(f)

        new_samples = []
        duplicates = 0
        for s in samples:
            source = s.get("source", "")
            sample_key = f"{source}_{s.get('action_type', '')}_{s.get('idx', '')}"
            if source and sample_key in seen_sources:
                duplicates += 1
                continue
            seen_sources.add(sample_key)
            new_samples.append(s)

        all_samples.extend(new_samples)
        stats_per_file.append({
            "file": input_path,
            "total": len(samples),
            "new": len(new_samples),
            "duplicates": duplicates,
        })
        print(f"  {input_path}: {len(samples)} total, {len(new_samples)} new, {duplicates} duplicates")

    missing_images = 0
    valid_samples = []
    for s in all_samples:
        images_ok = True
        for img_path in s.get("images", []):
            if not os.path.exists(img_path):
                missing_images += 1
                images_ok = False
                break
        if images_ok or not args.skip_missing_images:
            valid_samples.append(s)

    for i, s in enumerate(valid_samples):
        s["idx"] = i

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(valid_samples, f, indent=2)

    types = Counter(s.get("action_type", "?") for s in valid_samples)
    sources = set(s.get("source", "") for s in valid_samples)

    print(f"\n=== MERGED SFT DATASET ===")
    print(f"Input files:      {len(args.inputs)}")
    print(f"Total samples:    {len(valid_samples)}")
    print(f"Unique sources:   {len(sources)}")
    print(f"Missing images:   {missing_images}")
    print(f"Action types:")
    for t, c in types.most_common():
        print(f"  {t:20s}: {c}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
