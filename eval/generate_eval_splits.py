"""
Generate evaluation splits matching paper methodology.

Uses the same data generation logic as SFT training (data_engine/generate_sft_data.py)
to ensure consistency.

- ID Edge: all edges from subtrees 0-1 (same format as SFT training data) → 274 × 2 = 548
- ID Path: all paths from subtrees 0-1 (same format as SFT training data) → 12,439 × 2 = 24,878
- OOD Edge: all edges from subtree 4 → 274
- OOD Path: all paths from subtree 4 → 12,439

Paper references:
- "each subtree contains 12,439 path data instances and 274 edge data instances"
- "two are designated for SFT training, two for RL training, and one for OOD testing"
"""

import json
import os
import sys
import random

# Add data_engine to path for importing SFTDataGenerator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_engine'))
from generate_sft_data import SFTDataGenerator


def add_eval_fields(samples):
    """Add source_page and target_page fields expected by evaluate.py."""
    for sample in samples:
        msgs = sample.get("messages", [])
        if len(msgs) < 1:
            continue
        user_content = msgs[0]["content"]
        # Extract source and target from "Instruction: from X to Y"
        import re
        match = re.search(r'from (page_\d+) to (page_\d+)', user_content)
        if match:
            sample["source_page"] = match.group(1)
            sample["target_page"] = match.group(2)
        # Map 'path' field to 'path_length' for evaluate.py compatibility
        if "path" in sample and "path_length" not in sample:
            sample["path_length"] = sample["path"]
    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation splits (full dataset)")
    parser.add_argument("--env_dir", default="datas", help="Path to environment directory")
    parser.add_argument("--output_dir", default="datas", help="Output directory")
    args = parser.parse_args()

    if not os.path.isabs(args.env_dir):
        args.env_dir = os.path.join(os.getcwd(), args.env_dir)

    random.seed(42)

    print("=" * 60)
    print("EVALUATION SPLITS GENERATION (Full Dataset)")
    print("=" * 60)
    print(f"Environment: {args.env_dir}")
    print(f"Output: {args.output_dir}")
    print()

    generator = SFTDataGenerator(args.env_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- ID Edge: subtrees 0-1 (same as SFT training edge data) ---
    print("ID Edge (subtrees 0-1)...")
    id_edge = generator.generate_edge_samples(subtrees=[0, 1])
    id_edge = add_eval_fields(id_edge)
    for i, s in enumerate(id_edge):
        s["idx"] = i
    with open(os.path.join(args.output_dir, "test_id_edge.json"), "w") as f:
        json.dump(id_edge, f, indent=2)
    print(f"  Generated: {len(id_edge)} samples")

    # --- OOD Edge: subtree 4 ---
    print("OOD Edge (subtree 4)...")
    ood_edge = generator.generate_edge_samples(subtrees=[4])
    ood_edge = add_eval_fields(ood_edge)
    for i, s in enumerate(ood_edge):
        s["idx"] = i
    with open(os.path.join(args.output_dir, "test_ood_edge.json"), "w") as f:
        json.dump(ood_edge, f, indent=2)
    print(f"  Generated: {len(ood_edge)} samples")

    # --- ID Path: subtrees 0-1 (same as SFT training path data) ---
    print("ID Path (subtrees 0-1)...")
    id_path = generator.generate_path_samples(subtrees=[0, 1])
    id_path = add_eval_fields(id_path)
    for i, s in enumerate(id_path):
        s["idx"] = i
    with open(os.path.join(args.output_dir, "test_id_path.json"), "w") as f:
        json.dump(id_path, f, indent=2)
    print(f"  Generated: {len(id_path)} samples")

    # --- OOD Path: subtree 4 (equivalent data for test subtree) ---
    print("OOD Path (subtree 4)...")
    ood_path = generator.generate_path_samples(subtrees=[4])
    ood_path = add_eval_fields(ood_path)
    for i, s in enumerate(ood_path):
        s["idx"] = i
    with open(os.path.join(args.output_dir, "test_ood_path.json"), "w") as f:
        json.dump(ood_path, f, indent=2)
    print(f"  Generated: {len(ood_path)} samples")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  ID Edge:   {len(id_edge):>6} samples  (expected: 548)")
    print(f"  OOD Edge:  {len(ood_edge):>6} samples  (expected: 274)")
    print(f"  ID Path:   {len(id_path):>6} samples  (expected: 24,878)")
    print(f"  OOD Path:  {len(ood_path):>6} samples  (expected: 12,439)")
    print()
    print(f"Files created in {args.output_dir}:")
    print("  - test_id_edge.json")
    print("  - test_ood_edge.json")
    print("  - test_id_path.json")
    print("  - test_ood_path.json")


if __name__ == "__main__":
    main()
