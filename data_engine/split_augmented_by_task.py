"""Split amex-augmented-sft train.json into per-source subsets."""
import argparse
import json
import os

BUCKETS = {
    "amex_augmented_nav": "train_nav.json",
    "amex_augmented_grounding": "train_grounding.json",
    "amex_augmented_understanding": "train_understanding.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Full train.json")
    ap.add_argument("--output-dir", required=True, help="Where to write per-task JSONs")
    ap.add_argument("--force", action="store_true", help="Overwrite existing splits")
    args = ap.parse_args()

    data = json.load(open(args.input))
    split = {k: [] for k in BUCKETS}
    other = 0
    for s in data:
        src = s.get("source")
        if src in split:
            split[src].append(s)
        else:
            other += 1

    print(f"Total: {len(data)}")
    for k, v in split.items():
        print(f"  {k}: {len(v)}")
    if other:
        print(f"  <other>: {other}")

    for src, fname in BUCKETS.items():
        out = os.path.join(args.output_dir, fname)
        if os.path.exists(out) and not args.force:
            print(f"skip existing: {out}")
            continue
        with open(out, "w") as f:
            json.dump(split[src], f)
        print(f"wrote {out}: {len(split[src])} samples")


if __name__ == "__main__":
    main()
