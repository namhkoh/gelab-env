"""Convert the SeeClick-web grounding corpus into ms-swift SFT format.

Source (downloaded under /workspace/nhkoh/external_grounding/):
  - OS-Atlas web domain, SeeClick-web split (seeclick_web.json + extracted
    seeclick_web_imgs archive)

Prompt templates and the 0-1000 center normalization are copied verbatim from
prepare_desktop_web_grounding.py / prepare_real_grounding_data.py so the
grounding format matches the paper A.5 eval query exactly. Source bboxes are
already normalized to 0-1 [l, t, r, b], so the center is
int(1000 * (l + r) / 2) clamped to [0, 1000].

Filters (same as prepare_desktop_web_grounding.py):
  - bbox coords outside [-0.005, 1.005], or degenerate (l >= r or t >= b)
  - empty instructions, len < 2 or > 120 after whitespace collapse
  - mojibake / non-printable instructions
  - (image, instruction) pairs annotated with DIFFERENT bboxes (ambiguous
    grounding target -> all occurrences dropped); identical duplicates deduped
  - missing image files

The per-image cap bounds repetition of a screenshot; the quota is split
~80% grounding / ~20% understanding.

Example:
    python data_engine/prepare_seeclick_web_grounding.py \
        --root /workspace/nhkoh/external_grounding \
        --output datas/seeclick_web_100k.json
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

UNDERSTAND_FRAC = 0.20


def clean_instruction(raw):
    return re.sub(r"\s+", " ", str(raw)).strip()


def valid_instruction(instr):
    if len(instr) < 2 or len(instr) > 120:
        return False
    if "�" in instr:
        return False
    return instr.isprintable()


def valid_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        l, t, r, b = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return False
    for v in (l, t, r, b):
        if not (-0.005 <= v <= 1.005):
            return False
    return l < r and t < b


def center_1000(bbox):
    l, t, r, b = bbox
    cx = max(0, min(1000, int(1000 * (l + r) / 2)))
    cy = max(0, min(1000, int(1000 * (t + b) / 2)))
    return cx, cy


def iter_seeclick_web(json_path, img_dir):
    """Yield (img_path, instruction, bbox, tier) from the SeeClick-web split."""
    records = json.load(open(json_path))
    for rec in records:
        img_path = os.path.join(img_dir, rec.get("img_filename", ""))
        for el in rec.get("elements", []):
            yield img_path, el.get("instruction", ""), el.get("bbox"), 0
    del records


def build_pool(entries, per_image_cap, rng):
    """Filter, dedupe, cap per image; return list of (img, instr, bbox, tier)."""
    stats = defaultdict(int)
    # (img, instr) -> {rounded_bbox: (bbox, tier)}
    pairs = {}
    for img_path, raw_instr, bbox, tier in entries:
        stats["raw"] += 1
        instr = clean_instruction(raw_instr)
        if not valid_instruction(instr):
            stats["bad_instruction"] += 1
            continue
        if not valid_bbox(bbox):
            stats["bad_bbox"] += 1
            continue
        bbox = [float(v) for v in bbox]
        key = (img_path, instr)
        rkey = tuple(round(v, 4) for v in bbox)
        bucket = pairs.setdefault(key, {})
        if rkey in bucket:
            stats["dup_identical"] += 1
            # keep the most interactive tier seen for this exact annotation
            if tier < bucket[rkey][1]:
                bucket[rkey] = (bbox, tier)
        else:
            bucket[rkey] = (bbox, tier)

    by_image = defaultdict(list)
    for (img_path, instr), bucket in pairs.items():
        if len(bucket) > 1:
            stats["ambiguous_pair"] += 1
            continue
        (bbox, tier), = bucket.values()
        by_image[img_path].append((instr, bbox, tier))
    pairs.clear()

    pool = []
    img_missing = 0
    images = sorted(by_image)
    rng.shuffle(images)
    for img_path in images:
        if not os.path.isfile(img_path):
            img_missing += 1
            continue
        elems = by_image[img_path]
        rng.shuffle(elems)
        elems.sort(key=lambda e: e[2])  # stable: prefer interactive tiers
        for instr, bbox, tier in elems[:per_image_cap]:
            pool.append((img_path, instr, bbox, tier))
    stats["images_total"] = len(images)
    stats["images_missing"] = img_missing
    stats["pool"] = len(pool)
    return pool, stats


def make_samples(pool, quota, source, rng):
    """Order pool (tier-first, shuffled within tier), split into grounding
    and understanding samples up to quota."""
    rng.shuffle(pool)
    pool.sort(key=lambda e: e[3])
    take = pool[: min(len(pool), quota)]
    n_under = int(round(len(take) * UNDERSTAND_FRAC))
    n_ground = len(take) - n_under
    samples = []
    for i, (img_path, instr, bbox, _tier) in enumerate(take):
        cx, cy = center_1000(bbox)
        if i < n_ground:
            samples.append({
                "messages": [
                    {"role": "user", "content": f"<image>I want to click on {instr}. Please locate the target element I should interact with. (with point)"},
                    {"role": "assistant", "content": f"Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"},
                ],
                "images": [img_path],
                "source": source,
            })
        else:
            samples.append({
                "messages": [
                    {"role": "user", "content": f"<image>What is the icon at point ({cx},{cy}) in the image?"},
                    {"role": "assistant", "content": instr},
                ],
                "images": [img_path],
                "source": f"{source}_understanding",
            })
    return samples, n_ground, n_under


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/nhkoh/external_grounding")
    parser.add_argument("--output", default="datas/seeclick_web_100k.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quota", type=int, default=100000)
    parser.add_argument("--cap", type=int, default=2)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    web = os.path.join(root, "os_atlas", "web_domain")

    sources = [
        ("seeclick_web",
         lambda: iter_seeclick_web(os.path.join(web, "seeclick_web.json"),
                                   os.path.join(web, "images")),
         args.cap, args.quota),
    ]

    rng = random.Random(args.seed)
    all_samples = []
    for source, entry_fn, cap, quota in sources:
        pool, stats = build_pool(entry_fn(), cap, rng)
        samples, n_ground, n_under = make_samples(pool, quota, source, rng)
        all_samples.extend(samples)
        short = " (SHORT OF QUOTA)" if len(samples) < quota else ""
        print(f"[{source}] raw={stats['raw']} bad_instr={stats['bad_instruction']} "
              f"bad_bbox={stats['bad_bbox']} dup={stats['dup_identical']} "
              f"ambiguous={stats['ambiguous_pair']} images={stats['images_total']} "
              f"img_missing={stats['images_missing']} capped_pool={stats['pool']} "
              f"-> grounding={n_ground} understanding={n_under} "
              f"total={len(samples)}/{quota}{short}", flush=True)

    rng.shuffle(all_samples)
    for i, s in enumerate(all_samples):
        s["idx"] = i

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(all_samples, fp, ensure_ascii=False)
    print(f"total: {len(all_samples)} -> {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
