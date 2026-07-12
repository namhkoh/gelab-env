"""Convert external desktop/web grounding corpora into ms-swift SFT format.

Sources (downloaded under /workspace/nhkoh/external_grounding/):
  - OS-Atlas desktop domain (windows / linux / macos splits)
  - ShowUI-desktop and ShowUI-web

Prompt templates and the 0-1000 center normalization are copied verbatim from
prepare_real_grounding_data.py / prepare_amex_gelab_sft_v3.py so the grounding
format matches the paper A.5 eval query exactly. Source bboxes are already
normalized to 0-1 [l, t, r, b], so the center is int(1000 * (l + r) / 2)
clamped to [0, 1000].

Filters (from the corpus audit):
  - bbox coords outside [-0.005, 1.005], or degenerate (l >= r or t >= b)
  - empty instructions, len < 2 or > 120 after whitespace collapse
  - mojibake / non-printable instructions
  - (image, instruction) pairs annotated with DIFFERENT bboxes (ambiguous
    grounding target -> all occurrences dropped); identical duplicates deduped
  - missing image files

Per-image caps bound repetition of a screenshot; per-source quotas are split
~80% grounding / ~20% understanding. ShowUI-web elements are prioritized by
interactivity: Button/CheckBox/Edit/ComboBox/TabItem/MenuItem first, then
Hyperlink/ListItem, then the rest.

Example:
    python data_engine/prepare_desktop_web_grounding.py \
        --root /workspace/nhkoh/external_grounding \
        --output datas/desktop_web_grounding_150k.json
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

WEB_TIER1 = {"Button", "CheckBox", "Edit", "ComboBox", "TabItem", "MenuItem"}
WEB_TIER2 = {"Hyperlink", "ListItem"}

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


def web_tier(class_name):
    if class_name in WEB_TIER1:
        return 0
    if class_name in WEB_TIER2:
        return 1
    return 2


def iter_os_atlas(json_path, img_dir):
    """Yield (img_path, instruction, bbox, tier) from an OS-Atlas split."""
    records = json.load(open(json_path))
    for rec in records:
        img_path = os.path.join(img_dir, rec.get("img_filename", ""))
        for el in rec.get("elements", []):
            yield img_path, el.get("instruction", ""), el.get("bbox"), 0
    del records


def iter_showui(json_path, img_resolver):
    """Yield (img_path, instruction, bbox, tier) from a ShowUI metadata file."""
    records = json.load(open(json_path))
    for rec in records:
        img_path = img_resolver(rec.get("img_url", ""))
        for el in rec.get("element", []):
            tier = web_tier(el.get("class_name"))
            yield img_path, el.get("instruction", ""), el.get("bbox"), tier
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
    parser.add_argument("--output", default="datas/desktop_web_grounding_150k.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quota_windows", type=int, default=48000)
    parser.add_argument("--quota_linux", type=int, default=12000)
    parser.add_argument("--quota_macos", type=int, default=10000)
    parser.add_argument("--quota_showui_desktop", type=int, default=5000)
    parser.add_argument("--quota_showui_web", type=int, default=75000)
    parser.add_argument("--cap_windows", type=int, default=8)
    parser.add_argument("--cap_linux", type=int, default=4)
    parser.add_argument("--cap_macos", type=int, default=4)
    parser.add_argument("--cap_showui_desktop", type=int, default=4)
    parser.add_argument("--cap_showui_web", type=int, default=4)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    atlas = os.path.join(root, "os_atlas", "desktop_domain")
    sw_desk = os.path.join(root, "showui_desktop")
    sw_web = os.path.join(root, "showui_web")

    def desk_resolver(img_url):
        return os.path.join(sw_desk, "images", img_url)

    def web_resolver(img_url):
        return os.path.join(sw_web, "images", img_url.split("/images/")[-1])

    sources = [
        ("osatlas_windows",
         lambda: iter_os_atlas(os.path.join(atlas, "windows_splited.json"),
                               os.path.join(atlas, "images", "windows")),
         args.cap_windows, args.quota_windows),
        ("osatlas_linux",
         lambda: iter_os_atlas(os.path.join(atlas, "linux_splited.json"),
                               os.path.join(atlas, "images", "linux")),
         args.cap_linux, args.quota_linux),
        ("osatlas_macos",
         lambda: iter_os_atlas(os.path.join(atlas, "macos_splited.json"),
                               os.path.join(atlas, "images", "macos")),
         args.cap_macos, args.quota_macos),
        ("showui_desktop",
         lambda: iter_showui(os.path.join(sw_desk, "metadata", "hf_train.json"),
                             desk_resolver),
         args.cap_showui_desktop, args.quota_showui_desktop),
        ("showui_web",
         lambda: iter_showui(os.path.join(sw_web, "metadata", "hf_train.json"),
                             web_resolver),
         args.cap_showui_web, args.quota_showui_web),
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
