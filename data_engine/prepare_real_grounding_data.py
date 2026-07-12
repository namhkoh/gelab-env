"""Generate grounding + understanding SFT samples from real AMEX element annotations.

Uses the AMEX element_anno corpus (per-screenshot JSONs with named clickable
elements and pixel bboxes on real app screenshots) to produce the two task
formats that datas/continue_train_24k.json lacks. Prompt templates and the
0-1000 center normalization are copied verbatim from
prepare_amex_gelab_sft_v3.py so the grounding format matches the paper A.5
eval query exactly (the Run 6 alignment fix).

Example:
    python data_engine/prepare_real_grounding_data.py \
        --anno_dir datas/amex_raw/AMEX/element_anno_extracted/element_anno \
        --screenshot_dir datas/amex_raw/AMEX/screenshot_extracted/screenshot \
        --output datas/real_grounding_amex.json \
        --ground_max 40000 --func_max 15000 --understand_max 20000
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

from PIL import Image


def bbox_center_to_1000(bbox, canvas_size):
    w, h = canvas_size
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    x = max(0, min(1000, int(cx / w * 1000)))
    y = max(0, min(1000, int(cy / h * 1000)))
    return x, y


def clean_name(raw):
    name = re.sub(r"\s+", " ", str(raw)).strip()
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_dir", default="datas/amex_raw/AMEX/element_anno_extracted/element_anno")
    parser.add_argument("--screenshot_dir", default="datas/amex_raw/AMEX/screenshot_extracted/screenshot")
    parser.add_argument("--output", default="datas/real_grounding_amex.json")
    parser.add_argument("--ground_max", type=int, default=40000)
    parser.add_argument("--func_max", type=int, default=15000,
                        help="grounding samples phrased from the functionality description")
    parser.add_argument("--understand_max", type=int, default=20000)
    parser.add_argument("--per_image_cap", type=int, default=2,
                        help="max samples per task type drawn from one screenshot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    anno_files = sorted(Path(args.anno_dir).glob("*.json"))
    random.shuffle(anno_files)
    print(f"annotation files: {len(anno_files)}")

    shot_dir = Path(args.screenshot_dir)
    ground, func_ground, understand = [], [], []
    seen_shots = skipped = 0
    for af in anno_files:
        if (len(ground) >= args.ground_max and len(func_ground) >= args.func_max
                and len(understand) >= args.understand_max):
            break
        try:
            anno = json.load(open(af))
        except Exception:
            skipped += 1
            continue
        img_path = shot_dir / anno.get("image_path", "")
        if not img_path.exists():
            skipped += 1
            continue
        try:
            with Image.open(img_path) as im:
                cs = im.size
        except Exception:
            skipped += 1
            continue

        elems = []
        for e in anno.get("clickable_elements", []):
            bbox = e.get("bbox", [])
            descs = e.get("xml_desc") or []
            name = clean_name(descs[0]) if descs else ""
            if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            if len(name) < 2 or len(name) > 80 or name.startswith("("):
                continue
            elems.append((name, bbox, clean_name(e.get("functionality", ""))))

        # names appearing more than once on a screenshot are ambiguous grounding targets
        counts = {}
        for name, _, _ in elems:
            counts[name] = counts.get(name, 0) + 1
        elems = [e for e in elems if counts[e[0]] == 1]
        if not elems:
            continue
        seen_shots += 1
        random.shuffle(elems)

        abs_img = str(img_path.resolve())
        g = u = f = 0
        for name, bbox, functionality in elems:
            cx, cy = bbox_center_to_1000(bbox, cs)
            if f < args.per_image_cap and len(func_ground) < args.func_max and \
                    functionality.lower().startswith("click") and len(functionality) > 10:
                instr = functionality[0].lower() + functionality[1:].rstrip(".")
                func_ground.append({
                    "messages": [
                        {"role": "user", "content": f"<image>I want to {instr}. Please locate the target element I should interact with. (with point)"},
                        {"role": "assistant", "content": f"Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"},
                    ],
                    "images": [abs_img],
                    "source": "amex_real_grounding_func",
                })
                f += 1
            elif g < args.per_image_cap and len(ground) < args.ground_max:
                ground.append({
                    "messages": [
                        {"role": "user", "content": f"<image>I want to click on {name}. Please locate the target element I should interact with. (with point)"},
                        {"role": "assistant", "content": f"Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"},
                    ],
                    "images": [abs_img],
                    "source": "amex_real_grounding",
                })
                g += 1
            elif u < args.per_image_cap and len(understand) < args.understand_max:
                understand.append({
                    "messages": [
                        {"role": "user", "content": f"<image>What is the icon at point ({cx},{cy}) in the image?"},
                        {"role": "assistant", "content": name},
                    ],
                    "images": [abs_img],
                    "source": "amex_real_understanding",
                })
                u += 1

    out = ground + func_ground + understand
    random.shuffle(out)
    for i, s in enumerate(out):
        s["idx"] = i
    with open(args.output, "w") as fp:
        json.dump(out, fp, ensure_ascii=False)
    print(f"screenshots used: {seen_shots}, skipped: {skipped}")
    print(f"grounding: {len(ground)}, functional-grounding: {len(func_ground)}, understanding: {len(understand)}")
    print(f"total: {len(out)} -> {args.output}")


if __name__ == "__main__":
    main()
