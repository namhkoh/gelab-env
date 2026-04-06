"""
Convert amex-gelab trajectories to ms-swift SFT format with all 3 task types:
1. Navigation: trajectory-based click actions
2. Grounding: "Click on [element] in the image" -> click coordinates
3. Understanding: "What is the icon at point (x,y)?" -> element name

Usage:
    python data_engine/prepare_amex_gelab_sft_v3.py \
        --extracted_dir datas/amex_gelab_extracted/amex_sft \
        --output datas/amex_gelab_sft_v3.json \
        --image_dir datas/amex_gelab_images_v2
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def action_coord_to_1000(coord, canvas_size):
    w, h = canvas_size
    x = max(0, min(1000, int(coord[0] / w * 1000)))
    y = max(0, min(1000, int(coord[1] / h * 1000)))
    return x, y


def bbox_center_to_1000(bbox, canvas_size):
    w, h = canvas_size
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    x = max(0, min(1000, int(cx / w * 1000)))
    y = max(0, min(1000, int(cy / h * 1000)))
    return x, y


def get_image_path(image_dir, page_img_src, traj_action_coord_dir):
    """Copy screenshot and return absolute path. Reuse if already copied."""
    src = traj_action_coord_dir / page_img_src
    if not src.exists():
        return None
    # Use content-addressed name to avoid duplicates
    dest_name = f"{src.parent.parent.name}_{page_img_src}"
    dest = os.path.join(image_dir, dest_name)
    if not os.path.exists(dest):
        shutil.copy2(str(src), dest)
    return os.path.abspath(dest)


def process_trajectory(traj_dir, image_dir, canvas_size):
    ui_path = traj_dir / "ui_structure.json"
    if not ui_path.exists():
        return [], [], []

    with open(ui_path) as f:
        ui_structure = json.load(f)

    metadata = ui_structure["metadata"]
    pages = ui_structure["pages"]
    instruction = metadata["instruction"]
    cs = metadata.get("canvas_size", canvas_size)
    action_coord_dir = traj_dir / "action_coord"

    nav_samples = []
    grounding_samples = []
    understanding_samples = []

    for page_name, page_data in pages.items():
        img_file = page_data.get("image", f"{page_name}.png")
        img_path = get_image_path(image_dir, img_file, action_coord_dir)
        if img_path is None:
            continue

        layout = page_data.get("layout", {})

        # --- Navigation samples (from TAP transitions) ---
        for trans in page_data.get("transitions", []):
            if trans["action"] != "TAP":
                continue
            target_page = trans["target_page"]
            icon_name = trans.get("icon_name", "element")
            ax, ay = action_coord_to_1000(trans["action_coord"], cs)
            action_str = f"click(start_box='<|box_start|>({ax},{ay})<|box_end|>')"
            explain = f"click {icon_name} on {page_name} to navigate to {target_page}."

            nav_samples.append({
                "messages": [
                    {"role": "user", "content": f"<image>{instruction}"},
                    {"role": "assistant", "content": f"Explain: {explain}\tAction: {action_str}"},
                ],
                "images": [img_path],
                "source": "amex_gelab_nav",
            })

        # --- Grounding + Understanding samples (from layout) ---
        for elem_name, elem_data in layout.items():
            # Skip unnamed/garbage elements
            if len(elem_name) < 2 or elem_name.startswith("("):
                continue
            bbox = elem_data.get("bbox", [])
            if len(bbox) != 4:
                continue

            cx, cy = bbox_center_to_1000(bbox, cs)

            # Grounding: "Click on [element]" -> coordinates
            grounding_samples.append({
                "messages": [
                    {"role": "user", "content": f"<image>Click on {elem_name} in the image."},
                    {"role": "assistant", "content": f"Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"},
                ],
                "images": [img_path],
                "source": "amex_gelab_grounding",
            })

            # Understanding: "What is at (x,y)?" -> element name
            clean_name = elem_name.replace("_", " ")
            understanding_samples.append({
                "messages": [
                    {"role": "user", "content": f"<image>What is the icon at point ({cx},{cy}) in the image?"},
                    {"role": "assistant", "content": clean_name},
                ],
                "images": [img_path],
                "source": "amex_gelab_understanding",
            })

    return nav_samples, grounding_samples, understanding_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_dir", default="datas/amex_gelab_extracted/amex_sft")
    parser.add_argument("--output", default="datas/amex_gelab_sft_v3.json")
    parser.add_argument("--image_dir", default="datas/amex_gelab_images_v2")
    parser.add_argument("--nav_max", type=int, default=8000, help="Max navigation samples")
    parser.add_argument("--ground_max", type=int, default=8000, help="Max grounding samples")
    parser.add_argument("--understand_max", type=int, default=8000, help="Max understanding samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.image_dir, exist_ok=True)

    extracted = Path(args.extracted_dir)
    traj_dirs = sorted([d for d in extracted.iterdir() if d.is_dir()])
    print(f"Found {len(traj_dirs)} trajectories")

    all_nav = []
    all_ground = []
    all_understand = []

    for traj_dir in tqdm(traj_dirs, desc="Trajectories"):
        nav, ground, understand = process_trajectory(
            traj_dir, args.image_dir, [1080, 2400]
        )
        all_nav.extend(nav)
        all_ground.extend(ground)
        all_understand.extend(understand)

    print(f"\nRaw counts: nav={len(all_nav)}, grounding={len(all_ground)}, understanding={len(all_understand)}")

    # Subsample to target sizes
    random.shuffle(all_nav)
    random.shuffle(all_ground)
    random.shuffle(all_understand)
    all_nav = all_nav[:args.nav_max]
    all_ground = all_ground[:args.ground_max]
    all_understand = all_understand[:args.understand_max]

    all_samples = all_nav + all_ground + all_understand
    random.shuffle(all_samples)
    for i, s in enumerate(all_samples):
        s["idx"] = i

    with open(args.output, "w") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    from collections import Counter
    src_counts = Counter(s["source"] for s in all_samples)
    print(f"\nFinal dataset: {len(all_samples)} samples")
    for src, cnt in sorted(src_counts.items()):
        print(f"  {src}: {cnt}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
