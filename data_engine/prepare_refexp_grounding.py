"""Generate grounding + understanding SFT samples from the ivelin/ui_refexp TRAIN split.

The real-world eval (eval/evaluate_real_world.py::load_refexp) tests on the TEST
split of the same dataset with the query
    "I want to {prompt}. Please locate the target element I should interact with. (with point)"
where {prompt} is the raw referring expression (usually a verb phrase such as
"click on the save icon"). This script mirrors that distribution using ONLY the
train split. Prompt templates and the 0-1000 center normalization are copied
verbatim from prepare_real_grounding_data.py so the output matches the paper
A.5 grounding format exactly. The test/validation splits are never read.

Dataset facts (train split, v1.1.0):
  - 15,624 rows over 4,646 unique RICO screenshots (JPEG, PIL-decoded).
  - fields: image, image_id, image_file_path, prompt, target_bounding_box
  - target_bounding_box is a JSON string {"xmin","ymin","xmax","ymax"} in 0-1
    normalized coordinates (same convention the eval loader assumes).
  - prompts are mostly verb phrases ("click on X", "select Y", "go to Z");
    ~4% are bare noun phrases ("the red colored text mentioned as 706");
    ~50 rows are annotator junk markers ("improper tagging", "no bounding
    box", "it is clickable", "incorrect bounding box") that must be dropped.

Example:
    python data_engine/prepare_refexp_grounding.py \
        --cache_dir /workspace/nhkoh/hf_cache/datasets \
        --output datas/refexp_grounding.json \
        --image_dir datas/real_world_images/refexp \
        --understand_frac 0.15
"""
import argparse
import io
import json
import random
import re
from pathlib import Path

from PIL import Image

# Annotator failure markers, not referring expressions.
JUNK_RE = re.compile(
    r"^(improper|incorrect|no bounding|no tag|not tagged|it is clickable|"
    r"there is a bounding)", re.I)

# Verb-only prompts with no target ("go to", "click on") carry no grounding signal.
BARE_VERB_RE = re.compile(
    r"^(?:please\s+)?(?:go\s*to|click(?:\s+on)?|select|tap(?:\s+on)?|choose|"
    r"press|move\s+to)$", re.I)

# First words that already read as an imperative verb phrase after "I want to ...".
# Includes the click/select typos observed in the corpus so they are not treated
# as noun phrases. Anything else gets a "click on " prefix.
VERB_FIRST = {
    "click", "clcik", "clickl", "clicki", "cleck", "chick", "slick", "klick",
    "clik", "select", "selected", "slect", "sekect", "select2", "selecticon",
    "tap", "press", "hit", "pick", "choose", "choode", "go", "goto", "move",
    "navigate", "visit", "open", "close", "enter", "take", "share", "check",
    "uncheck", "mark", "tag", "read", "type", "view", "expand", "switch",
    "turn", "enable", "disable", "launch", "toggle", "skip", "hide", "show",
    "submit", "log", "search", "play", "pause", "drag", "scroll", "swipe",
    "activate", "consider", "resend",
}

# Strips a leading verb to recover the noun phrase for understanding samples,
# e.g. "click on the save icon" -> "the save icon".
STRIP_VERB_RE = re.compile(
    r"^(?:please\s+)?"
    r"(?:(?:click|clcik|clickl|clicki|cleck|chick|slick|klick|clik|"
    r"select|selected|slect|sekect|tap|press|hit|pick|choose|choode|open|"
    r"check|mark|tag|view|enter|read)\s+(?:on\s+|at\s+|onto\s+)?"
    r"|(?:go\s+to|goto|move\s+to|navigate\s+to|visit)\s+)", re.I)

GROUND_USER = ("<image>I want to {instr}. Please locate the target element "
               "I should interact with. (with point)")
GROUND_ASSISTANT = "Action: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"
UNDERSTAND_USER = "<image>What is the icon at point ({cx},{cy}) in the image?"


def clean_prompt(raw):
    p = re.sub(r"\s+", " ", str(raw)).strip()
    p = p.rstrip(". ").strip()
    if p:
        p = p[0].lower() + p[1:]
    return p


def bbox_center_to_1000(bb):
    """bb: dict with 0-1 normalized xmin/ymin/xmax/ymax -> (cx, cy) in 0-1000."""
    cx = (bb["xmin"] + bb["xmax"]) / 2
    cy = (bb["ymin"] + bb["ymax"]) / 2
    x = max(0, min(1000, int(round(cx * 1000))))
    y = max(0, min(1000, int(round(cy * 1000))))
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/workspace/nhkoh/hf_cache/datasets")
    parser.add_argument("--output", default="datas/refexp_grounding.json")
    parser.add_argument("--image_dir", default="datas/real_world_images/refexp")
    parser.add_argument("--understand_frac", type=float, default=0.15,
                        help="fraction of valid rows that also emit an understanding sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("ivelin/ui_refexp", split="train",
                      cache_dir=args.cache_dir, trust_remote_code=True)
    # Keep encoded JPEG bytes so each screenshot can be written to disk verbatim.
    ds = ds.cast_column("image", ds.features["image"].__class__(decode=False))
    print(f"train rows: {len(ds)}")

    ground, understand_cands = [], []
    saved_images = {}  # image_id -> absolute path
    stats = {"junk": 0, "bare_verb": 0, "short": 0, "bad_bbox": 0,
             "dup": 0, "bad_image": 0}
    seen = set()

    for row in ds:
        prompt = clean_prompt(row["prompt"])
        if len(prompt) < 1:
            stats["short"] += 1
            continue
        if JUNK_RE.match(prompt):
            stats["junk"] += 1
            continue
        if BARE_VERB_RE.match(prompt):
            stats["bare_verb"] += 1
            continue

        try:
            bb = json.loads(row["target_bounding_box"])
            vals = [float(bb[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (KeyError, TypeError, ValueError):
            stats["bad_bbox"] += 1
            continue
        x1, y1, x2, y2 = [max(0.0, min(1.0, v)) for v in vals]
        if x2 <= x1 or y2 <= y1:
            stats["bad_bbox"] += 1
            continue

        image_id = str(row["image_id"])
        key = (image_id, prompt)
        if key in seen:
            stats["dup"] += 1
            continue
        seen.add(key)

        if image_id not in saved_images:
            img_bytes = row["image"]["bytes"]
            try:
                with Image.open(io.BytesIO(img_bytes)) as im:
                    im.load()
            except Exception:
                stats["bad_image"] += 1
                saved_images[image_id] = None
                continue
            out_path = image_dir / f"{image_id}.jpg"
            if not out_path.exists():
                out_path.write_bytes(img_bytes)
            saved_images[image_id] = str(out_path.resolve())
        abs_img = saved_images[image_id]
        if abs_img is None:
            stats["bad_image"] += 1
            continue

        cx, cy = bbox_center_to_1000({"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2})

        first = prompt.split()[0]
        instr = prompt if first in VERB_FIRST else f"click on {prompt}"
        ground.append({
            "messages": [
                {"role": "user", "content": GROUND_USER.format(instr=instr)},
                {"role": "assistant", "content": GROUND_ASSISTANT.format(cx=cx, cy=cy)},
            ],
            "images": [abs_img],
            "source": "refexp_grounding",
        })

        # Understanding candidate: recover the noun phrase the point refers to.
        noun = STRIP_VERB_RE.sub("", prompt).strip()
        if first not in VERB_FIRST:
            noun = prompt  # already a noun phrase
        elif noun == prompt:
            noun = ""  # verb we cannot strip cleanly -> skip
        if len(noun) >= 3:
            understand_cands.append({
                "messages": [
                    {"role": "user", "content": UNDERSTAND_USER.format(cx=cx, cy=cy)},
                    {"role": "assistant", "content": noun},
                ],
                "images": [abs_img],
                "source": "refexp_understanding",
            })

    n_und = min(len(understand_cands), int(round(args.understand_frac * len(ground))))
    understand = random.sample(understand_cands, n_und)

    out = ground + understand
    random.shuffle(out)
    for i, s in enumerate(out):
        s["idx"] = i
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump(out, fp, ensure_ascii=False)

    n_imgs = sum(1 for v in saved_images.values() if v)
    print(f"filtered: {stats}")
    print(f"images saved: {n_imgs} -> {image_dir}")
    print(f"grounding: {len(ground)}, understanding: {len(understand)} "
          f"(candidates: {len(understand_cands)})")
    print(f"total: {len(out)} -> {args.output}")


if __name__ == "__main__":
    main()
