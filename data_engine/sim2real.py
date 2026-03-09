"""
Sim2Real Pipeline: Extract real-world icons from GUIOdyssey screenshots.

Uses OmniParser (YOLOv8 + Florence-2 + EasyOCR) to detect UI elements,
crop icons, and build a categorized icon pool for GE-Lab environment
augmentation.

Prerequisites:
    pip install ultralytics easyocr timm
    # OmniParser weights at /ext_hdd2/nhkoh/OmniParser/weights/
    # GUIOdyssey subset at /ext_hdd2/nhkoh/GUI-Odyssey/

Usage:
    python data_engine/sim2real.py \
        --screenshots_dir /ext_hdd2/nhkoh/GUI-Odyssey/screenshots \
        --annotations_dir /ext_hdd2/nhkoh/GUI-Odyssey/annotations \
        --omniparser_weights /ext_hdd2/nhkoh/OmniParser/weights \
        --output_dir data_engine/real_icons \
        --gpu 0
"""

import argparse
import json
import os
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_yolo_model(weights_dir, device="cuda"):
    """Load OmniParser's YOLOv8 icon detection model."""
    from ultralytics import YOLO
    model_path = os.path.join(weights_dir, "icon_detect", "model.pt")
    model = YOLO(model_path)
    return model


def load_caption_model(weights_dir, device="cuda"):
    """Load OmniParser's Florence-2 icon captioning model."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    model_path = os.path.join(weights_dir, "icon_caption")
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True
    )
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    return {"model": model, "processor": processor}


def load_ocr():
    """Load EasyOCR reader."""
    import easyocr
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available())


# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------

def detect_icons_yolo(yolo_model, image_pil, box_threshold=0.05, iou_threshold=0.1):
    """Run YOLO detection, return normalized [x1, y1, x2, y2] boxes and confidences."""
    w, h = image_pil.size
    result = yolo_model.predict(source=image_pil, conf=box_threshold, iou=iou_threshold, verbose=False)
    boxes_px = result[0].boxes.xyxy  # pixel coords
    confs = result[0].boxes.conf
    # Normalize to [0, 1]
    boxes_norm = boxes_px / torch.Tensor([w, h, w, h]).to(boxes_px.device)
    return boxes_norm.cpu(), confs.cpu()


def detect_text_ocr(ocr_reader, image_pil):
    """Run EasyOCR, return (texts, bboxes_xyxy_pixel)."""
    image_np = np.array(image_pil.convert("RGB"))
    results = ocr_reader.readtext(image_np, text_threshold=0.8)
    texts, bboxes = [], []
    for item in results:
        coord = item[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        x1 = int(min(c[0] for c in coord))
        y1 = int(min(c[1] for c in coord))
        x2 = int(max(c[0] for c in coord))
        y2 = int(max(c[1] for c in coord))
        texts.append(item[1])
        bboxes.append([x1, y1, x2, y2])
    return texts, bboxes


def caption_icons(caption_model_processor, image_np, boxes_norm, batch_size=64):
    """Caption icon crops using Florence-2. Returns list of captions."""
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    h, w = image_np.shape[:2]

    crops = []
    valid_indices = []
    for i, box in enumerate(boxes_norm):
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        crop = image_np[y1:y2, x1:x2]
        crop = cv2.resize(crop, (64, 64))
        crops.append(to_pil(crop))
        valid_indices.append(i)

    if not crops:
        return {}, valid_indices

    model = caption_model_processor["model"]
    processor = caption_model_processor["processor"]
    device = model.device
    prompt = "<CAPTION>"

    captions = {}
    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        batch_indices = valid_indices[start : start + batch_size]
        if model.device.type == "cuda":
            inputs = processor(
                images=batch, text=[prompt] * len(batch),
                return_tensors="pt", do_resize=False
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(
                images=batch, text=[prompt] * len(batch),
                return_tensors="pt"
            ).to(device=device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20, num_beams=1, do_sample=False,
            )
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for idx, text in zip(batch_indices, texts):
            captions[idx] = text.strip()

    return captions, valid_indices


def merge_detections(yolo_boxes, yolo_confs, ocr_texts, ocr_bboxes, img_w, img_h,
                     iou_threshold=0.7):
    """Merge YOLO icon detections with OCR text boxes, resolving overlaps."""
    elements = []

    # Add OCR text elements (normalized)
    for text, bbox in zip(ocr_texts, ocr_bboxes):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < 10:
            continue
        elements.append({
            "type": "text",
            "bbox": [bbox[0] / img_w, bbox[1] / img_h, bbox[2] / img_w, bbox[3] / img_h],
            "content": text,
            "interactivity": False,
            "source": "ocr",
        })

    # Add YOLO icon elements, checking overlap with OCR
    for i, (box, conf) in enumerate(zip(yolo_boxes.tolist(), yolo_confs.tolist())):
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area < 1e-5:
            continue
        # Check if this icon overlaps heavily with any OCR box
        ocr_label = None
        for text, ocr_box_px in zip(ocr_texts, ocr_bboxes):
            ocr_norm = [ocr_box_px[0] / img_w, ocr_box_px[1] / img_h,
                        ocr_box_px[2] / img_w, ocr_box_px[3] / img_h]
            iou = _compute_iou(box, ocr_norm)
            if iou > 0.3:
                ocr_label = text
                break

        elements.append({
            "type": "icon",
            "bbox": box,
            "content": ocr_label,  # filled by captioning later if None
            "interactivity": True,
            "confidence": conf,
            "source": "yolo_ocr" if ocr_label else "yolo",
            "yolo_idx": i,
        })

    return elements


def _compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter + 1e-8
    return inter / union


# ---------------------------------------------------------------------------
# Icon extraction and deduplication
# ---------------------------------------------------------------------------

def crop_and_save_icon(image_pil, bbox_norm, output_path, target_size=(50, 50)):
    """Crop icon from image using normalized bbox, resize, and save."""
    w, h = image_pil.size
    x1 = int(bbox_norm[0] * w)
    y1 = int(bbox_norm[1] * h)
    x2 = int(bbox_norm[2] * w)
    y2 = int(bbox_norm[3] * h)

    # Skip tiny crops
    if x2 - x1 < 10 or y2 - y1 < 10:
        return False

    crop = image_pil.crop((x1, y1, x2, y2))
    crop = crop.resize(target_size, Image.LANCZOS)
    crop.save(output_path, "PNG")
    return True


def compute_image_hash(image_path):
    """Compute perceptual hash (average hash) for deduplication."""
    img = Image.open(image_path).convert("L").resize((8, 8), Image.LANCZOS)
    pixels = np.array(img)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    hash_val = sum(1 << i for i, b in enumerate(bits) if b)
    return hash_val


def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count("1")


def deduplicate_icons(icon_dir, threshold=5):
    """Remove near-duplicate icons using perceptual hashing."""
    icons = sorted(Path(icon_dir).glob("*.png"))
    if not icons:
        return 0

    hashes = {}
    removed = 0
    for icon_path in icons:
        h = compute_image_hash(icon_path)
        is_dup = False
        for existing_path, existing_hash in hashes.items():
            if hamming_distance(h, existing_hash) <= threshold:
                is_dup = True
                break
        if is_dup:
            icon_path.unlink()
            removed += 1
        else:
            hashes[str(icon_path)] = h

    return removed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_screenshot(image_path, yolo_model, caption_mp, ocr_reader, output_dir,
                       icon_counter, target_size=(50, 50)):
    """Process a single screenshot: detect, caption, crop icons.

    Returns list of element dicts with icon paths.
    """
    image_pil = Image.open(image_path).convert("RGB")
    w, h = image_pil.size
    image_np = np.array(image_pil)

    # 1. YOLO detection
    yolo_boxes, yolo_confs = detect_icons_yolo(yolo_model, image_pil)

    # 2. OCR detection
    ocr_texts, ocr_bboxes = detect_text_ocr(ocr_reader, image_pil)

    # 3. Merge detections
    elements = merge_detections(yolo_boxes, yolo_confs, ocr_texts, ocr_bboxes, w, h)

    # 4. Caption uncaptioned icons
    icon_elements = [e for e in elements if e["type"] == "icon"]
    uncaptioned_indices = [e["yolo_idx"] for e in icon_elements if e["content"] is None]
    if uncaptioned_indices and caption_mp is not None:
        # Get boxes for uncaptioned icons
        uncaptioned_boxes = yolo_boxes[uncaptioned_indices]
        captions, _ = caption_icons(caption_mp, image_np, uncaptioned_boxes)
        # Map back
        caption_map = {}
        for i, orig_idx in enumerate(uncaptioned_indices):
            if i in captions:
                caption_map[orig_idx] = captions[i]
        for e in icon_elements:
            if e["content"] is None and e["yolo_idx"] in caption_map:
                e["content"] = caption_map[e["yolo_idx"]]
                e["source"] = "yolo_florence"

    # 5. Crop and save icons
    os.makedirs(output_dir, exist_ok=True)
    saved_elements = []
    for e in icon_elements:
        label = (e.get("content") or "unknown").replace("/", "_").replace(" ", "_")[:30]
        icon_name = f"icon_{icon_counter[0]:05d}_{label}.png"
        icon_path = os.path.join(output_dir, icon_name)
        success = crop_and_save_icon(image_pil, e["bbox"], icon_path, target_size)
        if success:
            e["icon_path"] = icon_path
            icon_counter[0] += 1
            saved_elements.append(e)

    return saved_elements


def run_pipeline(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print("Loading YOLO model...")
    yolo_model = load_yolo_model(args.omniparser_weights, device)
    yolo_model.to(device)

    caption_mp = None
    if not args.skip_caption:
        print("Loading Florence-2 caption model...")
        caption_mp = load_caption_model(args.omniparser_weights, device)

    print("Loading EasyOCR...")
    ocr_reader = load_ocr()

    # Collect screenshots
    screenshots_dir = Path(args.screenshots_dir)
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else None

    png_files = sorted(screenshots_dir.glob("*.png"))
    print(f"Found {len(png_files)} screenshots")

    # Process each screenshot
    os.makedirs(args.output_dir, exist_ok=True)
    all_metadata = []
    icon_counter = [0]  # mutable counter

    for i, img_path in enumerate(png_files):
        print(f"[{i+1}/{len(png_files)}] {img_path.name}", end="")
        elements = process_screenshot(
            str(img_path), yolo_model, caption_mp, ocr_reader,
            os.path.join(args.output_dir, "all_icons"), icon_counter,
            target_size=(args.icon_size, args.icon_size),
        )
        print(f"  -> {len(elements)} icons")

        for e in elements:
            e.pop("yolo_idx", None)
            e.pop("confidence", None)
            e["source_screenshot"] = img_path.name
        all_metadata.extend(elements)

    # Save metadata
    meta_path = os.path.join(args.output_dir, "icons_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\nSaved {len(all_metadata)} icon metadata to {meta_path}")

    # Deduplicate
    if not args.skip_dedup:
        print("Deduplicating icons...")
        icons_dir = os.path.join(args.output_dir, "all_icons")
        removed = deduplicate_icons(icons_dir, threshold=args.dedup_threshold)
        remaining = len(list(Path(icons_dir).glob("*.png")))
        print(f"Removed {removed} duplicates, {remaining} unique icons remaining")

    # Organize by category (using Florence-2 captions)
    if not args.skip_organize:
        organize_icons(args.output_dir)

    print("\nDone.")


def organize_icons(output_dir):
    """Organize icons into category subdirectories based on captions."""
    icons_dir = os.path.join(output_dir, "all_icons")
    meta_path = os.path.join(output_dir, "icons_metadata.json")

    if not os.path.exists(meta_path):
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    # Build mapping from icon filename to caption
    icon_to_caption = {}
    for m in metadata:
        if "icon_path" in m:
            fname = os.path.basename(m["icon_path"])
            icon_to_caption[fname] = m.get("content", "unknown")

    # Simple keyword-based categorization
    categories = {
        "Navigation": ["back", "home", "menu", "arrow", "hamburger", "drawer", "tab"],
        "Social": ["instagram", "facebook", "twitter", "tiktok", "threads", "x ", "share",
                    "like", "heart", "comment", "follow", "post", "story", "message"],
        "Shopping": ["cart", "buy", "shop", "amazon", "ebay", "price", "checkout",
                      "aliexpress", "product", "order", "delivery"],
        "Media": ["play", "pause", "video", "youtube", "spotify", "music", "audio",
                  "camera", "photo", "gallery", "image"],
        "Productivity": ["calendar", "email", "gmail", "docs", "office", "word", "note",
                         "todo", "clock", "alarm", "setting", "gear"],
        "Search": ["search", "magnif", "find", "lens", "google"],
        "Browser": ["chrome", "firefox", "opera", "edge", "safari", "browser", "web",
                     "duckduck", "url", "address"],
    }

    categorized = 0
    for icon_file in Path(icons_dir).glob("*.png"):
        caption = (icon_to_caption.get(icon_file.name) or "").lower()
        assigned = "Other"
        for cat, keywords in categories.items():
            if any(kw in caption for kw in keywords):
                assigned = cat
                break

        cat_dir = os.path.join(output_dir, assigned, "PNG")
        os.makedirs(cat_dir, exist_ok=True)
        dest = os.path.join(cat_dir, icon_file.name)
        if not os.path.exists(dest):
            icon_file.rename(dest)
            categorized += 1

    print(f"Organized {categorized} icons into categories")

    # Print distribution
    for cat_dir in sorted(Path(output_dir).iterdir()):
        if cat_dir.is_dir() and cat_dir.name != "all_icons":
            png_dir = cat_dir / "PNG"
            if png_dir.exists():
                count = len(list(png_dir.glob("*.png")))
                print(f"  {cat_dir.name}: {count} icons")


def parse_args():
    parser = argparse.ArgumentParser(description="Sim2Real icon extraction pipeline")
    parser.add_argument("--screenshots_dir", type=str,
                        default="/ext_hdd2/nhkoh/GUI-Odyssey/screenshots",
                        help="Directory with GUIOdyssey screenshots")
    parser.add_argument("--annotations_dir", type=str,
                        default="/ext_hdd2/nhkoh/GUI-Odyssey/annotations",
                        help="Directory with GUIOdyssey annotation JSONs")
    parser.add_argument("--omniparser_weights", type=str,
                        default="/ext_hdd2/nhkoh/OmniParser/weights",
                        help="OmniParser weights directory")
    parser.add_argument("--output_dir", type=str,
                        default="data_engine/real_icons",
                        help="Output directory for extracted icons")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--icon_size", type=int, default=50,
                        help="Target icon size (square)")
    parser.add_argument("--skip_caption", action="store_true",
                        help="Skip Florence-2 captioning (faster)")
    parser.add_argument("--skip_dedup", action="store_true",
                        help="Skip deduplication pass")
    parser.add_argument("--skip_organize", action="store_true",
                        help="Skip category organization")
    parser.add_argument("--dedup_threshold", type=int, default=5,
                        help="Perceptual hash hamming distance threshold for dedup")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
