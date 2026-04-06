"""
Real-World GUI Benchmark Evaluation (GE-Lab Section 6.2, Table 5)
=================================================================
Evaluates continue-trained models on 7 static grounding benchmarks:
  ScreenSpot, ScreenSpot-v2, FuncPred, MoTIF, Refexp, VWB-AG, VWB-EG

All benchmarks use click accuracy (point-in-bbox) as the metric.
AndroidWorld is excluded (requires Android emulator infrastructure).

Usage:
    # Evaluate all 7 benchmarks
    python eval/evaluate_real_world.py --model_path <checkpoint>

    # Evaluate specific benchmarks
    python eval/evaluate_real_world.py --model_path <checkpoint> \
        --benchmarks screenspot screenspot_v2

    # Multi-GPU evaluation
    python eval/evaluate_real_world.py --model_path <checkpoint> --num_gpus 3
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NORM_SIZE = 1000
MAX_NEW_TOKENS = 256

# Real-world system prompt (Paper Appendix A.5)
SYSTEM_PROMPT = (
    "You are a Multifaceted Mobile Interface Assistant. Your responsibilities include:\n"
    "\n"
    "- 1. Navigating a mobile phone interface to reach a target page based on user "
    "instructions, task history, and the current screen state.\n"
    "- 2. Understanding icons by identifying their name or function based on their "
    "location on the screen.\n"
    "- 3. Grounding icons by locating the coordinates of an icon based on its name "
    "or description.\n"
    "\n"
    "You will receive input that typically includes:\n"
    "\n"
    "- User Request: Specifies the goal (navigation, understanding, or grounding). "
    "This might be a complex instruction for navigation or a direct question/command "
    "for icon tasks.\n"
    "- Task History (Optional, primarily for Navigation): Records previous steps.\n"
    "- Current Screen State: Represents the current screen, an image (indicated by "
    "<image>).\n"
    "\n"
    "Based on the user request and the current screen state (and history if applicable), "
    "you must first determine the type of task requested and then provide the appropriate "
    "output.\n"
    "\n"
    "Task Types and Output Formats\n"
    "\n"
    "General GUI Task\n"
    "\n"
    "- Goal: Reach a target page step-by-step.\n"
    "- Typical Input: Multi-turn instruction, history, and state. screen description and "
    "screenshot.\n"
    "- Possible Actions:\n"
    "  - click: Tap a specific element. Provide coordinates (x, y) relative to a (0,0) "
    "top-left and (1000,1000) bottom-right system.\n"
    "  - type: Type text into an input field. Provide the string content to be typed. "
    'Example: TYPE("Texas BBQ")\n'
    "  - scroll: Scroll the screen. Provide the scroll distance. Example: SCROLL(5)\n"
    "  - wait: Pause the execution. Provide the duration to wait in seconds. Example: "
    "WAIT(3)\n"
    "  - complete: Task finished, current screen is the target.\n"
    "- Output Format:\n"
    "Explain: [Your brief explanation, e.g., 'click xxx icon on yyy page.', 'this is the "
    "target page.']\tAction: [click(start_box=<|box_start|>(x,y)<|box_end|>) or "
    'TYPE("Text to type") or SCROLL(5) or WAIT(3) or complete]\n'
    "\n"
    "--- General Instructions ---\n"
    "\n"
    "- Carefully analyze the user request to determine the task (Navigation, Grounding, "
    "Understanding).\n"
    "- Analyze the current screen state (description or image) thoroughly.\n"
    "- For actions involving coordinates (click), use the (0,0) to (1000,1000) system.\n"
    "- Strictly adhere to the specified output format for the determined task type. "
    "Use a tab character (\\t) as a separator where indicated."
)

# Grounding prompt for GE-Lab fine-tuned models (Paper Appendix A.5)
GROUNDING_PROMPT = (
    "I want to {instruction}. "
    "Please locate the target element I should interact with. (with point)"
)

# Grounding prompt for base Qwen2.5-VL (native bbox_2d output format)
GROUNDING_PROMPT_BASE = (
    "{instruction}\n"
    "Please provide the bounding box coordinate of the region this sentence describes."
)


# ---------------------------------------------------------------------------
# Coordinate parsing (reused from evaluate.py)
# ---------------------------------------------------------------------------
COORD_PATTERN_FULL = re.compile(
    r"(?:click|tap)\(start_box='<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>'\)"
)
COORD_PATTERN_XY = re.compile(r"(?:click|tap)\(x=(\d+),\s*y=(\d+)\)")
COORD_PATTERN_SIMPLE = re.compile(r"\((\d+),\s*(\d+)\)")
COORD_PATTERN_BBOX2D = re.compile(r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]')
# Qwen2.5-VL native <points> format: pixel coordinates
COORD_PATTERN_POINTS = re.compile(r'<points\s+x1="(\d+)"?\s+y1="(\d+)"')
COORD_PATTERN_POINTS_COMMA = re.compile(r'<points\s+x1="(\d+),(\d+)"')


def parse_click_coords(response: str, image_size=None):
    """Extract (x, y) coordinates from model response. Returns None if not found.

    Handles formats:
      - click(start_box='<|box_start|>(x,y)<|box_end|>')  (GE-Lab trained, 0-1000)
      - tap(start_box='<|box_start|>(x,y)<|box_end|>')    (AMEX SFT trained, 0-1000)
      - click(x=N, y=N)                                    (base model variant)
      - <points x1="X" y1="Y" ...>                         (Qwen2.5-VL native, pixel coords)
      - <points x1="X,Y" ...>                               (Qwen2.5-VL comma variant)
      - (x, y)                                              (simple pair, 0-1000)
      - bbox_2d: [x1,y1,x2,y2]                             (Qwen2.5-VL native, pixel coords)

    When image_size is provided and pixel-space formats are detected, converts
    to 0-1000 normalized coordinates.
    """
    # Try GE-Lab format first (already in 0-1000)
    m = COORD_PATTERN_FULL.search(response)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = COORD_PATTERN_XY.search(response)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Qwen2.5-VL native <points> format (pixel coordinates -> normalize)
    m = COORD_PATTERN_POINTS.search(response)
    if m and image_size:
        px, py = int(m.group(1)), int(m.group(2))
        return int(round(px / image_size[0] * NORM_SIZE)), int(round(py / image_size[1] * NORM_SIZE))

    # <points x1="X,Y"> comma variant
    m = COORD_PATTERN_POINTS_COMMA.search(response)
    if m and image_size:
        px, py = int(m.group(1)), int(m.group(2))
        return int(round(px / image_size[0] * NORM_SIZE)), int(round(py / image_size[1] * NORM_SIZE))

    # Try Qwen2.5-VL native bbox_2d format (pixel coordinates -> normalize)
    m = COORD_PATTERN_BBOX2D.search(response)
    if m and image_size:
        x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        cx = (x1 + x2) / 2 / image_size[0] * NORM_SIZE
        cy = (y1 + y2) / 2 / image_size[1] * NORM_SIZE
        return int(round(cx)), int(round(cy))

    # Simple (x,y) pattern (assumed 0-1000 from GE-Lab/continue-train)
    m = COORD_PATTERN_SIMPLE.search(response)
    if m:
        return int(m.group(1)), int(m.group(2))

    return None


def point_in_bbox(x, y, bbox):
    """Check if (x, y) falls inside bbox [x1, y1, x2, y2] in 0-1000 space."""
    if len(bbox) != 4:
        return False
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class ModelWrapper:
    """Wraps Qwen2.5-VL for real-world benchmark inference."""

    def __init__(self, model_path, lora_path=None, device="cuda:0", max_pixels=None):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.device = device
        print(f"Loading model from {model_path} on {device}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device,
        )
        proc_kwargs = {}
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)

        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()

    def predict(self, image, text, system_prompt=None):
        """Run inference. image can be PIL.Image or file path.
        
        When system_prompt is None (default for grounding benchmarks), no system
        message is injected so the model uses its native coordinate format.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]})
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text_input], images=[image], return_tensors="pt", padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=(
                    self.processor.tokenizer.pad_token_id
                    or self.processor.tokenizer.eos_token_id
                ),
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        return response


# ---------------------------------------------------------------------------
# Benchmark loaders
# Each returns list of dicts: {image: PIL.Image, instruction: str, bbox_norm: [x1,y1,x2,y2]}
# bbox_norm is in 0-1000 coordinate space.
# ---------------------------------------------------------------------------

def _norm01_to_1000(bbox_01):
    """Convert [x1,y1,x2,y2] in 0-1 range to 0-1000."""
    return [int(v * NORM_SIZE) for v in bbox_01]


def load_screenspot(cache_dir):
    """ScreenSpot: rootsautomation/ScreenSpot, bbox in normalized 0-1."""
    from datasets import load_dataset
    ds = load_dataset("rootsautomation/ScreenSpot", split="test", cache_dir=cache_dir,
                      trust_remote_code=True)
    samples = []
    for row in ds:
        bbox = row["bbox"]
        samples.append({
            "image": row["image"],
            "instruction": row["instruction"],
            "bbox_norm": _norm01_to_1000(bbox),
        })
    return samples


def load_screenspot_v2(cache_dir):
    """ScreenSpot-v2: HongxinLi/ScreenSpot_v2, bbox in normalized 0-1."""
    from datasets import load_dataset
    ds = load_dataset("HongxinLi/ScreenSpot_v2", split="test", cache_dir=cache_dir,
                      trust_remote_code=True)
    samples = []
    for row in ds:
        bbox = row["bbox"]
        samples.append({
            "image": row["image"],
            "instruction": row["instruction"],
            "bbox_norm": _norm01_to_1000(bbox),
        })
    return samples


def load_funcpred(cache_dir):
    """FuncPred: AutoGUI eval set, unnormalized_box in pixels.
    Tries multiple HuggingFace sources. If unavailable, download manually from
    the AutoGUI GitHub: https://github.com/ZJULiHongxin/AutoGUI
    """
    from datasets import load_dataset
    hf_ids = [
        "HongxinLi/AutoGUI-FuncPred",
        "AutoGUI/AutoGUI-FuncPred",
    ]
    ds = None
    for hf_id in hf_ids:
        try:
            ds = load_dataset(hf_id, split="test", cache_dir=cache_dir,
                              trust_remote_code=True)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(
            "Could not load FuncPred from HuggingFace. "
            "Download manually from https://github.com/ZJULiHongxin/AutoGUI "
            "and place as a local dataset, then use --benchmarks to skip funcpred."
        )
    samples = []
    for row in ds:
        img = row["image"]
        # Parse image_size "WxH"
        size_str = row.get("image_size", "")
        if "x" in str(size_str):
            w, h = [int(x) for x in str(size_str).split("x")]
        else:
            w, h = img.size
        ubox = row["unnormalized_box"]  # [left, top, right, bottom] in pixels
        bbox_norm = [
            int((ubox[0] / w) * NORM_SIZE),
            int((ubox[1] / h) * NORM_SIZE),
            int((ubox[2] / w) * NORM_SIZE),
            int((ubox[3] / h) * NORM_SIZE),
        ]
        instruction = row.get("instruction", row.get("func", ""))
        samples.append({
            "image": img,
            "instruction": instruction,
            "bbox_norm": bbox_norm,
        })
    return samples


def load_motif(cache_dir):
    """MoTIF: HongxinLi/MOTIF-EVAL, screen bbox + screen dimensions."""
    from datasets import load_dataset
    ds = load_dataset("HongxinLi/MOTIF-EVAL", split="test_au_tu", cache_dir=cache_dir,
                      trust_remote_code=True)
    samples = []
    for row in ds:
        screen_bbox = row.get("ui_pos_obj_screen_bbox", [])
        if not screen_bbox or len(screen_bbox) < 4:
            continue
        sw = row.get("screen_w", 1)
        sh = row.get("screen_h", 1)
        if sw <= 0 or sh <= 0:
            img = row["image"]
            sw, sh = img.size
        bbox_norm = [
            int((screen_bbox[0] / sw) * NORM_SIZE),
            int((screen_bbox[1] / sh) * NORM_SIZE),
            int((screen_bbox[2] / sw) * NORM_SIZE),
            int((screen_bbox[3] / sh) * NORM_SIZE),
        ]
        samples.append({
            "image": row["image"],
            "instruction": row.get("instr", ""),
            "bbox_norm": bbox_norm,
        })
    return samples


def load_refexp(cache_dir):
    """Refexp (UIBert): ivelin/ui_refexp, target_bounding_box JSON string."""
    from datasets import load_dataset
    ds = load_dataset("ivelin/ui_refexp", split="test", cache_dir=cache_dir,
                      trust_remote_code=True)
    samples = []
    for row in ds:
        bb_str = row.get("target_bounding_box", "{}")
        if isinstance(bb_str, str):
            bb = json.loads(bb_str)
        else:
            bb = bb_str
        xmin = bb.get("xmin", 0)
        ymin = bb.get("ymin", 0)
        xmax = bb.get("xmax", 0)
        ymax = bb.get("ymax", 0)
        bbox_norm = _norm01_to_1000([xmin, ymin, xmax, ymax])
        samples.append({
            "image": row["image"],
            "instruction": row.get("prompt", ""),
            "bbox_norm": bbox_norm,
        })
    return samples


def load_vwb_eg(cache_dir):
    """VWB Element Grounding: HongxinLi/VWB-EG, box in normalized 0-1."""
    from datasets import load_dataset
    ds = load_dataset("HongxinLi/VWB-EG", split="test", cache_dir=cache_dir,
                      trust_remote_code=True)
    samples = []
    for row in ds:
        bbox = row.get("box", [])
        if len(bbox) < 4:
            continue
        instruction = row.get("detailed_elem_desc", row.get("elem_desc", ""))
        samples.append({
            "image": row["image"],
            "instruction": instruction,
            "bbox_norm": _norm01_to_1000(bbox),
        })
    return samples


def load_vwb_ag(cache_dir):
    """VWB Agent Grounding: HongxinLi/VWB-AG, box in normalized 0-1."""
    from datasets import load_dataset
    ds = load_dataset("HongxinLi/VWB-AG", split="test", cache_dir=cache_dir,
                      trust_remote_code=True)
    samples = []
    for row in ds:
        bbox = row.get("box", [])
        if len(bbox) < 4:
            continue
        instruction = row.get("detailed_elem_desc", row.get("elem_desc", ""))
        samples.append({
            "image": row["image"],
            "instruction": instruction,
            "bbox_norm": _norm01_to_1000(bbox),
        })
    return samples


BENCHMARK_LOADERS = {
    "screenspot": ("ScreenSpot", load_screenspot),
    "screenspot_v2": ("ScreenSpot-v2", load_screenspot_v2),
    "funcpred": ("FuncPred", load_funcpred),
    "motif": ("MoTIF", load_motif),
    "refexp": ("Refexp", load_refexp),
    "vwb_eg": ("VWB-EG", load_vwb_eg),
    "vwb_ag": ("VWB-AG", load_vwb_ag),
}


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def evaluate_benchmark(model, samples, name, prompt_template=GROUNDING_PROMPT,
                       verbose=False, system_prompt=None):
    """Evaluate model on a list of grounding samples. Returns accuracy dict."""
    correct = 0
    total = 0

    for i, sample in enumerate(tqdm(samples, desc=f"[{name}]")):
        image = sample["image"]
        instruction = sample["instruction"]
        bbox_norm = sample["bbox_norm"]

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        image_size = image.size

        prompt_text = prompt_template.format(instruction=instruction)
        response = model.predict(image, prompt_text, system_prompt=system_prompt)
        coords = parse_click_coords(response, image_size=image_size)

        hit = False
        if coords:
            hit = point_in_bbox(coords[0], coords[1], bbox_norm)

        if hit:
            correct += 1
        elif verbose:
            pred_str = f"({coords[0]},{coords[1]})" if coords else "NO_COORDS"
            print(f"  [{name}] #{i}: MISS pred={pred_str} bbox={bbox_norm} "
                  f"resp={response[:80]}")
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------

def _worker(gpu_id, model_path, lora_path, max_pixels, benchmark_chunks,
            result_queue, prompt_template, verbose, system_prompt=None):
    """Worker process for multi-GPU evaluation."""
    device = f"cuda:{gpu_id}"
    model = ModelWrapper(model_path, lora_path, device, max_pixels)

    for bench_key, bench_name, chunk in benchmark_chunks:
        result = evaluate_benchmark(model, chunk, f"{bench_name}:GPU{gpu_id}",
                                    prompt_template, verbose, system_prompt)
        result_queue.put((bench_key, result))


def evaluate_multi_gpu(model_path, lora_path, max_pixels, all_benchmark_data,
                       num_gpus, prompt_template, verbose, system_prompt=None):
    """Distribute evaluation across GPUs and merge results."""
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()

    # Distribute each benchmark's samples across GPUs
    gpu_work = {i: [] for i in range(num_gpus)}
    for bench_key, (bench_name, samples) in all_benchmark_data.items():
        chunk_size = (len(samples) + num_gpus - 1) // num_gpus
        for gpu_id in range(num_gpus):
            start = gpu_id * chunk_size
            end = min(start + chunk_size, len(samples))
            if start < end:
                gpu_work[gpu_id].append((bench_key, bench_name, samples[start:end]))

    processes = []
    for gpu_id in range(num_gpus):
        if not gpu_work[gpu_id]:
            continue
        p = mp.Process(
            target=_worker,
            args=(gpu_id, model_path, lora_path, max_pixels, gpu_work[gpu_id],
                  result_queue, prompt_template, verbose, system_prompt),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge results by benchmark
    merged = defaultdict(lambda: {"correct": 0, "total": 0})
    while not result_queue.empty():
        bench_key, result = result_queue.get()
        merged[bench_key]["correct"] += result["correct"]
        merged[bench_key]["total"] += result["total"]

    final = {}
    for bench_key in merged:
        d = merged[bench_key]
        d["accuracy"] = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        final[bench_key] = d
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-world GUI benchmark evaluation")
    parser.add_argument("--model_path", required=True, help="Model checkpoint path")
    parser.add_argument("--lora_path", default=None, help="Optional LoRA adapter path")
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=list(BENCHMARK_LOADERS.keys()),
        choices=list(BENCHMARK_LOADERS.keys()),
        help="Benchmarks to evaluate",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="Max pixels for image processing (None = model default)")
    parser.add_argument("--output_file", default=None, help="Save results as JSON")
    parser.add_argument("--cache_dir", default="/home1/irteam/.cache/huggingface/datasets")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--base_model", action="store_true",
                        help="Use Qwen2.5-VL native bbox_2d prompt format instead of GE-Lab format")
    parser.add_argument("--use_system_prompt", action="store_true",
                        help="Pass the real-world system prompt so the model outputs click(start_box=...) format")
    args = parser.parse_args()

    prompt_template = GROUNDING_PROMPT_BASE if args.base_model else GROUNDING_PROMPT
    system_prompt = SYSTEM_PROMPT if args.use_system_prompt else None

    print("=" * 60)
    print("Real-World GUI Benchmark Evaluation (Table 5)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Prompt mode: {'base_model (bbox_2d)' if args.base_model else 'gelab (0-1000 click)'}")
    print(f"System prompt: {'yes' if system_prompt else 'no'}")
    print()

    # Load all requested benchmarks
    all_data = {}
    for key in args.benchmarks:
        bench_name, loader = BENCHMARK_LOADERS[key]
        print(f"Loading {bench_name}...")
        try:
            samples = loader(args.cache_dir)
            all_data[key] = (bench_name, samples)
            print(f"  {bench_name}: {len(samples)} samples loaded")
        except Exception as e:
            print(f"  {bench_name}: FAILED to load - {e}")
            print(f"  Skipping {bench_name}.")

    if not all_data:
        print("No benchmarks loaded. Exiting.")
        sys.exit(1)

    # Evaluate
    results = {}
    if args.num_gpus > 1:
        results = evaluate_multi_gpu(
            args.model_path, args.lora_path, args.max_pixels, all_data,
            args.num_gpus, prompt_template, args.verbose, system_prompt,
        )
    else:
        model = ModelWrapper(args.model_path, args.lora_path, "cuda:0", args.max_pixels)
        for key, (bench_name, samples) in all_data.items():
            result = evaluate_benchmark(model, samples, bench_name,
                                        prompt_template, args.verbose, system_prompt)
            results[key] = result

    # Print results table
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"{'Benchmark':<18} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 60)

    total_correct = 0
    total_samples = 0
    accuracies = []

    for key in args.benchmarks:
        if key not in results:
            continue
        r = results[key]
        bench_name = BENCHMARK_LOADERS[key][0]
        acc = r["accuracy"] * 100
        print(f"{bench_name:<18} {acc:>9.2f}% {r['correct']:>10} {r['total']:>10}")
        total_correct += r["correct"]
        total_samples += r["total"]
        accuracies.append(acc)

    if accuracies:
        avg = sum(accuracies) / len(accuracies)
        print("-" * 60)
        print(f"{'Average':<18} {avg:>9.2f}%")
        print(f"{'Weighted Avg':<18} {total_correct/total_samples*100:>9.2f}% "
              f"{total_correct:>10} {total_samples:>10}")

    # Save results
    if args.output_file:
        output = {
            "model_path": args.model_path,
            "benchmarks": {},
        }
        for key in args.benchmarks:
            if key in results:
                bench_name = BENCHMARK_LOADERS[key][0]
                output["benchmarks"][bench_name] = results[key]
        if accuracies:
            output["average_accuracy"] = sum(accuracies) / len(accuracies) / 100
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
