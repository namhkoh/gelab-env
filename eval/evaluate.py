"""
Unified GE-Lab Evaluation Script
=================================
Supports static (ID/OOD Edge/Path) and interactive (Pass@1, Pass@5) evaluation.
Follows the paper methodology: strict bbox containment, consistent system prompt,
Path@1-7 breakdown, and configurable multi-attempt interactive episodes.

Usage:
    # Static evaluation only
    python eval/evaluate.py --model_path <path> --mode static --env_dir datas/448_retrain

    # Interactive evaluation only
    python eval/evaluate.py --model_path <path> --mode interactive --env_dir datas/448_retrain

    # Full evaluation (static + interactive)
    python eval/evaluate.py --model_path <path> --mode all --env_dir datas/448_retrain

    # With LoRA adapter
    python eval/evaluate.py --model_path Qwen/Qwen2.5-VL-7B-Instruct --lora_path <adapter_dir> ...
"""

import argparse
import json
import os
import re
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANVAS_WIDTH = 448
CANVAS_HEIGHT = 448
NORM_SIZE = 1000
MAX_INTERACTIVE_STEPS = 12
DEFAULT_NUM_TASKS = 2162
DEFAULT_NUM_ATTEMPTS = 5
MAX_NEW_TOKENS = 256
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "You are a GUI Navigation Agent. Navigate to the target page by clicking icons.\n"
    "\n"
    "Input format:\n"
    "- Instruction: from <source> to <target>. History: <previous_steps>\n"
    "- Current screen image\n"
    "\n"
    "Output format:\n"
    "Explain: click <icon_name> icon on <page>.\tAction: click(start_box='<|box_start|>(x,y)<|box_end|>')\n"
    "OR\n"
    "Explain: this is target page.\tAction: complete\n"
    "\n"
    "Coordinates use (0,0) top-left to (1000,1000) bottom-right system."
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class GELabEnvironment:
    """Simulates the GE-Lab tree-structured UI environment."""

    def __init__(self, env_dir: str):
        self.env_dir = env_dir
        ui_path = os.path.join(env_dir, "ui_structure.json")
        layer_path = os.path.join(env_dir, "ui_structure_layer.json")
        self.pages_dir = os.path.join(env_dir, "pages")

        with open(ui_path, "r") as f:
            ui_data = json.load(f)
        self.pages = ui_data["pages"]

        with open(layer_path, "r") as f:
            self.layer_data = json.load(f)

        self.nav_graph = self._build_nav_graph()
        self.subtree_map = self._build_subtree_map()

    def _build_nav_graph(self):
        """Build page -> {target_page: (icon_name, normalized_bbox)} mapping."""
        graph = {}
        for page_id, info in self.pages.items():
            targets = {}
            for t in info.get("transitions", []):
                bbox = t.get("icon_bbox", t.get("bbox", []))
                norm_bbox = self._normalize_bbox(bbox)
                targets[t["target_page"]] = (t["action"], norm_bbox)
            graph[page_id] = targets
        return graph

    def _build_subtree_map(self):
        """Map each page to its subtree index."""
        mapping = {"page_0": -1}
        root = self.layer_data.get("root", {})
        subnodes = root.get("subnodes", [])
        for idx, child_data in enumerate(subnodes):
            page_id = child_data.get("image", "").replace(".png", "")
            self._traverse_subtree(page_id, child_data, idx, mapping)
        return mapping

    def _traverse_subtree(self, page_id, node_data, subtree_idx, mapping):
        mapping[page_id] = subtree_idx
        for child_data in node_data.get("subnodes", []):
            child_page = child_data.get("image", "").replace(".png", "")
            self._traverse_subtree(child_page, child_data, subtree_idx, mapping)

    @staticmethod
    def _normalize_bbox(bbox):
        """Convert pixel bbox [x1,y1,x2,y2] to 0-1000 normalized scale."""
        if len(bbox) != 4:
            return [0, 0, 0, 0]
        return [
            int(bbox[0] / CANVAS_WIDTH * NORM_SIZE),
            int(bbox[1] / CANVAS_HEIGHT * NORM_SIZE),
            int(bbox[2] / CANVAS_WIDTH * NORM_SIZE),
            int(bbox[3] / CANVAS_HEIGHT * NORM_SIZE),
        ]

    def get_page_image_path(self, page_id: str) -> str:
        return os.path.join(self.pages_dir, f"{page_id}.png")

    def get_pages_by_subtree(self, subtree_idx: int):
        return [p for p, s in self.subtree_map.items() if s == subtree_idx]

    def click(self, page_id: str, x: int, y: int):
        """Process a click at normalized (x,y) on page_id.
        Returns (new_page_id, clicked_icon_name) or (None, None) if miss."""
        page_info = self.pages.get(page_id)
        if not page_info:
            return None, None
        for t in page_info.get("transitions", []):
            bbox = t.get("icon_bbox", t.get("bbox", []))
            nb = self._normalize_bbox(bbox)
            if nb[0] <= x <= nb[2] and nb[1] <= y <= nb[3]:
                return t["target_page"], t["action"]
        return None, None

    def shortest_path_length(self, start: str, end: str) -> int:
        """BFS shortest path length between two pages."""
        if start == end:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.nav_graph.get(current, {}):
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return -1


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class ModelWrapper:
    """Wraps Qwen2.5-VL model loading and inference."""

    def __init__(self, model_path: str, lora_path: str = None, device: str = "cuda:0"):
        self.device = device
        print(f"Loading model from {model_path} on {device}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, max_pixels=CANVAS_WIDTH * CANVAS_HEIGHT,
        )

        if lora_path:
            print(f"Loading LoRA adapter from {lora_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()

    def predict(self, image_path: str, text: str, do_sample: bool = False,
                temperature: float = 0.7, top_p: float = 0.9) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ]},
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text_input], images=[image], return_tensors="pt", padding=True,
        ).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=do_sample,
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        return response


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------
COORD_PATTERN_FULL = re.compile(
    r"click\(start_box='<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>'\)"
)
COORD_PATTERN_SIMPLE = re.compile(r"\((\d+),\s*(\d+)\)")
ICON_NAME_PATTERN = re.compile(r"click\s+(.*?)\s+icon")


def parse_action(response: str):
    """Parse model response into (action_type, data).
    Returns:
        ('click', (x, y))  -- for click actions
        ('complete', None)  -- for complete actions
        ('invalid', None)   -- for unparseable responses
    """
    if "complete" in response.lower():
        return "complete", None
    m = COORD_PATTERN_FULL.search(response)
    if not m:
        m = COORD_PATTERN_SIMPLE.search(response)
    if m:
        return "click", (int(m.group(1)), int(m.group(2)))
    return "invalid", None


def extract_icon_name(response: str):
    """Extract icon name from 'click <name> icon' pattern."""
    m = ICON_NAME_PATTERN.search(response)
    return m.group(1).strip() if m else None


def is_coord_in_bbox(x: int, y: int, bbox: list) -> bool:
    """Strict bounding box containment check. No tolerance."""
    if len(bbox) != 4:
        return False
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


# ---------------------------------------------------------------------------
# Static evaluation
# ---------------------------------------------------------------------------
def evaluate_static(model: ModelWrapper, test_file: str, label: str, verbose: bool = False):
    """Evaluate static (edge or path first-step) accuracy.
    Returns dict with overall accuracy and per-path-length breakdown."""
    with open(test_file, "r") as f:
        data = json.load(f)

    correct = 0
    total = 0
    by_length = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, sample in enumerate(data):
        msgs = sample.get("messages", [])
        if len(msgs) < 2:
            continue

        user_content = msgs[0]["content"]
        gt_response = msgs[1]["content"]
        image_list = sample.get("images", [])
        bbox_norm = sample.get("bbox_norm", None)
        path_length = sample.get("path_length", sample.get("path", 1))

        if not image_list:
            continue
        image_path = image_list[0]
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.getcwd(), image_path)

        text = user_content.replace("<image>", "").strip()
        response = model.predict(image_path, text)

        action_type, coord = parse_action(response)
        gt_action_type, gt_coord = parse_action(gt_response)

        is_correct = False
        if gt_action_type == "complete" and action_type == "complete":
            is_correct = True
        elif gt_action_type == "click" and action_type == "click" and coord:
            if bbox_norm and len(bbox_norm) == 4:
                is_correct = is_coord_in_bbox(coord[0], coord[1], bbox_norm)
            elif gt_coord:
                fallback_bbox = [
                    gt_coord[0] - 50, gt_coord[1] - 50,
                    gt_coord[0] + 50, gt_coord[1] + 50,
                ]
                is_correct = is_coord_in_bbox(coord[0], coord[1], fallback_bbox)

        if is_correct:
            correct += 1
            by_length[path_length]["correct"] += 1
        total += 1
        by_length[path_length]["total"] += 1

        if verbose and not is_correct:
            print(f"  [{label}] Sample {i}: WRONG | pred={response[:80]} | gt={gt_response[:80]}")

        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{len(data)} evaluated, running acc: {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0.0
    length_breakdown = {}
    for pl in sorted(by_length.keys()):
        d = by_length[pl]
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        length_breakdown[pl] = {"accuracy": acc, "correct": d["correct"], "total": d["total"]}

    return {"accuracy": accuracy, "correct": correct, "total": total, "by_length": length_breakdown}


# ---------------------------------------------------------------------------
# Interactive evaluation
# ---------------------------------------------------------------------------
def generate_test_tasks(env: GELabEnvironment, test_subtree: int,
                        num_tasks: int, seed: int = RANDOM_SEED):
    """Generate interactive test tasks from the reserved test subtree."""
    rng = random.Random(seed)
    test_pages = env.get_pages_by_subtree(test_subtree)
    candidate_pages = test_pages + ["page_0"]

    tasks = []
    attempts = 0
    max_attempts = num_tasks * 20

    target_per_length = num_tasks // 7
    length_counts = defaultdict(int)

    while len(tasks) < num_tasks and attempts < max_attempts:
        attempts += 1
        start = rng.choice(candidate_pages)
        end = rng.choice(candidate_pages)
        if start == end:
            continue

        path_len = env.shortest_path_length(start, end)
        if path_len < 1 or path_len > 7:
            continue

        if length_counts[path_len] >= target_per_length * 2:
            continue

        tasks.append({"start": start, "end": end, "path_length": path_len})
        length_counts[path_len] += 1

    rng.shuffle(tasks)
    return tasks[:num_tasks]


def run_episode(model: ModelWrapper, env: GELabEnvironment, start: str, end: str,
                max_steps: int = MAX_INTERACTIVE_STEPS, do_sample: bool = False,
                temperature: float = 0.7, top_p: float = 0.9):
    """Run a single interactive navigation episode.
    Returns (success: bool, num_steps: int, trajectory: list)."""
    current_page = start
    history = []
    trajectory = []

    for step in range(1, max_steps + 1):
        history_str = "; ".join(history) if history else "Null"
        text = f"Instruction: from {start} to {end}. History: {history_str}"
        image_path = env.get_page_image_path(current_page)

        response = model.predict(
            image_path, text, do_sample=do_sample,
            temperature=temperature, top_p=top_p,
        )
        action_type, coord = parse_action(response)
        icon_name = extract_icon_name(response)

        trajectory.append({
            "step": step, "page": current_page, "response": response,
            "action": action_type, "coord": coord, "icon": icon_name,
        })

        if action_type == "complete":
            return current_page == end, step, trajectory

        if action_type == "click" and coord:
            new_page, clicked_icon = env.click(current_page, coord[0], coord[1])
            if new_page:
                history.append(
                    f"step{step}: click {clicked_icon} icon on {current_page}"
                )
                current_page = new_page
            else:
                name_str = icon_name if icon_name else "unknown"
                history.append(
                    f"step{step}: click {name_str} icon on {current_page} (missed)"
                )
        else:
            history.append(f"step{step}: invalid action on {current_page}")

    return False, max_steps, trajectory


def evaluate_interactive(model: ModelWrapper, env: GELabEnvironment,
                         test_subtree: int, num_tasks: int, num_attempts: int,
                         verbose: bool = False, max_steps: int = MAX_INTERACTIVE_STEPS):
    """Run interactive evaluation with Pass@1 and Pass@k."""
    tasks = generate_test_tasks(env, test_subtree, num_tasks)
    print(f"Generated {len(tasks)} interactive tasks")

    length_dist = defaultdict(int)
    for t in tasks:
        length_dist[t["path_length"]] += 1
    print(f"Task distribution by path length: {dict(sorted(length_dist.items()))}")

    results_by_length = defaultdict(lambda: {"pass1": 0, "passK": 0, "total": 0})
    total_pass1 = 0
    total_passK = 0

    for i, task in enumerate(tasks):
        start, end = task["start"], task["end"]
        path_len = task["path_length"]
        attempt_results = []

        for attempt in range(num_attempts):
            do_sample = attempt > 0
            success, steps, traj = run_episode(
                model, env, start, end, max_steps=max_steps,
                do_sample=do_sample,
            )
            attempt_results.append(success)

            if verbose and attempt == 0:
                status = "OK" if success else "FAIL"
                print(f"  Task {i+1}: {start}->{end} (len={path_len}) {status} in {steps} steps")

        pass1 = attempt_results[0]
        passK = any(attempt_results)

        results_by_length[path_len]["total"] += 1
        if pass1:
            results_by_length[path_len]["pass1"] += 1
            total_pass1 += 1
        if passK:
            results_by_length[path_len]["passK"] += 1
            total_passK += 1

        if (i + 1) % 50 == 0:
            n = i + 1
            print(
                f"  [{n}/{len(tasks)}] "
                f"Pass@1={total_pass1/n:.4f} "
                f"Pass@{num_attempts}={total_passK/n:.4f}"
            )

    total = len(tasks)
    return {
        "pass1": total_pass1 / total if total else 0.0,
        f"pass{num_attempts}": total_passK / total if total else 0.0,
        "total_tasks": total,
        "by_length": {
            pl: {
                "pass1": d["pass1"] / d["total"] if d["total"] else 0.0,
                f"pass{num_attempts}": d["passK"] / d["total"] if d["total"] else 0.0,
                "total": d["total"],
            }
            for pl, d in sorted(results_by_length.items())
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_static_report(results: dict):
    print("\n" + "=" * 70)
    print("STATIC EVALUATION RESULTS")
    print("=" * 70)

    for label in ["id_edge", "ood_edge", "id_path", "ood_path"]:
        if label not in results:
            continue
        r = results[label]
        print(f"\n  {label.upper()}: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")
        if r.get("by_length"):
            for pl, d in sorted(r["by_length"].items()):
                print(f"    Path@{pl}: {d['accuracy']:.4f} ({d['correct']}/{d['total']})")

    id_edge = results.get("id_edge", {}).get("accuracy", 0)
    id_path = results.get("id_path", {}).get("accuracy", 0)
    ood_edge = results.get("ood_edge", {}).get("accuracy", 0)
    ood_path = results.get("ood_path", {}).get("accuracy", 0)

    id_overall = (id_edge + id_path) / 2 if "id_edge" in results and "id_path" in results else 0
    ood_overall = (ood_edge + ood_path) / 2 if "ood_edge" in results and "ood_path" in results else 0

    print(f"\n  ID Overall:  {id_overall:.4f}")
    print(f"  OOD Overall: {ood_overall:.4f}")


def print_interactive_report(results: dict, num_attempts: int):
    print("\n" + "=" * 70)
    print("INTERACTIVE EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n  Pass@1: {results['pass1']:.4f}")
    print(f"  Pass@{num_attempts}: {results.get(f'pass{num_attempts}', 0):.4f}")
    print(f"  Total tasks: {results['total_tasks']}")
    print(f"\n  By path length:")
    for pl, d in sorted(results.get("by_length", {}).items()):
        print(
            f"    Path@{pl}: Pass@1={d['pass1']:.4f}  "
            f"Pass@{num_attempts}={d.get(f'pass{num_attempts}', 0):.4f}  "
            f"(n={d['total']})"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified GE-Lab Evaluation")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--env_dir", default="datas/448_retrain",
                        help="Environment directory with ui_structure.json, pages/, etc.")
    parser.add_argument("--mode", choices=["static", "interactive", "all"], default="all",
                        help="Evaluation mode")
    parser.add_argument("--device", default="cuda:0", help="Device for inference")

    # Static eval options
    parser.add_argument("--test_dir", default=None,
                        help="Directory with test JSON files (defaults to env_dir)")
    parser.add_argument("--id_edge_file", default="test_id_edge.json")
    parser.add_argument("--id_path_file", default="test_id_path.json")
    parser.add_argument("--ood_edge_file", default="test_ood_edge.json")
    parser.add_argument("--ood_path_file", default="test_ood_path.json")

    # Interactive eval options
    parser.add_argument("--test_subtree", type=int, default=4,
                        help="Test subtree index (default: 4, the 5th subtree)")
    parser.add_argument("--num_tasks", type=int, default=DEFAULT_NUM_TASKS,
                        help=f"Number of interactive tasks (default: {DEFAULT_NUM_TASKS})")
    parser.add_argument("--num_attempts", type=int, default=DEFAULT_NUM_ATTEMPTS,
                        help=f"Attempts per task for Pass@k (default: {DEFAULT_NUM_ATTEMPTS})")
    parser.add_argument("--max_steps", type=int, default=MAX_INTERACTIVE_STEPS,
                        help=f"Max steps per episode (default: {MAX_INTERACTIVE_STEPS})")

    # Output
    parser.add_argument("--output_file", default=None, help="Save results JSON to this path")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Limit number of static samples (0=all)")

    args = parser.parse_args()
    test_dir = args.test_dir or args.env_dir

    print("=" * 70)
    print("GE-Lab Unified Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    if args.lora_path:
        print(f"LoRA:  {args.lora_path}")
    print(f"Env:   {args.env_dir}")
    print(f"Mode:  {args.mode}")
    print(f"Device: {args.device}")
    print()

    model = ModelWrapper(args.model_path, args.lora_path, args.device)
    all_results = {}

    # --- Static evaluation ---
    if args.mode in ("static", "all"):
        print("\n--- Static Evaluation ---")
        for label, filename in [
            ("id_edge", args.id_edge_file),
            ("id_path", args.id_path_file),
            ("ood_edge", args.ood_edge_file),
            ("ood_path", args.ood_path_file),
        ]:
            filepath = os.path.join(test_dir, filename)
            if not os.path.exists(filepath):
                print(f"  Skipping {label}: {filepath} not found")
                continue
            print(f"\n  Evaluating {label} from {filepath}")

            if args.num_samples > 0:
                with open(filepath) as f:
                    data = json.load(f)
                tmp_path = filepath + ".subset.json"
                with open(tmp_path, "w") as f:
                    json.dump(data[:args.num_samples], f)
                result = evaluate_static(model, tmp_path, label, args.verbose)
                os.remove(tmp_path)
            else:
                result = evaluate_static(model, filepath, label, args.verbose)
            all_results[label] = result

        print_static_report(all_results)

    # --- Interactive evaluation ---
    if args.mode in ("interactive", "all"):
        print("\n--- Interactive Evaluation ---")
        env = GELabEnvironment(args.env_dir)
        interactive_results = evaluate_interactive(
            model, env, args.test_subtree,
            args.num_tasks, args.num_attempts, args.verbose,
            max_steps=args.max_steps,
        )
        all_results["interactive"] = interactive_results
        print_interactive_report(interactive_results, args.num_attempts)

    # --- Save results ---
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
