"""
Interactive evaluation matching Paper Table 2
Simulates actual navigation: start -> predict -> transition -> repeat until target
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
import json
import re
import argparse
from tqdm import tqdm
from collections import defaultdict
import os
import random


def parse_coordinates(text):
    """Extract coordinates from model output."""
    match = re.search(r'\((\d+),\s*(\d+)\)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def is_coord_in_bbox(x, y, bbox, tolerance=50):
    """Check if coordinate is within bbox."""
    if x is None or y is None:
        return False
    x1, y1, x2, y2 = bbox
    return (x1 - tolerance <= x <= x2 + tolerance and
            y1 - tolerance <= y <= y2 + tolerance)


class InteractiveEvaluator:
    def __init__(self, model_path, env_path, device="cuda:0", lora_path=None):
        print(f"Loading model from {model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        if lora_path:
            print(f"Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()
            print("LoRA adapter merged!")
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        self.device = device
        
        # Load environment
        self.env_path = env_path
        with open(os.path.join(env_path, "ui_structure.json")) as f:
            self.ui_structure = json.load(f)
        self.pages = self.ui_structure.get("pages", {})
        
        print(f"Loaded environment with {len(self.pages)} pages")

    def get_page_image_path(self, page_name):
        """Get the image path for a page."""
        return os.path.join(self.env_path, "pages", f"{page_name}.png")

    def predict(self, image_path, instruction, use_sampling=False):
        """Get model prediction."""
        if not os.path.exists(image_path):
            return ""
        
        image = Image.open(image_path).convert("RGB")
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction}
        ]}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if use_sampling:
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=150, 
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            else:
                outputs = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant")[-1].strip()

    def get_clicked_transition(self, page_name, pred_x, pred_y):
        """Find which icon was clicked and return target page."""
        if page_name not in self.pages:
            return None
        
        page_info = self.pages[page_name]
        for trans in page_info.get("transitions", []):
            bbox = trans.get("icon_bbox", [0, 0, 0, 0])
            # Normalize bbox to 0-1000 scale
            bbox_norm = [
                int(bbox[0] / 448 * 1000),
                int(bbox[1] / 448 * 1000),
                int(bbox[2] / 448 * 1000),
                int(bbox[3] / 448 * 1000)
            ]
            if is_coord_in_bbox(pred_x, pred_y, bbox_norm):
                return trans.get("target_page")
        return None

    def navigate(self, start_page, target_page, max_steps=10, history=None, use_sampling=False):
        """
        Attempt to navigate from start to target.
        Returns (success, steps_taken, trajectory)
        """
        current_page = start_page
        trajectory = [current_page]
        history_str = "Null" if history is None else history
        
        for step in range(max_steps):
            if current_page == target_page:
                return True, step, trajectory
            
            # Build instruction
            instruction = f"<image>Instruction: from {start_page} to {target_page}. History: {history_str}"
            
            # Get prediction
            image_path = self.get_page_image_path(current_page)
            response = self.predict(image_path, instruction, use_sampling=use_sampling)
            
            # Check for "complete" action
            if "complete" in response.lower():
                return current_page == target_page, step + 1, trajectory
            
            # Parse coordinates
            pred_x, pred_y = parse_coordinates(response)
            if pred_x is None:
                return False, step + 1, trajectory
            
            # Find clicked transition
            next_page = self.get_clicked_transition(current_page, pred_x, pred_y)
            if next_page is None:
                return False, step + 1, trajectory
            
            # Update history for next step
            # Extract action from response
            action_match = re.search(r'click (\S+) icon', response, re.IGNORECASE)
            if action_match:
                icon_name = action_match.group(1)
                if history_str == "Null":
                    history_str = f"step1: click {icon_name} icon on {current_page}"
                else:
                    step_num = len(trajectory)
                    history_str += f", step{step_num}: click {icon_name} icon on {current_page}"
            
            current_page = next_page
            trajectory.append(current_page)
        
        return current_page == target_page, max_steps, trajectory

    def evaluate_interactive(self, tasks, num_attempts=5):
        """
        Evaluate on interactive navigation tasks.
        Returns Pass@1 and Pass@N success rates by path length.
        """
        results = {
            "total": 0,
            "pass_1": 0,
            "pass_n": 0,
            "num_attempts": num_attempts,
            "by_path_length": defaultdict(lambda: {"total": 0, "pass_1": 0, "pass_n": 0})
        }
        
        for task in tqdm(tasks, desc="Interactive"):
            start = task["source_page"]
            target = task["target_page"]
            path_length = task.get("path_length", len(task.get("full_path", [])) - 1)
            
            attempt_results = []
            
            for attempt in range(num_attempts):
                # First attempt: greedy, others: sampling
                use_sampling = (attempt > 0)
                success, steps, trajectory = self.navigate(
                    start, target, 
                    max_steps=path_length + 3,
                    use_sampling=use_sampling
                )
                attempt_results.append(success)
            
            results["total"] += 1
            results["by_path_length"][path_length]["total"] += 1
            
            # Pass@1: first attempt success
            if attempt_results[0]:
                results["pass_1"] += 1
                results["by_path_length"][path_length]["pass_1"] += 1
            
            # Pass@N: any attempt success
            if any(attempt_results):
                results["pass_n"] += 1
                results["by_path_length"][path_length]["pass_n"] += 1
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA adapter path (optional)")
    parser.add_argument("--env_path", type=str, default="data_engine/ui_environment_448/latest")
    parser.add_argument("--test_file", type=str, default="datas/448_retrain/test_ood_path.json")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_attempts", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    evaluator = InteractiveEvaluator(args.model_path, args.env_path, args.device, args.lora_path)
    
    # Load test tasks
    with open(args.test_file) as f:
        tasks = json.load(f)
    
    if args.num_samples:
        tasks = tasks[:args.num_samples]
    
    print(f"\nEvaluating {len(tasks)} interactive tasks with {args.num_attempts} attempts each...")
    
    results = evaluator.evaluate_interactive(tasks, num_attempts=args.num_attempts)
    
    # Print results
    print("\n" + "="*70)
    print("INTERACTIVE EVALUATION RESULTS (Paper Table 2 Style)")
    print("="*70)
    
    overall_pass1 = results["pass_1"] / results["total"] * 100 if results["total"] > 0 else 0
    overall_passn = results["pass_n"] / results["total"] * 100 if results["total"] > 0 else 0
    print(f"\nOverall Pass@1: {overall_pass1:.2f}%")
    print(f"Overall Pass@{args.num_attempts}: {overall_passn:.2f}%")
    
    # Paper reference values
    paper_sft_pass1 = {1: 99.71, 2: 51.16, 3: 19.55, 4: 8.52, 5: 3.13, 6: 2.15, 7: 0.31}
    paper_sft_pass5 = {1: 100.00, 2: 74.15, 3: 36.04, 4: 19.75, 5: 6.71, 6: 5.04, 7: 1.30}
    
    print("\n" + "-"*70)
    print(f"{'Path':<8} {'Pass@1':<12} {'Paper P@1':<12} {'Pass@5':<12} {'Paper P@5':<12}")
    print("-"*70)
    
    for pl in sorted(results["by_path_length"].keys()):
        stats = results["by_path_length"][pl]
        p1 = stats["pass_1"] / stats["total"] * 100 if stats["total"] > 0 else 0
        pn = stats["pass_n"] / stats["total"] * 100 if stats["total"] > 0 else 0
        paper_p1 = paper_sft_pass1.get(pl, "-")
        paper_p5 = paper_sft_pass5.get(pl, "-")
        print(f"Path@{pl:<3} {p1:>6.2f}%{'':<5} {paper_p1:>6}%{'':<5} {pn:>6.2f}%{'':<5} {paper_p5:>6}%")
    
    print("-"*70)
    print(f"{'Total':<8} {overall_pass1:>6.2f}%{'':<17} {overall_passn:>6.2f}%")
    print(f"({'n='}{results['total']})")


if __name__ == "__main__":
    main()
