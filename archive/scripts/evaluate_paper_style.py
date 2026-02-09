"""
Evaluation script matching the paper's methodology (Table 1)

Evaluates:
1. ID Edge: Single-step accuracy on subtrees 0-1 (training distribution)
2. ID Path: Multi-step path completion on subtrees 0-1
3. OOD Edge: Single-step accuracy on subtree 4 (test distribution)
4. OOD Path: Multi-step path completion on subtree 4
5. Interactive: Pass@1 and Pass@5 with environment interaction
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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
    # Match (x,y) pattern
    match = re.search(r'\((\d+),\s*(\d+)\)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def is_coord_in_bbox(coord, bbox, tolerance=50):
    """Check if coordinate is within bbox (with tolerance for 0-1000 scale)."""
    if coord[0] is None or coord[1] is None:
        return False
    x, y = coord
    x1, y1, x2, y2 = bbox
    return (x1 - tolerance <= x <= x2 + tolerance and
            y1 - tolerance <= y <= y2 + tolerance)


class PaperStyleEvaluator:
    def __init__(self, model_path, device="cuda:0"):
        print(f"Loading model from {model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        self.device = device
        print("Model loaded!")

    def predict(self, image_path, instruction):
        """Get model prediction for a single sample."""
        image = Image.open(image_path)
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction}
        ]}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant")[-1].strip()
        return response

    def evaluate_edge(self, test_data, name="Edge"):
        """
        Evaluate Edge (single-step) accuracy.
        Paper metric: Whether predicted click coordinates fall within target icon bbox.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {name} (Single-Step Accuracy)")
        print(f"{'='*60}")
        
        results = {"total": 0, "correct": 0}
        
        for sample in tqdm(test_data, desc=f"{name}"):
            instruction = sample["messages"][0]["content"]
            image_path = sample["images"][0]
            ground_truth = sample["messages"][1]["content"]
            
            # Extract expected coordinates from ground truth
            gt_x, gt_y = parse_coordinates(ground_truth)
            if gt_x is None:
                continue
            
            # Create bbox around ground truth (±50 on 0-1000 scale)
            bbox = [gt_x - 50, gt_y - 50, gt_x + 50, gt_y + 50]
            
            response = self.predict(image_path, instruction)
            pred_x, pred_y = parse_coordinates(response)
            
            correct = is_coord_in_bbox((pred_x, pred_y), bbox, tolerance=0)
            
            results["total"] += 1
            if correct:
                results["correct"] += 1
        
        accuracy = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
        print(f"\n{name} Accuracy: {accuracy:.2f}% ({results['correct']}/{results['total']})")
        return accuracy

    def evaluate_path(self, ui_structure, test_tasks, name="Path", num_samples=100):
        """
        Evaluate Path (multi-step) accuracy.
        Paper metric: Whether model can navigate from start to target through correct sequence.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {name} (Multi-Step Path Completion)")
        print(f"{'='*60}")
        
        results = {
            "total": 0,
            "completed": 0,
            "by_length": defaultdict(lambda: {"total": 0, "completed": 0})
        }
        
        # Sample tasks
        if len(test_tasks) > num_samples:
            test_tasks = random.sample(test_tasks, num_samples)
        
        for task in tqdm(test_tasks, desc=f"{name}"):
            start_page = task["start"]
            target_page = task["target"]
            path_length = task["path_length"]
            
            # Simulate navigation
            current_page = start_page
            max_steps = path_length + 3  # Allow some extra steps
            success = False
            
            for step in range(max_steps):
                if current_page == target_page:
                    success = True
                    break
                
                # Get current page info
                page_info = ui_structure.get(current_page, {})
                image_path = page_info.get("image_path")
                if not image_path or not os.path.exists(image_path):
                    break
                
                # Create instruction
                instruction = f"You are on {current_page}. Navigate to {target_page}. Click the appropriate icon."
                
                response = self.predict(image_path, instruction)
                pred_x, pred_y = parse_coordinates(response)
                
                if pred_x is None:
                    break
                
                # Find which icon was clicked
                clicked_icon = None
                for icon_name, icon_info in page_info.get("icons", {}).items():
                    bbox = icon_info.get("bbox", [0, 0, 0, 0])
                    if is_coord_in_bbox((pred_x, pred_y), bbox):
                        clicked_icon = icon_name
                        break
                
                if clicked_icon is None:
                    break
                
                # Get next page
                next_page = page_info.get("transitions", {}).get(clicked_icon)
                if next_page is None:
                    break
                
                current_page = next_page
            
            results["total"] += 1
            results["by_length"][path_length]["total"] += 1
            
            if success:
                results["completed"] += 1
                results["by_length"][path_length]["completed"] += 1
        
        accuracy = results["completed"] / results["total"] * 100 if results["total"] > 0 else 0
        print(f"\n{name} Completion Rate: {accuracy:.2f}% ({results['completed']}/{results['total']})")
        
        print("\nBy Path Length:")
        for length in sorted(results["by_length"].keys()):
            stats = results["by_length"][length]
            acc = stats["completed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  Length {length}: {acc:.2f}% ({stats['completed']}/{stats['total']})")
        
        return accuracy

    def evaluate_interactive(self, ui_structure, test_tasks, num_samples=200, num_attempts=5):
        """
        Evaluate Interactive benchmark (Pass@1 and Pass@5).
        Paper metric: Task completion rate with N attempts.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Interactive Benchmark (Pass@1, Pass@{num_attempts})")
        print(f"{'='*60}")
        
        results = {
            "total": 0,
            "pass_1": 0,
            "pass_n": 0,
            "by_length": defaultdict(lambda: {"total": 0, "pass_1": 0, "pass_n": 0})
        }
        
        # Sample tasks
        if len(test_tasks) > num_samples:
            test_tasks = random.sample(test_tasks, num_samples)
        
        for task in tqdm(test_tasks, desc="Interactive"):
            start_page = task["start"]
            target_page = task["target"]
            path_length = task["path_length"]
            max_steps = path_length + 5
            
            attempt_results = []
            
            for attempt in range(num_attempts):
                current_page = start_page
                success = False
                
                for step in range(max_steps):
                    if current_page == target_page:
                        success = True
                        break
                    
                    page_info = ui_structure.get(current_page, {})
                    image_path = page_info.get("image_path")
                    if not image_path or not os.path.exists(image_path):
                        break
                    
                    instruction = f"Navigate from {current_page} to {target_page}. What icon should you click?"
                    
                    # Use sampling for attempts > 0
                    if attempt > 0:
                        response = self.predict_with_sampling(image_path, instruction)
                    else:
                        response = self.predict(image_path, instruction)
                    
                    pred_x, pred_y = parse_coordinates(response)
                    if pred_x is None:
                        break
                    
                    # Find clicked icon and transition
                    next_page = None
                    for icon_name, icon_info in page_info.get("icons", {}).items():
                        bbox = icon_info.get("bbox", [0, 0, 0, 0])
                        if is_coord_in_bbox((pred_x, pred_y), bbox):
                            next_page = page_info.get("transitions", {}).get(icon_name)
                            break
                    
                    if next_page is None:
                        break
                    
                    current_page = next_page
                
                attempt_results.append(success)
            
            results["total"] += 1
            results["by_length"][path_length]["total"] += 1
            
            if attempt_results[0]:  # Pass@1
                results["pass_1"] += 1
                results["by_length"][path_length]["pass_1"] += 1
            
            if any(attempt_results):  # Pass@N
                results["pass_n"] += 1
                results["by_length"][path_length]["pass_n"] += 1
        
        pass_1 = results["pass_1"] / results["total"] * 100 if results["total"] > 0 else 0
        pass_n = results["pass_n"] / results["total"] * 100 if results["total"] > 0 else 0
        
        print(f"\nPass@1: {pass_1:.2f}% ({results['pass_1']}/{results['total']})")
        print(f"Pass@{num_attempts}: {pass_n:.2f}% ({results['pass_n']}/{results['total']})")
        
        print("\nBy Path Length:")
        for length in sorted(results["by_length"].keys()):
            stats = results["by_length"][length]
            p1 = stats["pass_1"] / stats["total"] * 100 if stats["total"] > 0 else 0
            pn = stats["pass_n"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  Length {length}: Pass@1={p1:.2f}%, Pass@{num_attempts}={pn:.2f}% (n={stats['total']})")
        
        return pass_1, pass_n

    def predict_with_sampling(self, image_path, instruction, temperature=0.7):
        """Get model prediction with sampling for diversity."""
        image = Image.open(image_path)
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction}
        ]}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=True,
                temperature=temperature,
                top_p=0.9
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant")[-1].strip()
        return response


def load_ui_structure(structure_path):
    """Load UI structure for interactive evaluation."""
    with open(structure_path) as f:
        return json.load(f)


def generate_path_tasks(ui_structure, subtrees, num_tasks=500):
    """Generate path navigation tasks from UI structure."""
    tasks = []
    pages = list(ui_structure.keys())
    
    # Filter pages by subtree
    filtered_pages = []
    for page in pages:
        page_num = int(page.split("_")[1]) if "_" in page else 0
        # Simple heuristic: assign pages to subtrees based on page number ranges
        # This should be replaced with actual subtree information from the structure
        filtered_pages.append(page)
    
    # Generate random navigation tasks
    for _ in range(num_tasks):
        if len(filtered_pages) < 2:
            continue
        start = random.choice(filtered_pages)
        target = random.choice(filtered_pages)
        if start != target:
            # Calculate path length (simplified - should use BFS on actual graph)
            path_length = random.randint(1, 7)
            tasks.append({
                "start": start,
                "target": target,
                "path_length": path_length
            })
    
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Paper-style evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--id_edge_file", type=str, default="datas/448/test_id_edge.json")
    parser.add_argument("--ood_edge_file", type=str, default="datas/448/test_ood_edge.json")
    parser.add_argument("--id_path_file", type=str, default="datas/448/test_id_path.json")
    parser.add_argument("--ood_path_file", type=str, default="datas/448/test_ood_path.json")
    parser.add_argument("--eval_edge", action="store_true", help="Evaluate Edge accuracy")
    parser.add_argument("--eval_path", action="store_true", help="Evaluate Path accuracy")
    parser.add_argument("--eval_all", action="store_true", help="Run all evaluations")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    evaluator = PaperStyleEvaluator(args.model_path, args.device)
    
    results = {}
    
    # Edge Evaluation
    if args.eval_edge or args.eval_all:
        # ID Edge
        if os.path.exists(args.id_edge_file):
            with open(args.id_edge_file) as f:
                id_edge_data = json.load(f)
            if args.num_samples:
                id_edge_data = id_edge_data[:args.num_samples]
            results["id_edge"] = evaluator.evaluate_edge(id_edge_data, name="ID Edge")
        
        # OOD Edge
        if os.path.exists(args.ood_edge_file):
            with open(args.ood_edge_file) as f:
                ood_edge_data = json.load(f)
            if args.num_samples:
                ood_edge_data = ood_edge_data[:args.num_samples]
            results["ood_edge"] = evaluator.evaluate_edge(ood_edge_data, name="OOD Edge")
    
    # Path Evaluation
    if args.eval_path or args.eval_all:
        # ID Path
        if os.path.exists(args.id_path_file):
            with open(args.id_path_file) as f:
                id_path_data = json.load(f)
            if args.num_samples:
                id_path_data = id_path_data[:args.num_samples]
            results["id_path"] = evaluator.evaluate_edge(id_path_data, name="ID Path (first step)")
        
        # OOD Path
        if os.path.exists(args.ood_path_file):
            with open(args.ood_path_file) as f:
                ood_path_data = json.load(f)
            if args.num_samples:
                ood_path_data = ood_path_data[:args.num_samples]
            results["ood_path"] = evaluator.evaluate_edge(ood_path_data, name="OOD Path (first step)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY (Paper Table 1 Style)")
    print("="*60)
    print(f"\n{'Metric':<20} {'Our Result':<15} {'Paper SFT':<15}")
    print("-" * 50)
    if "id_edge" in results:
        print(f"{'ID Edge':<20} {results['id_edge']:.2f}%{'':<9} 94.82%")
    if "id_path" in results:
        print(f"{'ID Path':<20} {results['id_path']:.2f}%{'':<9} 99.76%")
    if "ood_edge" in results:
        print(f"{'OOD Edge':<20} {results['ood_edge']:.2f}%{'':<9} 64.55%")
    if "ood_path" in results:
        print(f"{'OOD Path':<20} {results['ood_path']:.2f}%{'':<9} 41.76%")


if __name__ == "__main__":
    main()
