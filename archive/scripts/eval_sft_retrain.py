"""
Evaluation script for retrained SFT model
Tests on ID Edge, ID Path, OOD Edge, OOD Path
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


class SFTEvaluator:
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
        # Handle relative paths
        if not os.path.isabs(image_path):
            image_path = os.path.join("/ext_hdd2/nhkoh/gelab-env", image_path)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return ""
        
        image = Image.open(image_path).convert("RGB")
        
        # No system prompt - matches training
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction}
        ]}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant")[-1].strip()
        return response

    def evaluate_test_set(self, test_file, name="Test", verbose=False):
        """Evaluate on a test set."""
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        
        with open(test_file) as f:
            data = json.load(f)
        
        print(f"Samples: {len(data)}")
        
        results = {
            "total": 0,
            "correct": 0,
            "by_path_length": defaultdict(lambda: {"total": 0, "correct": 0})
        }
        
        for i, sample in enumerate(tqdm(data, desc=name)):
            instruction = sample["messages"][0]["content"]
            image_path = sample["images"][0]
            ground_truth = sample["messages"][1]["content"]
            bbox = sample.get("bbox_norm", None)
            
            # Get path length
            path_len = sample.get("path_length", 1)
            
            # If no bbox, extract from ground truth
            if bbox is None:
                gt_x, gt_y = parse_coordinates(ground_truth)
                if gt_x is not None:
                    bbox = [gt_x - 50, gt_y - 50, gt_x + 50, gt_y + 50]
                else:
                    continue
            
            response = self.predict(image_path, instruction)
            pred_x, pred_y = parse_coordinates(response)
            
            correct = is_coord_in_bbox((pred_x, pred_y), bbox)
            
            if verbose and i < 3:
                print(f"\n[{i}] {sample.get('source_page', '?')} -> {sample.get('target_page', '?')}")
                print(f"    Instruction: {instruction[:60]}...")
                print(f"    Expected: {ground_truth[:60]}...")
                print(f"    Got: {response[:60]}...")
                print(f"    Pred: ({pred_x}, {pred_y}), BBox: {bbox}, Correct: {correct}")
            
            results["total"] += 1
            if correct:
                results["correct"] += 1
            
            results["by_path_length"][path_len]["total"] += 1
            if correct:
                results["by_path_length"][path_len]["correct"] += 1
        
        accuracy = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
        print(f"\n{name} Accuracy: {accuracy:.2f}% ({results['correct']}/{results['total']})")
        
        if len(results["by_path_length"]) > 1:
            print("\nBy Path Length:")
            for pl in sorted(results["by_path_length"].keys()):
                stats = results["by_path_length"][pl]
                acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"  Path {pl}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
        
        return accuracy, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default="datas/448_retrain")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    evaluator = SFTEvaluator(args.model_path, args.device)
    
    test_sets = {
        "ID Edge": os.path.join(args.test_dir, "test_id_edge.json"),
        "ID Path": os.path.join(args.test_dir, "test_id_path.json"),
        "OOD Edge": os.path.join(args.test_dir, "test_ood_edge.json"),
        "OOD Path": os.path.join(args.test_dir, "test_ood_path.json"),
    }
    
    results = {}
    
    for name, test_file in test_sets.items():
        if os.path.exists(test_file):
            acc, _ = evaluator.evaluate_test_set(test_file, name, args.verbose)
            results[name] = acc
        else:
            print(f"Warning: {test_file} not found")
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\n{'Test Set':<15} {'Accuracy':<12} {'Paper SFT':<12}")
    print("-" * 40)
    paper_baselines = {
        "ID Edge": 94.82,
        "ID Path": 99.76,
        "OOD Edge": 64.55,
        "OOD Path": 41.76
    }
    for name, acc in results.items():
        paper = paper_baselines.get(name, "-")
        print(f"{name:<15} {acc:.2f}%{'':<6} {paper}%")


if __name__ == "__main__":
    main()
