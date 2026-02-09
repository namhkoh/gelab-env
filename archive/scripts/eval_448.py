"""
Evaluation script for 448x448 SFT model
Evaluates both static (Edge/Path accuracy) and interactive (Pass@1/Pass@5) benchmarks
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
    # Try to match (x,y) pattern
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
    # Add tolerance
    return (x1 - tolerance <= x <= x2 + tolerance and 
            y1 - tolerance <= y <= y2 + tolerance)


class Evaluator:
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
    
    def evaluate_static(self, test_file, num_samples=None):
        """Evaluate static benchmark (Edge/Path accuracy)."""
        print(f"\nLoading test data from {test_file}...")
        with open(test_file) as f:
            data = json.load(f)
        
        if num_samples:
            data = data[:num_samples]
        
        print(f"Evaluating {len(data)} samples...")
        
        results = {
            "total": 0,
            "coord_correct": 0,
            "by_path_length": defaultdict(lambda: {"total": 0, "correct": 0})
        }
        
        for sample in tqdm(data, desc="Evaluating"):
            instruction = sample["messages"][0]["content"]
            image_path = sample["images"][0]
            ground_truth = sample["messages"][1]["content"]
            path_len = sample.get("path", 1)
            
            # Extract expected coordinates from ground truth
            gt_x, gt_y = parse_coordinates(ground_truth)
            if gt_x is None:
                continue
            
            # Get bbox from ground truth (use tolerance around expected coord)
            # For 0-1000 scale, ±50 is reasonable tolerance
            bbox = [gt_x - 50, gt_y - 50, gt_x + 50, gt_y + 50]
            
            response = self.predict(image_path, instruction)
            pred_x, pred_y = parse_coordinates(response)
            
            # Check if prediction is correct (within tolerance of ground truth)
            correct = is_coord_in_bbox((pred_x, pred_y), bbox, tolerance=0)
            
            results["total"] += 1
            if correct:
                results["coord_correct"] += 1
            
            results["by_path_length"][path_len]["total"] += 1
            if correct:
                results["by_path_length"][path_len]["correct"] += 1
        
        # Calculate metrics
        overall_acc = results["coord_correct"] / results["total"] * 100 if results["total"] > 0 else 0
        
        print("\n" + "="*60)
        print("STATIC EVALUATION RESULTS")
        print("="*60)
        print(f"\nOverall Accuracy: {overall_acc:.2f}%")
        print(f"  Correct: {results['coord_correct']}/{results['total']}")
        
        print("\nBy Path Length:")
        for path_len in sorted(results["by_path_length"].keys()):
            stats = results["by_path_length"][path_len]
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  Path {path_len}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate 448x448 SFT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_file", type=str, default="datas/448/test_448.json", help="Test data file")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()
    
    evaluator = Evaluator(args.model_path, args.device)
    evaluator.evaluate_static(args.test_file, args.num_samples)


if __name__ == "__main__":
    main()
