"""
Evaluation script for ST-RL model with LoRA adapter
Evaluates on ID Edge, ID Path, OOD Edge, OOD Path test sets
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

# System prompt used during training (must match exactly)
SYSTEM_PROMPT = """You are a GUI Navigation Agent. Navigate to the target page by clicking icons.

Input format:
- Instruction: from <source> to <target>. History: <previous_steps>
- Current screen image

Output format:
Explain: click <icon_name> icon on <page>.\tAction: click(start_box='<|box_start|>(x,y)<|box_end|>')
OR
Explain: this is target page.\tAction: complete

Coordinates use (0,0) top-left to (1000,1000) bottom-right system."""


def parse_action(text):
    """Extract action type and coordinates from model output."""
    # Match click(start_box='<|box_start|>(x,y)<|box_end|>')
    click_match = re.search(r"click\(start_box='<\|box_start\|\>\((\d+),(\d+)\)<\|box_end\|\>'\)", text)
    if click_match:
        return "click", int(click_match.group(1)), int(click_match.group(2))
    
    # Fallback: try to match any (x,y) pattern
    coord_match = re.search(r'\((\d+),\s*(\d+)\)', text)
    if coord_match:
        return "click", int(coord_match.group(1)), int(coord_match.group(2))
    
    return None, None, None


def is_coord_in_bbox(pred_x, pred_y, bbox, tolerance=0):
    """Check if predicted coordinate is within bbox."""
    if pred_x is None or pred_y is None:
        return False
    x1, y1, x2, y2 = bbox
    return (x1 - tolerance <= pred_x <= x2 + tolerance and 
            y1 - tolerance <= pred_y <= y2 + tolerance)


class STRLEvaluator:
    def __init__(self, base_model_path, lora_path=None, device="cuda:0"):
        print(f"Loading base model from {base_model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        if lora_path:
            print(f"Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()
            print("LoRA adapter merged!")
        
        self.processor = AutoProcessor.from_pretrained(base_model_path)
        self.model.eval()
        self.device = device
        print("Model ready for evaluation!")
    
    def predict(self, image_path, instruction):
        """Get model prediction for a single sample."""
        if not os.path.exists(image_path):
            # Try relative path from workspace
            alt_path = os.path.join("/ext_hdd2/nhkoh/gelab-env", image_path)
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                print(f"Warning: Image not found: {image_path}")
                return ""
        
        image = Image.open(image_path).convert("RGB")
        
        # Include system prompt as used during training
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]}
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        return response
    
    def evaluate_test_set(self, test_file, num_samples=None, tolerance=50, verbose=False):
        """Evaluate on a test set."""
        print(f"\nLoading test data from {test_file}...")
        with open(test_file) as f:
            data = json.load(f)
        
        if num_samples:
            data = data[:num_samples]
        
        print(f"Evaluating {len(data)} samples...")
        
        results = {
            "total": 0,
            "correct": 0,
            "by_path_length": defaultdict(lambda: {"total": 0, "correct": 0}),
            "predictions": []
        }
        
        for i, sample in enumerate(tqdm(data, desc="Evaluating")):
            # Extract instruction (user message content)
            instruction = sample["messages"][0]["content"]
            image_path = sample["images"][0]
            ground_truth = sample["messages"][1]["content"]
            
            # Get path length for path tests, default to 1 for edge tests
            path_len = sample.get("path_length", sample.get("path", 1))
            if path_len is None:
                path_len = 1
            
            # Get bbox from sample
            bbox = sample.get("bbox_norm", None)
            if bbox is None:
                # Try to extract from ground truth
                _, gt_x, gt_y = parse_action(ground_truth)
                if gt_x is not None:
                    bbox = [gt_x - 50, gt_y - 50, gt_x + 50, gt_y + 50]
                else:
                    continue
            
            # Get model prediction
            response = self.predict(image_path, instruction)
            action, pred_x, pred_y = parse_action(response)
            
            # Check if prediction is correct
            correct = is_coord_in_bbox(pred_x, pred_y, bbox, tolerance=tolerance)
            
            # Debug output for first few samples
            if verbose and i < 5:
                print(f"\n--- Sample {i} ---")
                print(f"Instruction: {instruction}")
                print(f"Ground truth: {ground_truth}")
                print(f"Prediction: {response}")
                print(f"Parsed: ({pred_x}, {pred_y}), bbox: {bbox}, correct: {correct}")
            
            results["total"] += 1
            if correct:
                results["correct"] += 1
            
            results["by_path_length"][path_len]["total"] += 1
            if correct:
                results["by_path_length"][path_len]["correct"] += 1
            
            results["predictions"].append({
                "instruction": instruction[:100] + "..." if len(instruction) > 100 else instruction,
                "ground_truth": ground_truth,
                "prediction": response,
                "correct": correct,
                "path_length": path_len
            })
        
        return results
    
    def print_results(self, results, test_name):
        """Print evaluation results."""
        print("\n" + "="*60)
        print(f"RESULTS: {test_name}")
        print("="*60)
        
        overall_acc = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
        print(f"\nOverall Accuracy: {overall_acc:.2f}%")
        print(f"  Correct: {results['correct']}/{results['total']}")
        
        if len(results["by_path_length"]) > 1:
            print("\nBy Path Length:")
            for path_len in sorted(results["by_path_length"].keys()):
                stats = results["by_path_length"][path_len]
                acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"  Path {path_len}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
        
        return overall_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate ST-RL model")
    parser.add_argument("--base_model", type=str, default="/ext_hdd2/nhkoh/models/gelab-sft-448",
                        help="Path to base model")
    parser.add_argument("--lora_path", type=str, 
                        default="/ext_hdd2/nhkoh/gelab-env/checkpoint/gui_exp/st_rl_448_lora/v0-20260131_151021/checkpoint-576",
                        help="Path to LoRA checkpoint")
    parser.add_argument("--test_dir", type=str, default="/ext_hdd2/nhkoh/gelab-env/datas/448_paper",
                        help="Directory containing test files")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples per test set (None for all)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--tolerance", type=int, default=50, 
                        help="Coordinate tolerance for bbox matching")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for detailed results (JSON)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output for first few samples")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = STRLEvaluator(args.base_model, args.lora_path, args.device)
    
    # Define test sets
    test_sets = {
        "ID Edge": os.path.join(args.test_dir, "test_id_edge_fixed.json"),
        "ID Path": os.path.join(args.test_dir, "test_id_path_fixed.json"),
        "OOD Edge": os.path.join(args.test_dir, "test_ood_edge_fixed.json"),
        "OOD Path": os.path.join(args.test_dir, "test_ood_path_fixed.json"),
    }
    
    all_results = {}
    summary = {}
    
    for test_name, test_file in test_sets.items():
        if not os.path.exists(test_file):
            print(f"Warning: Test file not found: {test_file}")
            continue
        
        results = evaluator.evaluate_test_set(test_file, args.num_samples, args.tolerance, args.verbose)
        acc = evaluator.print_results(results, test_name)
        all_results[test_name] = results
        summary[test_name] = acc
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for test_name, acc in summary.items():
        print(f"  {test_name}: {acc:.2f}%")
    
    # Save results if output file specified
    if args.output_file:
        # Remove non-serializable items
        for test_name in all_results:
            all_results[test_name]["by_path_length"] = dict(all_results[test_name]["by_path_length"])
        
        output_data = {
            "summary": summary,
            "detailed_results": all_results
        }
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to {args.output_file}")


if __name__ == "__main__":
    main()
