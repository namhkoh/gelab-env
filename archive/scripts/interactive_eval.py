"""
Interactive Environment Evaluation for GE-Lab

This script evaluates the SFT model in an interactive setting where:
1. Model receives current screen + task instruction + history
2. Model outputs an action (click coordinates or complete)
3. Environment executes action and transitions to new page
4. Repeat until "complete" or step limit
5. Success = reaching the correct target page

Metrics: Pass@1, Pass@5 by path length (matching paper Table 2)
"""

import json
import os
import re
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import torch
from PIL import Image


# System prompt matching ST-RL training (gui_scripts/st_rl_448.sh)
SYSTEM_PROMPT = """You are a GUI Navigation Agent. Navigate to the target page by clicking icons.

Input format:
- Instruction: from <source> to <target>. History: <previous_steps>
- Current screen image

Output format:
Explain: click <icon_name> icon on <page>.\tAction: click(start_box='<|box_start|>(x,y)<|box_end|>')
OR
Explain: this is target page.\tAction: complete

Coordinates use (0,0) top-left to (1000,1000) bottom-right system."""


class GELabEnvironment:
    """Interactive GE-Lab environment for evaluation."""
    
    def __init__(self, ui_structure_path: str, ui_layer_path: str, pages_dir: str):
        # Load UI structure
        with open(ui_structure_path, 'r') as f:
            data = json.load(f)
        self.pages = data['pages']
        self.pages_dir = pages_dir
        
        # Build page-to-subtree mapping
        with open(ui_layer_path, 'r') as f:
            self.layer_data = json.load(f)
        self.page_to_subtree = self._build_subtree_mapping()
        
        # Build navigation graph
        self.nav_graph = self._build_nav_graph()
        
        # Coordinate normalization - must match training data (448x448 canvas)
        self.CANVAS_WIDTH = 448
        self.CANVAS_HEIGHT = 448
        self.NORM_SIZE = 1000
        
    def _build_subtree_mapping(self) -> Dict[str, int]:
        """Map pages to subtrees."""
        mapping = {'page_0': -1}
        
        def collect(node, subtree_idx):
            img = node.get('image', '')
            if img:
                page_id = img.replace('.png', '')
                mapping[page_id] = subtree_idx
            for child in node.get('subnodes', []):
                collect(child, subtree_idx)
        
        root = self.layer_data.get('root', {})
        for i, child in enumerate(root.get('subnodes', [])):
            collect(child, i)
        
        return mapping
    
    def _build_nav_graph(self) -> Dict[str, Dict[str, Tuple[str, List[int]]]]:
        """Build navigation graph: page -> {target_page: (icon_name, bbox)}"""
        graph = {}
        for page_id, page_data in self.pages.items():
            graph[page_id] = {}
            for trans in page_data.get('transitions', []):
                target = trans['target_page']
                icon = trans['action']
                bbox = trans.get('icon_bbox', [0, 0, 0, 0])
                graph[page_id][target] = (icon, bbox)
        return graph
    
    def get_page_image_path(self, page_id: str) -> str:
        """Get path to page image."""
        return os.path.join(self.pages_dir, f"{page_id}.png")
    
    def get_clickable_icons(self, page_id: str) -> Dict[str, List[int]]:
        """Get all clickable icons on a page with their bboxes."""
        if page_id not in self.pages:
            return {}
        return {name: info['bbox'] for name, info in self.pages[page_id].get('layout', {}).items()}
    
    def normalize_bbox(self, bbox: List[int]) -> List[int]:
        """Normalize bbox to 0-1000 scale."""
        x1, y1, x2, y2 = bbox
        return [
            int(x1 / self.CANVAS_WIDTH * self.NORM_SIZE),
            int(y1 / self.CANVAS_HEIGHT * self.NORM_SIZE),
            int(x2 / self.CANVAS_WIDTH * self.NORM_SIZE),
            int(y2 / self.CANVAS_HEIGHT * self.NORM_SIZE)
        ]
    
    def click(self, page_id: str, x: int, y: int) -> Tuple[str, bool]:
        """
        Execute click action at normalized coordinates (x, y).
        Returns: (new_page_id, is_valid_click)
        """
        if page_id not in self.pages:
            return page_id, False
        
        # Check which icon was clicked
        layout = self.pages[page_id].get('layout', {})
        for icon_name, icon_info in layout.items():
            bbox = icon_info['bbox']
            norm_bbox = self.normalize_bbox(bbox)
            x_min, y_min, x_max, y_max = norm_bbox
            
            if x_min <= x <= x_max and y_min <= y <= y_max:
                # Found clicked icon - get target page
                for trans in self.pages[page_id].get('transitions', []):
                    if trans['action'] == icon_name:
                        return trans['target_page'], True
        
        # No valid icon clicked - stay on same page
        return page_id, False
    
    def get_shortest_path_length(self, start: str, end: str) -> int:
        """Get shortest path length between two pages using BFS."""
        if start == end:
            return 0
        
        from collections import deque
        visited = {start}
        queue = deque([(start, 0)])
        
        while queue:
            page, dist = queue.popleft()
            for target in self.nav_graph.get(page, {}).keys():
                if target == end:
                    return dist + 1
                if target not in visited:
                    visited.add(target)
                    queue.append((target, dist + 1))
        
        return -1  # No path found
    
    def get_pages_by_subtree(self, subtree: int) -> List[str]:
        """Get all pages in a subtree."""
        return [p for p, s in self.page_to_subtree.items() if s == subtree]


class InteractiveEvaluator:
    """Evaluator for interactive benchmark."""
    
    def __init__(self, model, processor, env: GELabEnvironment, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.env = env
        self.device = device
        self.max_steps = 15  # Maximum steps per task
        
    def parse_action(self, response: str) -> Tuple[str, Optional[Tuple[int, int]]]:
        """
        Parse model response to extract action.
        Returns: (action_type, coordinates or None)
        """
        response_lower = response.lower()
        
        # Check for complete action
        if 'complete' in response_lower:
            return 'complete', None
        
        # Extract coordinates
        coord_match = re.search(r'\((\d+),\s*(\d+)\)', response)
        if coord_match:
            x, y = int(coord_match.group(1)), int(coord_match.group(2))
            return 'click', (x, y)
        
        return 'invalid', None
    
    def build_prompt(self, task: str, current_page: str, history: List[str]) -> str:
        """Build prompt for the model."""
        history_str = " ".join(history) if history else "Null"
        return f"<image>Instruction: {task}. History: {history_str}"
    
    def _extract_icon_name(self, response: str) -> Optional[str]:
        """Extract icon name from model response."""
        match = re.search(r'click\s+(\S+)\s+icon', response, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _find_icon_at_coords(self, page_id: str, x: int, y: int) -> Optional[str]:
        """Reverse-lookup: find which icon was clicked at normalized (x, y)."""
        if page_id not in self.env.pages:
            return None
        layout = self.env.pages[page_id].get('layout', {})
        for icon_name, icon_info in layout.items():
            bbox = icon_info['bbox']
            norm_bbox = self.env.normalize_bbox(bbox)
            x_min, y_min, x_max, y_max = norm_bbox
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return icon_name
        return None

    def run_episode(self, start_page: str, target_page: str,
                    use_sampling: bool = False) -> Tuple[bool, int, List[str]]:
        """
        Run single episode.
        Returns: (success, steps_taken, trajectory)
        """
        current_page = start_page
        task = f"from {start_page} to {target_page}"
        history = []
        trajectory = [current_page]

        for step in range(self.max_steps):
            img_path = self.env.get_page_image_path(current_page)
            if not os.path.exists(img_path):
                return False, step + 1, trajectory

            prompt = self.build_prompt(task, current_page, history)
            response = self._get_model_response(img_path, prompt, use_sampling=use_sampling)
            action_type, coords = self.parse_action(response)

            if action_type == 'complete':
                return current_page == target_page, step + 1, trajectory

            elif action_type == 'click' and coords:
                x, y = coords
                # Extract icon name from response, fall back to env lookup
                icon_name = self._extract_icon_name(response)
                if icon_name is None:
                    icon_name = self._find_icon_at_coords(current_page, x, y)

                new_page, valid = self.env.click(current_page, x, y)

                if icon_name:
                    history.append(f"step{step+1}: click {icon_name} icon on {current_page}")
                else:
                    history.append(f"step{step+1}: click icon on {current_page}")

                if new_page != current_page:
                    trajectory.append(new_page)
                current_page = new_page

            else:
                history.append(f"step{step+1}: invalid action on {current_page}")

        return current_page == target_page, self.max_steps, trajectory
    
    def _get_model_response(self, img_path: str, prompt: str, use_sampling: bool = False) -> str:
        """Get model response for given image and prompt."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image = Image.open(img_path).convert('RGB')

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=256,
            pad_token_id=self.processor.tokenizer.pad_token_id
        )
        if use_sampling:
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return response


def generate_test_tasks(env: GELabEnvironment, test_subtree: int = 4, 
                        num_tasks: int = 2162) -> List[Dict]:
    """
    Generate test tasks from the test subtree.
    Matches paper's distribution: randomly select start/end pairs.
    """
    # Get pages in test subtree
    test_pages = env.get_pages_by_subtree(test_subtree)
    test_pages.append('page_0')  # Include root
    
    tasks = []
    task_id = 0
    
    # Generate random start/end pairs
    while len(tasks) < num_tasks:
        start = random.choice(test_pages)
        end = random.choice(test_pages)
        
        if start == end:
            continue
        
        path_len = env.get_shortest_path_length(start, end)
        if path_len <= 0 or path_len > 7:
            continue
        
        tasks.append({
            'id': task_id,
            'start': start,
            'end': end,
            'path_length': path_len
        })
        task_id += 1
    
    return tasks


def evaluate_interactive(evaluator: InteractiveEvaluator, tasks: List[Dict], 
                         num_attempts: int = 5) -> Dict:
    """
    Run interactive evaluation with Pass@1 and Pass@5.
    """
    results_by_path = defaultdict(lambda: {'pass1': [], 'pass5': []})
    
    for task in tqdm(tasks, desc="Evaluating"):
        start = task['start']
        end = task['end']
        path_len = task['path_length']

        # Run multiple attempts: first greedy, rest with sampling
        successes = []
        for attempt in range(num_attempts):
            use_sampling = (attempt > 0)
            success, steps, trajectory = evaluator.run_episode(
                start, end, use_sampling=use_sampling
            )
            successes.append(success)

        # Pass@1 = first attempt success (greedy)
        results_by_path[path_len]['pass1'].append(successes[0])
        # Pass@5 = any of 5 attempts succeeded
        results_by_path[path_len]['pass5'].append(any(successes))
    
    # Calculate metrics
    metrics = {}
    for path_len in sorted(results_by_path.keys()):
        data = results_by_path[path_len]
        pass1 = sum(data['pass1']) / len(data['pass1']) * 100
        pass5 = sum(data['pass5']) / len(data['pass5']) * 100
        metrics[path_len] = {
            'pass1': pass1,
            'pass5': pass5,
            'num_tasks': len(data['pass1'])
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Interactive GE-Lab Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--env_dir", type=str, 
                        default="data_engine/ui_environment/20260129_085348",
                        help="Path to UI environment directory")
    parser.add_argument("--test_subtree", type=int, default=4,
                        help="Subtree to use for testing (OOD)")
    parser.add_argument("--num_tasks", type=int, default=500,
                        help="Number of test tasks (default 500 for quick eval)")
    parser.add_argument("--num_attempts", type=int, default=5,
                        help="Number of attempts for Pass@N")
    parser.add_argument("--save_results", type=str, default="results/interactive_eval.json",
                        help="Path to save results")
    args = parser.parse_args()
    
    # Load environment
    print("Loading environment...")
    env = GELabEnvironment(
        ui_structure_path=os.path.join(args.env_dir, "ui_structure.json"),
        ui_layer_path=os.path.join(args.env_dir, "ui_structure_layer.json"),
        pages_dir=os.path.join(args.env_dir, "pages")
    )
    print(f"Loaded {len(env.pages)} pages")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    
    # Check if it's a LoRA adapter checkpoint
    is_lora = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    
    if is_lora:
        # Load base SFT model first, then apply ST-RL LoRA
        sft_base_path = "./checkpoint/gui_exp/sft_448_retrain/v0-20260201_054616/checkpoint-850"
        print(f"Loading SFT base model from {sft_base_path}...")
        
        # Check if SFT is also LoRA
        sft_is_lora = os.path.exists(os.path.join(sft_base_path, "adapter_config.json"))
        
        if sft_is_lora:
            # Load original Qwen model, apply SFT LoRA, then ST-RL LoRA
            base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
            print(f"Loading base Qwen model from {base_model_path}...")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print("Applying SFT LoRA...")
            model = PeftModel.from_pretrained(base_model, sft_base_path)
            model = model.merge_and_unload()  # Merge SFT LoRA
            print("Applying ST-RL LoRA...")
            model = PeftModel.from_pretrained(model, args.model_path)
        else:
            # SFT is full model, just load and apply ST-RL LoRA
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                sft_base_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        # Full model checkpoint
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    model.eval()
    
    # Load processor from base Qwen model
    base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    # Create evaluator
    evaluator = InteractiveEvaluator(model, processor, env)
    
    # Generate test tasks
    print(f"Generating {args.num_tasks} test tasks from subtree {args.test_subtree}...")
    random.seed(42)
    tasks = generate_test_tasks(env, args.test_subtree, args.num_tasks)
    
    # Show task distribution
    path_dist = defaultdict(int)
    for t in tasks:
        path_dist[t['path_length']] += 1
    print("Task distribution by path length:")
    for p in sorted(path_dist.keys()):
        print(f"  Path@{p}: {path_dist[p]}")
    
    # Run evaluation
    print(f"\nRunning interactive evaluation with {args.num_attempts} attempts...")
    metrics = evaluate_interactive(evaluator, tasks, args.num_attempts)
    
    # Print results
    print("\n" + "="*70)
    print("INTERACTIVE BENCHMARK RESULTS")
    print("="*70)
    print(f"\n{'Path':<8} {'Pass@1':<12} {'Pass@5':<12} {'Tasks':<10}")
    print("-"*42)
    
    total_pass1 = 0
    total_pass5 = 0
    total_tasks = 0
    
    for path_len in sorted(metrics.keys()):
        m = metrics[path_len]
        print(f"{path_len:<8} {m['pass1']:<12.2f} {m['pass5']:<12.2f} {m['num_tasks']:<10}")
        total_pass1 += m['pass1'] * m['num_tasks']
        total_pass5 += m['pass5'] * m['num_tasks']
        total_tasks += m['num_tasks']
    
    print("-"*42)
    print(f"{'Overall':<8} {total_pass1/total_tasks:<12.2f} {total_pass5/total_tasks:<12.2f} {total_tasks:<10}")
    
    # Save results
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    with open(args.save_results, 'w') as f:
        json.dump({
            'metrics': metrics,
            'config': vars(args)
        }, f, indent=2)
    print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
