"""
Interactive Benchmark Evaluation

Simulates agent navigation through the environment:
1. Agent receives current screen + task instruction
2. Agent predicts action (click coordinates)
3. Environment executes action and transitions to new page
4. Repeat until: target reached (success), max steps (fail), or invalid action (fail)

Metrics:
- Pass@1: Success rate with single attempt
- Pass@5: Success rate with 5 attempts (any success counts)
"""

import os
import sys
import json
import re
import argparse
import random
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor


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

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GUIEnvironment:
    """Simulates the GUI navigation environment."""
    
    def __init__(self, env_dir: str):
        self.env_dir = env_dir
        self.pages_dir = os.path.join(env_dir, "pages")
        
        with open(os.path.join(env_dir, "ui_structure.json")) as f:
            ui_structure = json.load(f)
        
        self.pages = ui_structure.get("pages", {})
        self.canvas_size = 448
        
    def get_page_image_path(self, page_id: str) -> str:
        return os.path.join(self.pages_dir, f"{page_id}.png")
    
    def get_transitions(self, page_id: str) -> List[Dict]:
        """Get all valid transitions from a page."""
        page_data = self.pages.get(page_id, {})
        return page_data.get("transitions", [])
    
    def execute_action(self, page_id: str, click_x: int, click_y: int) -> Tuple[Optional[str], str]:
        """
        Execute a click action and return (next_page, action_name).
        Returns (None, "invalid") if click doesn't hit any icon.
        
        click_x, click_y are in 0-1000 normalized coordinates.
        """
        # Convert normalized coords to pixel coords
        px = int(click_x / 1000 * self.canvas_size)
        py = int(click_y / 1000 * self.canvas_size)
        
        transitions = self.get_transitions(page_id)
        
        for trans in transitions:
            bbox = trans.get("icon_bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            
            if x1 <= px <= x2 and y1 <= py <= y2:
                return trans["target_page"], trans["action"]
        
        return None, "invalid"


class InteractiveEvaluator:
    """Evaluates model on interactive navigation tasks."""
    
    def __init__(self, model_path: str, env_dir: str, max_steps: int = 12, device: str = "cuda:0"):
        self.env = GUIEnvironment(env_dir)
        self.max_steps = max_steps
        self.model_path = model_path
        self.device = device
        self.llm = None
        self.processor = None
        
    def load_model(self):
        """Load model for inference using vllm."""
        print(f"Loading model from {self.model_path}...")
        
        # Load processor for chat template
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Load model with vllm
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            limit_mm_per_prompt={"image": 1},
        )
        print("Model loaded.")
        
    def parse_action(self, response: str) -> Tuple[str, Optional[Tuple[int, int]]]:
        """
        Parse model response to extract action type and coordinates.
        Returns: (action_type, (x, y)) or (action_type, None)
        """
        response = response.strip()
        
        # Check for complete action
        if "complete" in response.lower():
            return "complete", None
        
        # Extract coordinates from click action
        # Pattern: click(start_box='<|box_start|>(x,y)<|box_end|>')
        coord_pattern = r'\((\d+)\s*,\s*(\d+)\)'
        matches = re.findall(coord_pattern, response)
        
        if matches:
            x, y = int(matches[-1][0]), int(matches[-1][1])
            return "click", (x, y)
        
        return "invalid", None
    
    def build_prompt(self, task: str, current_page: str, history: List[str]) -> str:
        """Build the prompt for the model."""
        if history:
            history_str = "; ".join(history)
        else:
            history_str = "Null"
        
        prompt = f"<image>Instruction: {task}. History: {history_str}"
        return prompt
    
    def predict(self, image_path: str, instruction: str, temperature: float = 0.0) -> str:
        """Get model prediction for a single sample using vllm."""
        image = Image.open(image_path).convert("RGB")
        
        # Build chat message with system prompt and image placeholder
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        
        # Apply chat template to get proper prompt format
        prompt = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Set sampling params
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            max_tokens=100,
        )
        
        # Generate with vllm
        outputs = self.llm.generate(
            [{
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }],
            sampling_params=sampling_params,
        )
        
        response = outputs[0].outputs[0].text.strip()
        return response
    
    def run_episode(self, start_page: str, target_page: str, temperature: float = 0.0) -> Tuple[bool, int, List[str]]:
        """
        Run a single navigation episode.
        Returns: (success, steps_taken, trajectory)
        """
        current_page = start_page
        history = []
        trajectory = [current_page]
        task = f"from {start_page} to {target_page}"
        
        for step in range(self.max_steps):
            # Check if already at target
            if current_page == target_page:
                return True, step, trajectory
            
            # Build prompt
            prompt = self.build_prompt(task, current_page, history)
            image_path = self.env.get_page_image_path(current_page)
            
            # Get model prediction
            response = self.predict(image_path, prompt, temperature)
            action_type, coords = self.parse_action(response)
            
            if action_type == "complete":
                # Model thinks it's at target
                return current_page == target_page, step + 1, trajectory
            
            if action_type == "invalid" or coords is None:
                # Invalid action format
                return False, step + 1, trajectory
            
            # Execute action
            next_page, action_name = self.env.execute_action(current_page, coords[0], coords[1])
            
            if next_page is None:
                # Click didn't hit any valid target
                return False, step + 1, trajectory
            
            # Update state
            history.append(f"step{step+1}: click {action_name} icon on {current_page}")
            current_page = next_page
            trajectory.append(current_page)
        
        # Max steps reached
        return current_page == target_page, self.max_steps, trajectory
    
    def evaluate(self, tasks: List[Dict], num_attempts: int = 5) -> Dict:
        """
        Evaluate on a list of tasks.
        Returns metrics including Pass@1 and Pass@N.
        """
        results = {
            "pass_at_1": 0,
            f"pass_at_{num_attempts}": 0,
            "total": len(tasks),
            "by_path_length": {},
        }
        
        for task in tqdm(tasks, desc="Evaluating"):
            start_page = task["start"]
            target_page = task["end"]
            path_length = task.get("path_length", "unknown")
            
            # Initialize path length stats
            if path_length not in results["by_path_length"]:
                results["by_path_length"][path_length] = {
                    "pass_at_1": 0,
                    f"pass_at_{num_attempts}": 0,
                    "total": 0,
                }
            results["by_path_length"][path_length]["total"] += 1
            
            # Pass@1: Single attempt with temperature=0
            success_1, steps_1, traj_1 = self.run_episode(start_page, target_page, temperature=0.0)
            
            if success_1:
                results["pass_at_1"] += 1
                results["by_path_length"][path_length]["pass_at_1"] += 1
            
            # Pass@N: Multiple attempts with temperature>0
            any_success = success_1
            for attempt in range(1, num_attempts):
                if any_success:
                    break
                success_n, _, _ = self.run_episode(start_page, target_page, temperature=1.0)
                if success_n:
                    any_success = True
            
            if any_success:
                results[f"pass_at_{num_attempts}"] += 1
                results["by_path_length"][path_length][f"pass_at_{num_attempts}"] += 1
        
        # Calculate percentages
        results["pass_at_1_pct"] = 100 * results["pass_at_1"] / results["total"]
        results[f"pass_at_{num_attempts}_pct"] = 100 * results[f"pass_at_{num_attempts}"] / results["total"]
        
        for pl in results["by_path_length"]:
            pl_data = results["by_path_length"][pl]
            pl_data["pass_at_1_pct"] = 100 * pl_data["pass_at_1"] / pl_data["total"] if pl_data["total"] > 0 else 0
            pl_data[f"pass_at_{num_attempts}_pct"] = 100 * pl_data[f"pass_at_{num_attempts}"] / pl_data["total"] if pl_data["total"] > 0 else 0
        
        return results


def generate_interactive_tasks(env: GUIEnvironment, subtree: int, num_tasks: int = 500) -> List[Dict]:
    """Generate random navigation tasks within a subtree."""
    import networkx as nx
    
    # Build graph
    G = nx.DiGraph()
    for page_id, page_data in env.pages.items():
        for trans in page_data.get("transitions", []):
            G.add_edge(page_id, trans["target_page"])
    
    # Find pages in target subtree (pages reachable from subtree root)
    root_children = sorted([n for n in G.successors("page_0") if n != "page_0"],
                          key=lambda x: int(x.split('_')[1]))
    
    if subtree >= len(root_children):
        raise ValueError(f"Subtree {subtree} not found. Only {len(root_children)} subtrees.")
    
    subtree_root = root_children[subtree]
    
    # BFS to find all pages in subtree
    subtree_pages = {subtree_root}
    queue = [subtree_root]
    while queue:
        current = queue.pop(0)
        for neighbor in G.successors(current):
            if neighbor not in subtree_pages and neighbor != "page_0":
                subtree_pages.add(neighbor)
                queue.append(neighbor)
    
    subtree_pages = list(subtree_pages)
    print(f"Subtree {subtree} has {len(subtree_pages)} pages")
    
    # Generate random tasks
    tasks = []
    attempts = 0
    max_attempts = num_tasks * 10
    
    while len(tasks) < num_tasks and attempts < max_attempts:
        attempts += 1
        start = random.choice(subtree_pages)
        end = random.choice(subtree_pages)
        
        if start == end:
            continue
        
        try:
            path = nx.shortest_path(G, start, end)
            path_length = len(path) - 1
            
            tasks.append({
                "start": start,
                "end": end,
                "path_length": path_length,
            })
        except nx.NetworkXNoPath:
            continue
    
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Interactive Benchmark Evaluation")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--env_dir", default="data_engine/ui_environment_448/latest",
                       help="Path to environment directory")
    parser.add_argument("--subtree", type=int, default=4, help="Subtree for OOD testing (default: 4)")
    parser.add_argument("--num_tasks", type=int, default=500, help="Number of tasks to evaluate")
    parser.add_argument("--max_steps", type=int, default=12, help="Max steps per episode")
    parser.add_argument("--num_attempts", type=int, default=5, help="Attempts for Pass@N")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    parser.add_argument("--device", default="cuda:0", help="Device to run on")
    args = parser.parse_args()
    
    # Resolve paths
    if not os.path.isabs(args.env_dir):
        args.env_dir = os.path.join(os.getcwd(), args.env_dir)
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(os.getcwd(), args.model_path)
    
    print("=" * 60)
    print("INTERACTIVE BENCHMARK EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Environment: {args.env_dir}")
    print(f"Subtree: {args.subtree} (OOD)")
    print(f"Tasks: {args.num_tasks}")
    print(f"Max steps: {args.max_steps}")
    print()
    
    # Initialize environment
    env = GUIEnvironment(args.env_dir)
    
    # Generate tasks
    print("Generating tasks...")
    tasks = generate_interactive_tasks(env, args.subtree, args.num_tasks)
    print(f"Generated {len(tasks)} tasks")
    
    # Path length distribution
    pl_dist = {}
    for t in tasks:
        pl = t["path_length"]
        pl_dist[pl] = pl_dist.get(pl, 0) + 1
    print("Path length distribution:")
    for pl in sorted(pl_dist.keys()):
        print(f"  Length {pl}: {pl_dist[pl]}")
    print()
    
    # Initialize evaluator
    evaluator = InteractiveEvaluator(args.model_path, args.env_dir, args.max_steps, args.device)
    evaluator.load_model()
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate(tasks, num_attempts=args.num_attempts)
    
    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Pass@1: {results['pass_at_1_pct']:.2f}% ({results['pass_at_1']}/{results['total']})")
    print(f"Pass@{args.num_attempts}: {results[f'pass_at_{args.num_attempts}_pct']:.2f}% ({results[f'pass_at_{args.num_attempts}']}/{results['total']})")
    print()
    print("By path length:")
    for pl in sorted(results["by_path_length"].keys()):
        pl_data = results["by_path_length"][pl]
        print(f"  Path@{pl}: Pass@1={pl_data['pass_at_1_pct']:.1f}%, Pass@{args.num_attempts}={pl_data[f'pass_at_{args.num_attempts}_pct']:.1f}% (n={pl_data['total']})")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
