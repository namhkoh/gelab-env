"""
Multi-GPU Interactive Environment Evaluation for GE-Lab

Distributes tasks across multiple GPUs for parallel evaluation.
"""

import json
import os
import re
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, Manager
import torch
from PIL import Image
from tqdm import tqdm


class GELabEnvironment:
    """Interactive GE-Lab environment for evaluation."""
    
    def __init__(self, ui_structure_path: str, ui_layer_path: str, pages_dir: str):
        with open(ui_structure_path, 'r') as f:
            data = json.load(f)
        self.pages = data['pages']
        self.pages_dir = pages_dir
        
        with open(ui_layer_path, 'r') as f:
            self.layer_data = json.load(f)
        self.page_to_subtree = self._build_subtree_mapping()
        self.nav_graph = self._build_nav_graph()
        
        self.CANVAS_WIDTH = 1179
        self.CANVAS_HEIGHT = 2556
        self.NORM_SIZE = 1000
        
    def _build_subtree_mapping(self) -> Dict[str, int]:
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
        return os.path.join(self.pages_dir, f"{page_id}.png")
    
    def normalize_bbox(self, bbox: List[int]) -> List[int]:
        x1, y1, x2, y2 = bbox
        return [
            int(x1 / self.CANVAS_WIDTH * self.NORM_SIZE),
            int(y1 / self.CANVAS_HEIGHT * self.NORM_SIZE),
            int(x2 / self.CANVAS_WIDTH * self.NORM_SIZE),
            int(y2 / self.CANVAS_HEIGHT * self.NORM_SIZE)
        ]
    
    def click(self, page_id: str, x: int, y: int) -> Tuple[str, bool]:
        if page_id not in self.pages:
            return page_id, False
        layout = self.pages[page_id].get('layout', {})
        for icon_name, icon_info in layout.items():
            bbox = icon_info['bbox']
            norm_bbox = self.normalize_bbox(bbox)
            x_min, y_min, x_max, y_max = norm_bbox
            if x_min <= x <= x_max and y_min <= y <= y_max:
                for trans in self.pages[page_id].get('transitions', []):
                    if trans['action'] == icon_name:
                        return trans['target_page'], True
        return page_id, False
    
    def get_shortest_path_length(self, start: str, end: str) -> int:
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
        return -1
    
    def get_pages_by_subtree(self, subtree: int) -> List[str]:
        return [p for p, s in self.page_to_subtree.items() if s == subtree]


def worker_init(gpu_id: int, model_path: str):
    """Initialize model on specific GPU for worker process."""
    global worker_model, worker_processor, worker_device
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    worker_device = "cuda:0"
    
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    worker_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    worker_model.eval()
    worker_processor = AutoProcessor.from_pretrained(model_path)
    
    return gpu_id


def parse_action(response: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    response_lower = response.lower()
    if 'complete' in response_lower:
        return 'complete', None
    coord_match = re.search(r'\((\d+),\s*(\d+)\)', response)
    if coord_match:
        x, y = int(coord_match.group(1)), int(coord_match.group(2))
        return 'click', (x, y)
    return 'invalid', None


def get_model_response(img_path: str, prompt: str) -> str:
    """Get model response using worker's model."""
    global worker_model, worker_processor, worker_device
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = worker_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image = Image.open(img_path).convert('RGB')
    
    inputs = worker_processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(worker_device)
    
    with torch.no_grad():
        output_ids = worker_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=worker_processor.tokenizer.pad_token_id
        )
    
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = worker_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    
    return response


def run_episode(env: GELabEnvironment, start_page: str, target_page: str, 
                max_steps: int = 15) -> Tuple[bool, int, List[str]]:
    """Run single episode."""
    current_page = start_page
    task = f"from {start_page} to {target_page}"
    history = []
    trajectory = [current_page]
    
    for step in range(max_steps):
        img_path = env.get_page_image_path(current_page)
        if not os.path.exists(img_path):
            return False, step + 1, trajectory
        
        history_str = " ".join(history) if history else "Null"
        prompt = f"Instruction: {task}. History: {history_str}"
        
        response = get_model_response(img_path, prompt)
        action_type, coords = parse_action(response)
        
        if action_type == 'complete':
            success = (current_page == target_page)
            return success, step + 1, trajectory
        
        elif action_type == 'click' and coords:
            x, y = coords
            new_page, valid = env.click(current_page, x, y)
            history.append(f"step{step+1}: click at ({x},{y}) on {current_page}")
            if new_page != current_page:
                trajectory.append(new_page)
            current_page = new_page
        else:
            history.append(f"step{step+1}: invalid action on {current_page}")
    
    return current_page == target_page, max_steps, trajectory


def evaluate_task(args):
    """Evaluate single task - called by worker."""
    task, env_config, num_attempts = args
    
    # Reconstruct environment in worker
    env = GELabEnvironment(
        env_config['ui_structure_path'],
        env_config['ui_layer_path'],
        env_config['pages_dir']
    )
    
    start = task['start']
    end = task['end']
    path_len = task['path_length']
    
    successes = []
    for attempt in range(num_attempts):
        success, steps, trajectory = run_episode(env, start, end)
        successes.append(success)
    
    return {
        'task_id': task['id'],
        'path_length': path_len,
        'pass1': successes[0],
        'pass_any': any(successes),
        'successes': successes
    }


def generate_test_tasks(env: GELabEnvironment, test_subtree: int = 4, 
                        num_tasks: int = 500) -> List[Dict]:
    """Generate test tasks from the test subtree."""
    test_pages = env.get_pages_by_subtree(test_subtree)
    test_pages.append('page_0')
    
    tasks = []
    task_id = 0
    
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


def run_gpu_worker(gpu_id: int, tasks: List[Dict], env_config: Dict, 
                   model_path: str, num_attempts: int, results_queue):
    """Worker function for each GPU."""
    # Initialize model on this GPU
    worker_init(gpu_id, model_path)
    
    # Process assigned tasks
    for task in tqdm(tasks, desc=f"GPU {gpu_id}", position=gpu_id):
        result = evaluate_task((task, env_config, num_attempts))
        results_queue.put(result)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Interactive GE-Lab Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_dir", type=str, 
                        default="data_engine/ui_environment/20260129_085348")
    parser.add_argument("--test_subtree", type=int, default=4)
    parser.add_argument("--num_tasks", type=int, default=200)
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--save_results", type=str, default="results/interactive_eval_multigpu.json")
    args = parser.parse_args()
    
    # Load environment for task generation
    print("Loading environment...")
    env = GELabEnvironment(
        ui_structure_path=os.path.join(args.env_dir, "ui_structure.json"),
        ui_layer_path=os.path.join(args.env_dir, "ui_structure_layer.json"),
        pages_dir=os.path.join(args.env_dir, "pages")
    )
    print(f"Loaded {len(env.pages)} pages")
    
    # Generate test tasks
    print(f"Generating {args.num_tasks} test tasks...")
    random.seed(42)
    tasks = generate_test_tasks(env, args.test_subtree, args.num_tasks)
    
    # Show distribution
    path_dist = defaultdict(int)
    for t in tasks:
        path_dist[t['path_length']] += 1
    print("Task distribution:")
    for p in sorted(path_dist.keys()):
        print(f"  Path@{p}: {path_dist[p]}")
    
    # Environment config for workers
    env_config = {
        'ui_structure_path': os.path.join(args.env_dir, "ui_structure.json"),
        'ui_layer_path': os.path.join(args.env_dir, "ui_structure_layer.json"),
        'pages_dir': os.path.join(args.env_dir, "pages")
    }
    
    # Split tasks across GPUs
    tasks_per_gpu = len(tasks) // args.num_gpus
    gpu_tasks = []
    for i in range(args.num_gpus):
        start_idx = i * tasks_per_gpu
        end_idx = start_idx + tasks_per_gpu if i < args.num_gpus - 1 else len(tasks)
        gpu_tasks.append(tasks[start_idx:end_idx])
    
    print(f"\nDistributing {len(tasks)} tasks across {args.num_gpus} GPUs...")
    
    # Use multiprocessing with spawn
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    manager = Manager()
    results_queue = manager.Queue()
    
    # Start workers
    processes = []
    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=run_gpu_worker,
            args=(gpu_id, gpu_tasks[gpu_id], env_config, args.model_path, 
                  args.num_attempts, results_queue)
        )
        p.start()
        processes.append(p)
    
    # Wait for all workers
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    # Calculate metrics
    results_by_path = defaultdict(lambda: {'pass1': [], 'pass_any': []})
    for r in results:
        path_len = r['path_length']
        results_by_path[path_len]['pass1'].append(r['pass1'])
        results_by_path[path_len]['pass_any'].append(r['pass_any'])
    
    # Print results
    print("\n" + "="*70)
    print("INTERACTIVE BENCHMARK RESULTS (Multi-GPU)")
    print("="*70)
    print(f"\n{'Path':<8} {'Pass@1':<12} {'Pass@N':<12} {'Tasks':<10}")
    print("-"*42)
    
    metrics = {}
    total_pass1 = 0
    total_passn = 0
    total_tasks = 0
    
    for path_len in sorted(results_by_path.keys()):
        data = results_by_path[path_len]
        pass1 = sum(data['pass1']) / len(data['pass1']) * 100
        passn = sum(data['pass_any']) / len(data['pass_any']) * 100
        n_tasks = len(data['pass1'])
        
        metrics[path_len] = {'pass1': pass1, 'pass_n': passn, 'num_tasks': n_tasks}
        print(f"{path_len:<8} {pass1:<12.2f} {passn:<12.2f} {n_tasks:<10}")
        
        total_pass1 += pass1 * n_tasks
        total_passn += passn * n_tasks
        total_tasks += n_tasks
    
    print("-"*42)
    print(f"{'Overall':<8} {total_pass1/total_tasks:<12.2f} {total_passn/total_tasks:<12.2f} {total_tasks:<10}")
    
    # Save results
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    with open(args.save_results, 'w') as f:
        json.dump({'metrics': metrics, 'config': vars(args)}, f, indent=2)
    print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
