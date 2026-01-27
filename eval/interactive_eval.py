"""
Interactive Evaluation Script for GeLab Navigation.

This script runs the interactive benchmark evaluation where:
1. Agent receives instruction and current screenshot
2. Agent predicts action (click or complete)
3. Environment executes action and returns new screenshot
4. Continue until complete or max steps reached
5. Compute Pass@K metrics by path length
"""

import os
import sys
import json
import argparse
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.interactive_env import InteractiveEnvironment, TaskGenerator


@dataclass
class EvalResult:
    """Result of a single task evaluation."""
    task_id: int
    source: str
    target: str
    path_length: int
    success: bool
    steps_taken: int
    trajectory: List[Dict]
    final_page: str


def parse_model_output(output: str) -> Tuple[str, Optional[int], Optional[int], Optional[str]]:
    """
    Parse model output to extract action type and coordinates.
    
    Expected format:
    "Explain: click [icon_name] icon on page_X. Action: click(x,y)"
    or
    "Explain: this is target page. Action: complete"
    
    Returns:
        (action_type, x, y, explanation)
    """
    output = output.strip()
    
    # Check for complete action
    if 'action: complete' in output.lower() or 'action:complete' in output.lower():
        return 'complete', None, None, output
    
    # Try to extract click coordinates
    # Pattern 1: click(x,y) or click (x,y)
    click_pattern = r'click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
    match = re.search(click_pattern, output.lower())
    
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return 'click', x, y, output
    
    # Pattern 2: coordinates in parentheses
    coord_pattern = r'\((\d+)\s*,\s*(\d+)\)'
    match = re.search(coord_pattern, output)
    
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return 'click', x, y, output
    
    # Pattern 3: box format <|box_start|>(x,y)<|box_end|>
    box_pattern = r'<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>'
    match = re.search(box_pattern, output)
    
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return 'click', x, y, output
    
    # Could not parse - return unknown
    return 'unknown', None, None, output


def load_model_vllm(model_path: str, gpu_id: int = 0):
    """Load model using vLLM for fast inference."""
    from vllm import LLM, SamplingParams
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy for evaluation
        max_tokens=256,
        stop=['<|im_end|>'],
    )
    
    return llm, sampling_params


def format_prompt(instruction: str, history: str, image_path: str) -> str:
    """Format the prompt for the model."""
    if history:
        prompt = f"<image>Instruction: {instruction}. History: {history}"
    else:
        prompt = f"<image>Instruction: {instruction}."
    
    return prompt


def run_single_task(
    env: InteractiveEnvironment,
    model,
    sampling_params,
    processor,
    task: Dict,
    max_steps: int = 12,
    verbose: bool = False
) -> EvalResult:
    """
    Run a single interactive evaluation task.
    
    Args:
        env: Interactive environment
        model: vLLM model
        sampling_params: vLLM sampling parameters
        processor: Image processor
        task: Task dictionary with source, target, instruction
        max_steps: Maximum steps allowed
        verbose: Print debug info
        
    Returns:
        EvalResult
    """
    from vllm import LLM
    from PIL import Image
    
    source = task['source']
    target = task['target']
    instruction = f"from {source} to {target}"
    
    # Reset environment
    env.reset(source)
    trajectory = []
    
    for step in range(max_steps):
        current_page = env.get_current_page()
        
        # Check if already at target
        if env.is_at_target(target):
            break
        
        # Get current screenshot
        screenshot_path = env.get_screenshot_path()
        
        # Format history
        history = env.get_history_string()
        
        # Create prompt
        prompt = format_prompt(instruction, history, screenshot_path)
        
        # Load image
        try:
            image = Image.open(screenshot_path).convert('RGB')
        except Exception as e:
            if verbose:
                print(f"Error loading image {screenshot_path}: {e}")
            break
        
        # Generate model prediction
        from vllm.multimodal import MultiModalData
        
        # Use vLLM's chat interface
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt.replace("<image>", "")}
                ]
            }
        ]
        
        outputs = model.chat(
            messages=messages,
            sampling_params=sampling_params,
        )
        
        response = outputs[0].outputs[0].text
        
        # Parse action from response
        action_type, x, y, explanation = parse_model_output(response)
        
        trajectory.append({
            'step': step + 1,
            'page': current_page,
            'prompt': prompt,
            'response': response,
            'action_type': action_type,
            'coords': (x, y) if x is not None else None
        })
        
        if verbose:
            print(f"  Step {step+1}: Page={current_page}, Action={action_type}, Response={response[:100]}...")
        
        # Execute action
        if action_type == 'complete':
            break
        elif action_type == 'click' and x is not None and y is not None:
            result = env.execute_click(x, y)
            trajectory[-1]['click_result'] = {
                'success': result.success,
                'new_page': result.new_page,
                'clicked_element': result.clicked_element
            }
        else:
            # Failed to parse action - count as failed step
            trajectory[-1]['error'] = 'Failed to parse action'
            break
    
    # Check success
    success = env.is_at_target(target)
    
    return EvalResult(
        task_id=task['id'],
        source=source,
        target=target,
        path_length=task['path_length'],
        success=success,
        steps_taken=len(trajectory),
        trajectory=trajectory,
        final_page=env.get_current_page()
    )


def run_evaluation_simple(
    model_path: str,
    ui_structure_path: str,
    images_dir: str,
    output_path: str,
    subtree_indices: List[int] = None,
    max_steps: int = 12,
    num_attempts: int = 1,
    gpu_id: int = 0,
    verbose: bool = False,
    limit: int = None
):
    """
    Run interactive evaluation using vLLM for fast inference.
    
    This version uses the same approach as inference_qwen2p5_mixed_vllm.py
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    
    print(f"Loading model from {model_path}...")
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Load model using vLLM (same as inference script)
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=256,
        stop_token_ids=[],
    )
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    print("Model loaded.")
    
    # System prompt (same as inference script)
    system_prompt = '''You are a **Multifaceted Mobile Interface Assistant**. Your responsibilities include:
1.  **Navigating** a mobile phone interface to reach a target page based on user instructions, task history, and the current screen state.

Based on the user request and the current screen state (and history if applicable), provide the appropriate output.

**Task: Navigation**
   *   **Goal:** Reach a target page step-by-step.
   *   **Possible Actions:**
      *   `click`: Tap a specific element. Provide coordinates (x, y) relative to a (0,0) top-left and (1000,1000) bottom-right system.
      *   `complete`: Task finished, current screen is the target.
   *   **Output Format:**
      Explain: [Your brief explanation]\tAction: [click(start_box='<|box_start|>(x,y)<|box_end|>') or complete]
'''
    
    # Initialize environment and task generator
    env = InteractiveEnvironment(ui_structure_path, images_dir)
    generator = TaskGenerator(ui_structure_path)
    
    # Generate tasks
    if subtree_indices is None:
        subtree_indices = [4]  # Default to test subtree
    
    tasks = generator.generate_tasks(subtree_indices=subtree_indices, use_paper_distribution=True)
    
    if limit:
        tasks = tasks[:limit]
    
    print(f"Running evaluation on {len(tasks)} tasks...")
    
    # Results storage
    all_results = []
    results_by_path = defaultdict(list)
    
    for task in tqdm(tasks, desc="Evaluating"):
        source = task['source']
        target = task['target']
        instruction = f"from {source} to {target}"
        path_length = task['path_length']
        
        # Run task (potentially multiple attempts for Pass@K)
        task_successes = []
        
        for attempt in range(num_attempts):
            # Reset environment
            env.reset(source)
            trajectory = []
            
            for step in range(max_steps):
                current_page = env.get_current_page()
                
                # Check if already at target
                if env.is_at_target(target):
                    break
                
                # Get current screenshot path
                screenshot_path = env.get_screenshot_path()
                
                # Format history
                history = env.get_history_string()
                
                # Create prompt (match training format exactly)
                if history:
                    text_prompt = f"Instruction: {instruction}. History: {history}"
                else:
                    text_prompt = f"Instruction: {instruction}. History: Null"
                
                # Build messages for vLLM (same format as inference script)
                content = [
                    {"type": "image", "image": screenshot_path},
                    {"type": "text", "text": text_prompt}
                ]
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ]
                
                try:
                    # Process with vLLM
                    prompt = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    mm_data = {}
                    if image_inputs is not None:
                        mm_data["image"] = image_inputs
                    if video_inputs is not None:
                        mm_data["video"] = video_inputs
                    
                    llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
                    
                    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
                    response = outputs[0].outputs[0].text
                    
                except Exception as e:
                    if verbose:
                        print(f"Error generating response: {e}")
                    response = ""
                
                # Parse action
                action_type, x, y, _ = parse_model_output(response)
                
                trajectory.append({
                    'step': step + 1,
                    'page': current_page,
                    'response': response[:200],  # Truncate for storage
                    'action_type': action_type,
                    'coords': (x, y) if x is not None else None
                })
                
                if verbose:
                    print(f"  Step {step+1}: {current_page} -> {action_type} ({x}, {y})")
                    print(f"    Response: {response[:100]}...")
                
                # Execute action
                if action_type == 'complete':
                    break
                elif action_type == 'click' and x is not None and y is not None:
                    result = env.execute_click(x, y)
                    trajectory[-1]['click_result'] = result.success
                    trajectory[-1]['clicked_element'] = result.clicked_element
                    trajectory[-1]['new_page'] = result.new_page
                    if verbose and result.clicked_element:
                        print(f"    Clicked {result.clicked_element} -> {result.new_page}")
                else:
                    # Failed to parse - break
                    if verbose:
                        print(f"    Failed to parse action from: {response[:100]}")
                    break
            
            # Check success
            success = env.is_at_target(target)
            task_successes.append(success)
        
        # Record result
        result = {
            'task_id': task['id'],
            'source': source,
            'target': target,
            'path_length': path_length,
            'successes': task_successes,
            'pass_at_1': task_successes[0] if task_successes else False,
            'pass_at_k': any(task_successes),
            'final_page': env.get_current_page(),
            'steps_taken': len(trajectory),
            'trajectory': trajectory
        }
        
        all_results.append(result)
        results_by_path[path_length].append(result)
    
    # Compute metrics
    print("\n" + "="*60)
    print("INTERACTIVE EVALUATION RESULTS")
    print("="*60)
    
    metrics = {}
    total_pass_1 = 0
    total_pass_k = 0
    total_tasks = 0
    
    for path_length in sorted(results_by_path.keys()):
        path_results = results_by_path[path_length]
        n_tasks = len(path_results)
        pass_1 = sum(1 for r in path_results if r['pass_at_1'])
        pass_k = sum(1 for r in path_results if r['pass_at_k'])
        
        pass_1_rate = (pass_1 / n_tasks * 100) if n_tasks > 0 else 0
        pass_k_rate = (pass_k / n_tasks * 100) if n_tasks > 0 else 0
        
        metrics[f'path_{path_length}'] = {
            'n_tasks': n_tasks,
            'pass_at_1': pass_1_rate,
            'pass_at_k': pass_k_rate
        }
        
        total_pass_1 += pass_1
        total_pass_k += pass_k
        total_tasks += n_tasks
        
        if num_attempts > 1:
            print(f"Path@{path_length}: {n_tasks:4d} tasks | Pass@1: {pass_1_rate:6.2f}% | Pass@{num_attempts}: {pass_k_rate:6.2f}%")
        else:
            print(f"Path@{path_length}: {n_tasks:4d} tasks | Pass@1: {pass_1_rate:6.2f}%")
    
    # Overall metrics
    overall_pass_1 = (total_pass_1 / total_tasks * 100) if total_tasks > 0 else 0
    overall_pass_k = (total_pass_k / total_tasks * 100) if total_tasks > 0 else 0
    
    print("-"*60)
    if num_attempts > 1:
        print(f"Overall:  {total_tasks:4d} tasks | Pass@1: {overall_pass_1:6.2f}% | Pass@{num_attempts}: {overall_pass_k:6.2f}%")
    else:
        print(f"Overall:  {total_tasks:4d} tasks | Pass@1: {overall_pass_1:6.2f}%")
    print("="*60)
    
    # Save results
    output_data = {
        'config': {
            'model_path': model_path,
            'subtree_indices': subtree_indices,
            'max_steps': max_steps,
            'num_attempts': num_attempts
        },
        'metrics': metrics,
        'overall': {
            'total_tasks': total_tasks,
            'pass_at_1': overall_pass_1,
            'pass_at_k': overall_pass_k
        },
        'results': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Interactive Evaluation for GeLab Navigation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--ui_structure', type=str, default='demo/ui_structure.json',
                        help='Path to UI structure JSON')
    parser.add_argument('--images_dir', type=str, default='datas/images',
                        help='Directory containing page screenshots')
    parser.add_argument('--output', type=str, default='eval_results/interactive_eval.json',
                        help='Output path for results')
    parser.add_argument('--subtrees', type=int, nargs='+', default=[4],
                        help='Subtree indices to evaluate (0-4). Default: 4 (test)')
    parser.add_argument('--max_steps', type=int, default=12,
                        help='Maximum steps per task')
    parser.add_argument('--num_attempts', type=int, default=1,
                        help='Number of attempts for Pass@K (default: 1 for Pass@1)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tasks (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run evaluation
    run_evaluation_simple(
        model_path=args.model_path,
        ui_structure_path=args.ui_structure,
        images_dir=args.images_dir,
        output_path=args.output,
        subtree_indices=args.subtrees,
        max_steps=args.max_steps,
        num_attempts=args.num_attempts,
        gpu_id=args.gpu,
        verbose=args.verbose,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
