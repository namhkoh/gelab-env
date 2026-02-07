#!/usr/bin/env python3
"""
Trajectory-based evaluation matching the paper's methodology.
- Start at source page
- Navigate step-by-step to target page
- Success = all steps correct AND reach target
- Failure = any wrong step OR exceed max steps
"""

import os
import json
import argparse
import torch
import re
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Constants
CANVAS_WIDTH = 448  # Must match training data canvas size
CANVAS_HEIGHT = 448  # Must match training data canvas size
NORM_SIZE = 1000
MAX_STEPS = 10  # Maximum allowed steps per trajectory

SYSTEM_PROMPT = open('/ext_hdd2/nhkoh/gelab-env/eval/training_system_prompt.txt').read()


def load_ui_structure(path="environment/demo/ui_structure.json"):
    """Load UI structure and build transition map."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    pages = data['pages']
    
    # Build: page -> {icon_name -> (target_page, bbox_norm)}
    transitions = {}
    for page_id, page_data in pages.items():
        transitions[page_id] = {}
        for t in page_data.get('transitions', []):
            icon_name = t['action']
            target = t['target_page']
            bbox = t['icon_bbox']
            # Normalize bbox to 0-1000
            bbox_norm = [
                int(bbox[0] * NORM_SIZE / CANVAS_WIDTH),
                int(bbox[1] * NORM_SIZE / CANVAS_HEIGHT),
                int(bbox[2] * NORM_SIZE / CANVAS_WIDTH),
                int(bbox[3] * NORM_SIZE / CANVAS_HEIGHT),
            ]
            transitions[page_id][icon_name] = {
                'target': target,
                'bbox_norm': bbox_norm
            }
    
    return pages, transitions


def find_clicked_button(x, y, page_id, transitions):
    """Find which button was clicked based on coordinates."""
    if page_id not in transitions:
        return None, None
    
    for icon_name, data in transitions[page_id].items():
        bbox = data['bbox_norm']
        x_min, y_min, x_max, y_max = bbox
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return icon_name, data['target']
    
    return None, None


def load_test_trajectories(test_file="datas/test.json", samples_per_path=None):
    """Load trajectories from actual test.json file."""
    import json
    import random
    
    with open(test_file) as f:
        test_data = json.load(f)
    
    trajectories = defaultdict(list)
    
    for item in test_data:
        task = item.get('task', '')
        path_len = item.get('path', 0)
        
        # Parse task: "From page_X to page_Y"
        if 'From ' in task and ' to ' in task:
            parts = task.replace('From ', '').split(' to ')
            source = parts[0]
            target = parts[1]
            
            trajectories[path_len].append({
                'source': source,
                'target': target,
                'path_length': path_len,
                'original_item': item
            })
    
    # Sample if needed
    if samples_per_path:
        for path_len in trajectories:
            if len(trajectories[path_len]) > samples_per_path:
                trajectories[path_len] = random.sample(trajectories[path_len], samples_per_path)
    
    return trajectories


def generate_trajectories(pages, transitions, samples_per_path=100):
    """Generate test trajectories - DEPRECATED, use load_test_trajectories instead."""
    # Keep for backward compatibility but recommend using load_test_trajectories
    from collections import deque
    
    trajectories = defaultdict(list)
    
    # BFS to find shortest paths from page_0
    distances = {'page_0': 0}
    paths = {'page_0': ['page_0']}
    queue = deque(['page_0'])
    
    while queue:
        current = queue.popleft()
        if current not in transitions:
            continue
        for icon_name, data in transitions[current].items():
            target = data['target']
            if target not in distances:
                distances[target] = distances[current