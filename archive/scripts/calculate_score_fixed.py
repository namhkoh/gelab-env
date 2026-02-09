"""
Fixed scoring script that handles both coordinate formats:
- <|box_start|>(x,y)<|box_end|>
- (x,y)
"""

import json
import re
import os
import argparse
from tqdm import tqdm
from collections import defaultdict


def extract_coordinates(text):
    """Extract coordinates from various formats."""
    # Try <|box_start|>(x,y)<|box_end|> format
    match = re.search(r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Try plain (x,y) format
    match = re.search(r'\((\d+),(\d+)\)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    return None, None


def extract_icon_name(text):
    """Extract icon name from text."""
    match = re.search(r'click\s+(.*?)\s+icon', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def calculate_score(gt, pred, bbox_norm):
    """
    Calculate score with flexible format matching.
    
    Returns dict with detailed scores:
    - action_correct: Action type matches (click/complete)
    - icon_correct: Icon name matches
    - coord_correct: Coordinates within bbox
    - overall: All correct
    """
    result = {
        'action_correct': 0,
        'icon_correct': 0,
        'coord_correct': 0,
        'overall': 0
    }
    
    # Handle </think> prefix if present
    pred = pred.split("</think>")[-1].strip()
    
    # Check for complete action
    gt_has_complete = 'complete' in gt.lower()
    pred_has_complete = 'complete' in pred.lower()
    
    if gt_has_complete:
        if pred_has_complete:
            result['action_correct'] = 1
            result['overall'] = 1
        return result
    
    # For click actions
    gt_has_click = 'click' in gt.lower()
    pred_has_click = 'click' in pred.lower()
    
    if gt_has_click and pred_has_click:
        result['action_correct'] = 1
    
    # Check icon name
    gt_icon = extract_icon_name(gt)
    pred_icon = extract_icon_name(pred)
    
    if gt_icon and pred_icon and gt_icon.lower() == pred_icon.lower():
        result['icon_correct'] = 1
    
    # Check coordinates
    pred_x, pred_y = extract_coordinates(pred)
    
    if pred_x is not None and bbox_norm:
        x_min, y_min, x_max, y_max = bbox_norm
        if x_min <= pred_x <= x_max and y_min <= pred_y <= y_max:
            result['coord_correct'] = 1
    
    # Overall: need action + (icon OR coord) correct
    if result['action_correct'] and (result['icon_correct'] or result['coord_correct']):
        result['overall'] = 1
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()
    
    # Load results
    results = []
    with open(args.file, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    
    print(f"Loaded {len(results)} results")
    
    # Score tracking
    scores = defaultdict(list)
    path_scores = {}
    
    for data in tqdm(results, desc="Scoring"):
        gt = data['messages'][1]['content']
        pred = data.get('prediction', '')
        bbox_norm = data.get('bbox_norm', [0, 0, 1000, 1000])
        path_len = data.get('path', 1)
        task = data.get('task', '')
        
        score = calculate_score(gt, pred, bbox_norm)
        
        # Track scores
        for key, val in score.items():
            scores[key].append(val)
        
        # Track by path length
        if path_len not in path_scores:
            path_scores[path_len] = {'overall': [], 'tasks': set()}
        path_scores[path_len]['overall'].append(score['overall'])
        path_scores[path_len]['tasks'].add(task)
    
    # Print results
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    
    for key in ['action_correct', 'icon_correct', 'coord_correct', 'overall']:
        vals = scores[key]
        acc = sum(vals) / len(vals) * 100 if vals else 0
        print(f"{key}: {acc:.2f}% ({sum(vals)}/{len(vals)})")
    
    print("\n" + "="*60)
    print("METRICS BY PATH LENGTH")
    print("="*60)
    
    for path_len in sorted(path_scores.keys()):
        data = path_scores[path_len]
        overall = data['overall']
        acc = sum(overall) / len(overall) * 100 if overall else 0
        n_tasks = len(data['tasks'])
        print(f"Path {path_len}: {acc:.2f}% ({sum(overall)}/{len(overall)}) - {n_tasks} unique tasks")
    
    # Task completion rate (complete actions only)
    complete_gt = [1 for d in results if 'complete' in d['messages'][1]['content'].lower()]
    complete_pred = [1 for d in results if 'complete' in d.get('prediction', '').lower() and 'complete' in d['messages'][1]['content'].lower()]
    
    print("\n" + "="*60)
    print("TASK COMPLETION DETECTION")
    print("="*60)
    print(f"Complete action accuracy: {len(complete_pred)}/{len(complete_gt)} = {len(complete_pred)/len(complete_gt)*100:.2f}%")


if __name__ == "__main__":
    main()
