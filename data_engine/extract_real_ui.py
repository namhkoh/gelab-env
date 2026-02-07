"""
Extract UI elements from real screenshots using omni parser results.
Create a GE-Lab environment that replicates the Amazon shopping flow.
"""

import os
import json
from PIL import Image
from typing import List, Dict, Tuple
import shutil


def extract_icons_from_screenshot(
    screenshot_path: str,
    bbox_json_path: str,
    output_dir: str,
    target_size: Tuple[int, int] = (50, 50)
) -> List[Dict]:
    """
    Extract icons from screenshot using bbox coordinates.
    
    Args:
        screenshot_path: Path to original screenshot
        bbox_json_path: Path to JSON with bbox annotations
        output_dir: Directory to save extracted icons
        target_size: Size to resize icons to
    
    Returns:
        List of extracted icon info
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load screenshot
    img = Image.open(screenshot_path)
    img_width, img_height = img.size
    
    # Load bbox annotations
    with open(bbox_json_path, 'r') as f:
        elements = json.load(f)
    
    extracted = []
    
    for i, elem in enumerate(elements):
        if elem.get('type') != 'icon':
            continue
        
        # Get normalized bbox [x1, y1, x2, y2]
        bbox_norm = elem['bbox']
        
        # Convert to pixel coordinates
        x1 = int(bbox_norm[0] * img_width)
        y1 = int(bbox_norm[1] * img_height)
        x2 = int(bbox_norm[2] * img_width)
        y2 = int(bbox_norm[3] * img_height)
        
        # Ensure valid bbox
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Skip if too small
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue
        
        # Crop icon
        icon = img.crop((x1, y1, x2, y2))
        
        # Create safe filename from content
        content = elem.get('content', f'icon_{i}')
        safe_name = "".join(c if c.isalnum() else '_' for c in content)[:30]
        filename = f"{i:02d}_{safe_name}.png"
        
        # Save original size
        icon_path = os.path.join(output_dir, filename)
        icon.save(icon_path)
        
        # Also save resized version
        icon_resized = icon.resize(target_size, Image.Resampling.LANCZOS)
        icon_resized_path = os.path.join(output_dir, f"resized_{filename}")
        icon_resized.save(icon_resized_path)
        
        extracted.append({
            'id': i,
            'content': content,
            'bbox_norm': bbox_norm,
            'bbox_pixel': [x1, y1, x2, y2],
            'interactivity': elem.get('interactivity', False),
            'icon_path': icon_path,
            'icon_resized_path': icon_resized_path,
            'source': elem.get('source', 'unknown')
        })
    
    return extracted


def create_gelab_env_from_sequence(
    sequence_id: str,
    screenshots_dir: str,
    omni_json_dir: str,
    sequence_json_path: str,
    output_dir: str,
    canvas_size: Tuple[int, int] = (448, 448)
) -> str:
    """
    Create a GE-Lab environment from a real UI sequence.
    
    Args:
        sequence_id: ID of the sequence (e.g., "2451545728360121")
        screenshots_dir: Directory containing original screenshots
        omni_json_dir: Directory containing omni parser JSON results
        sequence_json_path: Path to sequence JSON with actions
        output_dir: Output directory for GE-Lab environment
        canvas_size: Size of GE-Lab canvas
    
    Returns:
        Path to generated ui_structure.json
    """
    os.makedirs(output_dir, exist_ok=True)
    pages_dir = os.path.join(output_dir, "pages")
    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    # Load sequence info
    with open(sequence_json_path, 'r') as f:
        sequence = json.load(f)
    
    steps = sequence['steps']
    num_steps = len(steps)
    
    pages_data = {}
    all_icons_info = {}
    
    print(f"Processing {num_steps} steps...")
    
    for step_idx in range(num_steps):
        step = steps[step_idx]
        screenshot_name = f"{sequence_id}_{step_idx}.png"
        screenshot_path = os.path.join(screenshots_dir, screenshot_name)
        
        json_name = f"{sequence_id}_{step_idx}_agent.json"
        json_path = os.path.join(omni_json_dir, json_name)
        
        if not os.path.exists(screenshot_path):
            print(f"Warning: Screenshot not found: {screenshot_path}")
            continue
        
        if not os.path.exists(json_path):
            print(f"Warning: JSON not found: {json_path}")
            continue
        
        # Create step assets directory
        step_assets_dir = os.path.join(assets_dir, f"step_{step_idx}")
        
        # Extract icons
        print(f"  Step {step_idx}: Extracting icons...")
        icons = extract_icons_from_screenshot(
            screenshot_path, json_path, step_assets_dir
        )
        
        # Load and resize screenshot for GE-Lab page
        orig_img = Image.open(screenshot_path)
        page_img = orig_img.resize(canvas_size, Image.Resampling.LANCZOS)
        page_path = os.path.join(pages_dir, f"step_{step_idx}.png")
        page_img.save(page_path)
        
        # Calculate scale factor
        scale_x = canvas_size[0] / orig_img.width
        scale_y = canvas_size[1] / orig_img.height
        
        # Build page layout with scaled bboxes
        layout = {}
        for icon in icons:
            bbox_pixel = icon['bbox_pixel']
            scaled_bbox = [
                int(bbox_pixel[0] * scale_x),
                int(bbox_pixel[1] * scale_y),
                int(bbox_pixel[2] * scale_x),
                int(bbox_pixel[3] * scale_y)
            ]
            
            # Use content as key, sanitized
            key = "".join(c if c.isalnum() else '_' for c in icon['content'])[:20]
            if not key:
                key = f"icon_{icon['id']}"
            
            layout[key] = {
                'bbox': scaled_bbox,
                'content': icon['content'],
                'interactivity': icon['interactivity']
            }
        
        # Get action info from sequence
        action = step.get('action', 'CLICK')
        action_info = step.get('info', [])
        
        page_id = f"step_{step_idx}"
        pages_data[page_id] = {
            'image': f"step_{step_idx}.png",
            'step_idx': step_idx,
            'layout': layout,
            'action': action,
            'action_info': action_info,
            'description': step.get('description', ''),
            'low_level_instruction': step.get('low_level_instruction', ''),
            'transitions': []
        }
        
        all_icons_info[page_id] = icons
    
    # Build transitions based on sequence flow
    page_ids = sorted(pages_data.keys(), key=lambda x: int(x.split('_')[1]))
    
    for i, page_id in enumerate(page_ids[:-1]):
        next_page_id = page_ids[i + 1]
        step = steps[int(page_id.split('_')[1])]
        
        action = step.get('action', 'CLICK')
        action_info = step.get('info', '')
        
        # Determine action type and bbox
        if action == 'TEXT':
            action_type = 'TEXT'
            text_input = action_info if isinstance(action_info, str) else ''
            bbox = None  # TEXT actions don't have specific bbox
        elif action == 'SCROLL':
            action_type = 'SCROLL'
            text_input = None
            bbox = None
        elif action == 'CLICK' and action_info == 'KEY_HOME':
            action_type = 'KEY_HOME'
            text_input = None
            bbox = None
        elif action == 'COMPLETE':
            action_type = 'COMPLETE'
            text_input = None
            bbox = None
        else:
            action_type = 'CLICK'
            text_input = None
            # Get bbox from sam2_bbox if available
            sam_bbox = step.get('sam2_bbox', [])
            if sam_bbox:
                # Scale sam2_bbox to canvas size
                orig_w, orig_h = 1344, 2992  # From device_info
                bbox = [
                    int(sam_bbox[0] * canvas_size[0] / orig_w),
                    int(sam_bbox[1] * canvas_size[1] / orig_h),
                    int(sam_bbox[2] * canvas_size[0] / orig_w),
                    int(sam_bbox[3] * canvas_size[1] / orig_h)
                ]
            else:
                bbox = None
        
        pages_data[page_id]['transitions'].append({
            'action': f"{action_type}:{action_info}" if text_input else action_type,
            'action_type': action_type,
            'target_page': next_page_id,
            'bbox': bbox,
            'text_input': text_input,
            'instruction': step.get('low_level_instruction', '')
        })
    
    # Save ui_structure.json
    ui_structure = {
        'version': '2.0',
        'source': 'real_ui_sequence',
        'sequence_id': sequence_id,
        'pages': pages_data,
        'metadata': {
            'total_pages': len(pages_data),
            'task_info': sequence.get('task_info', {}),
            'device_info': sequence.get('device_info', {}),
            'supported_actions': ['CLICK', 'TEXT', 'SCROLL', 'KEY_HOME', 'COMPLETE']
        }
    }
    
    json_path = os.path.join(output_dir, 'ui_structure_real.json')
    with open(json_path, 'w') as f:
        json.dump(ui_structure, f, indent=2)
    
    # Save icons info
    icons_info_path = os.path.join(output_dir, 'icons_info.json')
    with open(icons_info_path, 'w') as f:
        json.dump(all_icons_info, f, indent=2, default=str)
    
    print(f"\nGE-Lab environment created at: {output_dir}")
    print(f"  Pages: {len(pages_data)}")
    print(f"  UI Structure: {json_path}")
    
    return json_path


if __name__ == "__main__":
    import time
    
    # Configuration
    sequence_id = "2451545728360121"
    screenshots_dir = "Sequence/screenshots"
    omni_json_dir = "omni_result/agent_json"
    sequence_json_path = f"Sequence/{sequence_id}.json"
    output_dir = f"data_engine/ui_environment_real/{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Create environment
    json_path = create_gelab_env_from_sequence(
        sequence_id=sequence_id,
        screenshots_dir=screenshots_dir,
        omni_json_dir=omni_json_dir,
        sequence_json_path=sequence_json_path,
        output_dir=output_dir
    )
    
    print(f"\nDone! Environment saved to: {output_dir}")
