"""
Interactive Environment for GeLab Navigation Evaluation.

This module provides an environment simulator that:
1. Tracks the current page state
2. Executes click actions based on coordinates
3. Returns screenshots for the current page
4. Supports home/back navigation
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class ClickResult:
    """Result of executing a click action."""
    success: bool
    new_page: str
    clicked_element: Optional[str] = None
    message: str = ""


class InteractiveEnvironment:
    """
    Interactive environment for GUI navigation tasks.
    
    The environment maintains:
    - Current page state
    - Page graph with transitions
    - Icon bounding boxes for click detection
    """
    
    def __init__(self, ui_structure_path: str, images_dir: str):
        """
        Initialize the environment.
        
        Args:
            ui_structure_path: Path to ui_structure.json
            images_dir: Directory containing page screenshots
        """
        self.ui_structure_path = ui_structure_path
        self.images_dir = images_dir
        
        # Load UI structure
        with open(ui_structure_path, 'r') as f:
            self.ui_data = json.load(f)
        
        self.pages = self.ui_data.get('pages', {})
        self.current_page = None
        self.history = []
        self.step_count = 0
        
        # Build navigation graph
        self._build_graph()
        
    def _build_graph(self):
        """Build navigation graph from UI structure."""
        self.graph = {}  # page -> {icon: target_page}
        self.parent = {}  # page -> parent_page
        
        for page_id, page_data in self.pages.items():
            self.graph[page_id] = {}
            
            for transition in page_data.get('transitions', []):
                action = transition.get('action')
                target = transition.get('target_page')
                
                if action and target:
                    self.graph[page_id][action] = target
                    
                    # Track parent relationships
                    if action == 'back':
                        self.parent[page_id] = target
    
    def reset(self, start_page: str) -> str:
        """
        Reset the environment to a starting page.
        
        Args:
            start_page: The page to start from
            
        Returns:
            Path to the screenshot of the starting page
        """
        if start_page not in self.pages:
            raise ValueError(f"Unknown page: {start_page}")
        
        self.current_page = start_page
        self.history = []
        self.step_count = 0
        
        return self.get_screenshot_path()
    
    def get_screenshot_path(self) -> str:
        """Get the path to the current page's screenshot."""
        page_data = self.pages.get(self.current_page, {})
        image_name = page_data.get('image', f'{self.current_page}.png')
        return os.path.join(self.images_dir, image_name)
    
    def get_current_page(self) -> str:
        """Get the current page ID."""
        return self.current_page
    
    def get_page_layout(self) -> Dict[str, Any]:
        """Get the layout (icons and bboxes) of the current page."""
        return self.pages.get(self.current_page, {}).get('layout', {})
    
    def get_available_actions(self) -> List[str]:
        """Get list of available actions (icon names) on current page."""
        return list(self.graph.get(self.current_page, {}).keys())
    
    def execute_click(self, x: int, y: int) -> ClickResult:
        """
        Execute a click action at the given coordinates.
        
        Args:
            x: X coordinate of the click
            y: Y coordinate of the click
            
        Returns:
            ClickResult with success status and new page
        """
        self.step_count += 1
        page_data = self.pages.get(self.current_page, {})
        layout = page_data.get('layout', {})
        
        # Find which element was clicked
        clicked_element = None
        for element_name, element_data in layout.items():
            bbox = element_data.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                if x1 <= x <= x2 and y1 <= y <= y2:
                    clicked_element = element_name
                    break
        
        if clicked_element is None:
            # Click missed all elements - stay on current page
            return ClickResult(
                success=False,
                new_page=self.current_page,
                clicked_element=None,
                message="Click missed all interactive elements"
            )
        
        # Check if this element has a transition
        transitions = self.graph.get(self.current_page, {})
        if clicked_element in transitions:
            target_page = transitions[clicked_element]
            
            # Record history
            self.history.append({
                'page': self.current_page,
                'action': clicked_element,
                'coords': (x, y),
                'target': target_page
            })
            
            # Transition to new page
            old_page = self.current_page
            self.current_page = target_page
            
            return ClickResult(
                success=True,
                new_page=target_page,
                clicked_element=clicked_element,
                message=f"Clicked {clicked_element} on {old_page}, navigated to {target_page}"
            )
        else:
            # Element exists but has no transition (shouldn't happen in well-formed data)
            return ClickResult(
                success=False,
                new_page=self.current_page,
                clicked_element=clicked_element,
                message=f"Clicked {clicked_element} but no transition defined"
            )
    
    def execute_action(self, action_type: str, x: int = 0, y: int = 0) -> ClickResult:
        """
        Execute an action (click or complete).
        
        Args:
            action_type: 'click' or 'complete'
            x: X coordinate (for click)
            y: Y coordinate (for click)
            
        Returns:
            ClickResult
        """
        if action_type == 'complete':
            return ClickResult(
                success=True,
                new_page=self.current_page,
                clicked_element=None,
                message="Agent declared task complete"
            )
        elif action_type == 'click':
            return self.execute_click(x, y)
        else:
            return ClickResult(
                success=False,
                new_page=self.current_page,
                clicked_element=None,
                message=f"Unknown action type: {action_type}"
            )
    
    def is_at_target(self, target_page: str) -> bool:
        """Check if currently at the target page."""
        return self.current_page == target_page
    
    def get_history_string(self) -> str:
        """Get formatted history string for model input."""
        if not self.history:
            return ""
        
        history_parts = []
        for i, h in enumerate(self.history):
            history_parts.append(f"step{i+1}: click {h['action']} on {h['page']}")
        
        return " ".join(history_parts)
    
    def get_step_count(self) -> int:
        """Get the number of steps taken."""
        return self.step_count


class TaskGenerator:
    """
    Generate navigation tasks for evaluation.
    
    Creates source-target pairs with specified path length distribution
    matching the paper's Table 6.
    """
    
    # Paper's task distribution (Table 6)
    PAPER_DISTRIBUTION = {
        1: 137,
        2: 147,
        3: 222,
        4: 324,
        5: 492,
        6: 456,
        7: 384
    }
    
    def __init__(self, ui_structure_path: str):
        """Initialize task generator."""
        with open(ui_structure_path, 'r') as f:
            self.ui_data = json.load(f)
        
        self.pages = self.ui_data.get('pages', {})
        self._build_graph()
        self._compute_shortest_paths()
    
    def _build_graph(self):
        """Build bidirectional navigation graph."""
        self.graph = {}  # page -> list of (neighbor, action)
        
        for page_id, page_data in self.pages.items():
            if page_id not in self.graph:
                self.graph[page_id] = []
            
            for transition in page_data.get('transitions', []):
                target = transition.get('target_page')
                action = transition.get('action')
                if target:
                    self.graph[page_id].append((target, action))
    
    def _compute_shortest_paths(self):
        """Compute shortest paths between all pairs of pages using BFS."""
        self.shortest_paths = {}
        
        for start_page in self.pages.keys():
            distances = {start_page: 0}
            parents = {start_page: None}
            queue = deque([start_page])
            
            while queue:
                current = queue.popleft()
                
                for neighbor, _ in self.graph.get(current, []):
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        parents[neighbor] = current
                        queue.append(neighbor)
            
            self.shortest_paths[start_page] = distances
    
    def get_path_length(self, source: str, target: str) -> int:
        """Get shortest path length between two pages."""
        if source not in self.shortest_paths:
            return -1
        return self.shortest_paths[source].get(target, -1)
    
    def generate_tasks(self, subtree_indices: List[int] = None, 
                       use_paper_distribution: bool = True,
                       seed: int = 42) -> List[Dict]:
        """
        Generate evaluation tasks.
        
        Args:
            subtree_indices: Which subtrees to use (0-4). None = all subtrees.
            use_paper_distribution: Match paper's task count per path length.
            seed: Random seed for reproducibility.
            
        Returns:
            List of task dictionaries with source, target, path_length.
        """
        random.seed(seed)
        
        # Identify pages in specified subtrees
        if subtree_indices is None:
            subtree_indices = [0, 1, 2, 3, 4]
        
        # Map subtree index to root page
        subtree_roots = {
            0: 'page_1',
            1: 'page_2', 
            2: 'page_3',
            3: 'page_4',
            4: 'page_5'
        }
        
        # Get pages in each subtree using BFS from root
        def get_subtree_pages(root):
            pages = set([root, 'page_0'])  # Include home page
            queue = deque([root])
            visited = set([root])
            
            while queue:
                current = queue.popleft()
                for neighbor, action in self.graph.get(current, []):
                    # Only follow forward edges (not home/back)
                    if action not in ['home', 'back'] and neighbor not in visited:
                        visited.add(neighbor)
                        pages.add(neighbor)
                        queue.append(neighbor)
            
            return pages
        
        valid_pages = set(['page_0'])
        for idx in subtree_indices:
            if idx in subtree_roots:
                valid_pages.update(get_subtree_pages(subtree_roots[idx]))
        
        valid_pages = list(valid_pages)
        
        # Group page pairs by path length
        pairs_by_length = {i: [] for i in range(1, 8)}
        
        for source in valid_pages:
            for target in valid_pages:
                if source != target:
                    length = self.get_path_length(source, target)
                    if 1 <= length <= 7:
                        pairs_by_length[length].append((source, target))
        
        # Sample tasks
        tasks = []
        task_id = 0
        
        if use_paper_distribution:
            distribution = self.PAPER_DISTRIBUTION
        else:
            # Use all available pairs
            distribution = {i: len(pairs_by_length[i]) for i in range(1, 8)}
        
        for path_length, target_count in distribution.items():
            available = pairs_by_length[path_length]
            
            if len(available) == 0:
                print(f"Warning: No pairs available for path length {path_length}")
                continue
            
            # Sample with replacement if needed
            if len(available) >= target_count:
                selected = random.sample(available, target_count)
            else:
                # Sample with replacement
                selected = random.choices(available, k=target_count)
            
            for source, target in selected:
                tasks.append({
                    'id': task_id,
                    'source': source,
                    'target': target,
                    'path_length': path_length,
                    'instruction': f"Navigate from {source} to {target}"
                })
                task_id += 1
        
        return tasks
    
    def save_tasks(self, tasks: List[Dict], output_path: str):
        """Save tasks to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(tasks, f, indent=2)
        print(f"Saved {len(tasks)} tasks to {output_path}")


if __name__ == "__main__":
    # Test the environment
    import sys
    
    ui_path = "demo/ui_structure.json"
    images_dir = "datas/images"
    
    if not os.path.exists(ui_path):
        print(f"UI structure not found at {ui_path}")
        sys.exit(1)
    
    # Test environment
    print("Testing InteractiveEnvironment...")
    env = InteractiveEnvironment(ui_path, images_dir)
    
    # Reset to page_0
    screenshot = env.reset('page_0')
    print(f"Reset to page_0, screenshot: {screenshot}")
    print(f"Available actions: {env.get_available_actions()}")
    
    # Test click on first available action
    layout = env.get_page_layout()
    if layout:
        first_icon = list(layout.keys())[0]
        bbox = layout[first_icon].get('bbox', [0, 0, 0, 0])
        x = (bbox[0] + bbox[2]) // 2
        y = (bbox[1] + bbox[3]) // 2
        
        result = env.execute_click(x, y)
        print(f"Clicked at ({x}, {y}): {result}")
    
    # Test task generator
    print("\nTesting TaskGenerator...")
    generator = TaskGenerator(ui_path)
    
    # Generate tasks for test subtree (sub4 = page_5)
    tasks = generator.generate_tasks(subtree_indices=[4], use_paper_distribution=True)
    print(f"Generated {len(tasks)} tasks for subtree 4")
    
    # Show distribution
    from collections import Counter
    dist = Counter(t['path_length'] for t in tasks)
    print("Task distribution by path length:")
    for length in sorted(dist.keys()):
        print(f"  Path@{length}: {dist[length]} tasks")
