"""
Shared utilities for GE-Lab data generation scripts.

Provides GELabEnvUtils base class with:
- UI structure loading (ui_structure.json, ui_structure_layer.json)
- Topology grouping metadata (legacy subtrees or GT-step groupings)
- Navigation graph and NetworkX graph construction
- Bbox normalization (448 -> 1000) and center computation
- Shortest path finding with action/bbox info
- Action formatting (click/complete strings)
"""

import json
import os
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple


class GELabEnvUtils:
    """Base class with shared GE-Lab environment utilities."""

    CANVAS_WIDTH = 448
    CANVAS_HEIGHT = 448
    NORM_SIZE = 1000

    def __init__(self, env_dir: str):
        self.env_dir = env_dir
        self.pages_dir = os.path.join(env_dir, "pages")

        # Load UI structure
        ui_structure_path = os.path.join(env_dir, "ui_structure.json")
        with open(ui_structure_path) as f:
            data = json.load(f)
        self.pages = data["pages"]
        self.metadata = data.get("metadata", {})

        # Read canvas size from metadata if available (sim2real envs may differ)
        canvas = self.metadata.get("canvas_size")
        if canvas:
            self.CANVAS_WIDTH = canvas[0]
            self.CANVAS_HEIGHT = canvas[1]

        # Build page-to-subtree mapping from layer structure
        ui_layer_path = os.path.join(env_dir, "ui_structure_layer.json")
        self.page_to_subtree = self._build_subtree_mapping(ui_layer_path)

        # Build navigation graph (full with back/home)
        self.graph = self._build_graph()

        # Build NetworkX graph for path finding
        self.nx_graph = self._build_nx_graph()

        print(f"Loaded {len(self.pages)} pages")
        print(f"Topology distribution: {self._count_subtrees()}")

    def _build_subtree_mapping(self, ui_layer_path: str) -> Dict[str, int]:
        """Build page grouping metadata for either legacy or GT-spine environments."""
        if self.metadata.get("topology_type") == "gt_spine_branches":
            page_to_group = {}
            for page_id, page_data in self.pages.items():
                page_to_group[page_id] = int(page_data.get("source_step_index", 0))
            return page_to_group

        with open(ui_layer_path) as f:
            ui = json.load(f)

        root = ui.get("root", {})
        page_to_subtree = {"page_0": -1}  # root is special

        def collect_pages(node, subtree_idx):
            img = node.get("image", "")
            page_id = img.replace(".png", "") if img else None
            if page_id:
                page_to_subtree[page_id] = subtree_idx
            for child in node.get("subnodes", []):
                collect_pages(child, subtree_idx)

        for i, child in enumerate(root.get("subnodes", [])):
            collect_pages(child, i)

        return page_to_subtree

    def _count_subtrees(self) -> Dict[int, int]:
        """Count pages per topology group."""
        counts = defaultdict(int)
        for page_id, subtree in self.page_to_subtree.items():
            counts[subtree] += 1
        return dict(counts)

    def _build_graph(self) -> Dict[str, List[Tuple[str, str, List[int]]]]:
        """Build navigation graph INCLUDING back/home for full connectivity."""
        graph = {}
        for page_id, page_data in self.pages.items():
            graph[page_id] = []
            for trans in page_data.get("transitions", []):
                action = trans["action"]
                target = trans["target_page"]
                bbox = trans.get("icon_bbox", [0, 0, 0, 0])
                graph[page_id].append((target, action, bbox))
        return graph

    def _build_nx_graph(self) -> nx.DiGraph:
        """Build NetworkX graph for efficient path finding."""
        G = nx.DiGraph()
        for page_id, page_data in self.pages.items():
            G.add_node(page_id, depth=page_data.get("depth", 0))
            for trans in page_data.get("transitions", []):
                action = trans["action"]
                target = trans["target_page"]
                bbox = trans.get("icon_bbox", [0, 0, 0, 0])
                G.add_edge(page_id, target, action=action, bbox=bbox)
        return G

    def bbox_to_normalized(self, bbox: List[int]) -> List[int]:
        """Convert pixel bbox to normalized 0-1000 coordinates."""
        if not bbox or bbox == [0, 0, 0, 0]:
            return [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        return [
            int(x1 * self.NORM_SIZE / self.CANVAS_WIDTH),
            int(y1 * self.NORM_SIZE / self.CANVAS_HEIGHT),
            int(x2 * self.NORM_SIZE / self.CANVAS_WIDTH),
            int(y2 * self.NORM_SIZE / self.CANVAS_HEIGHT),
        ]

    def bbox_center_normalized(self, bbox: List[int]) -> Tuple[int, int]:
        """Get center of bbox in normalized coordinates."""
        norm = self.bbox_to_normalized(bbox)
        cx = (norm[0] + norm[2]) // 2
        cy = (norm[1] + norm[3]) // 2
        return cx, cy

    def format_click_action(self, action_name: str, page_id: str, bbox: List[int]) -> str:
        """Format click action string."""
        cx, cy = self.bbox_center_normalized(bbox)
        return f"Explain: click {action_name} icon on {page_id}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"

    def format_complete_action(self) -> str:
        """Format complete action string."""
        return "Explain: this is target page.\tAction: complete"

    def get_pages_in_subtrees(self, subtree_indices: List[int]) -> List[str]:
        """Get all pages belonging to specified topology groups (sorted)."""
        return sorted(p for p, s in self.page_to_subtree.items() if s in subtree_indices)

    def get_sorted_page_ids(self) -> List[str]:
        """Return page ids sorted by numeric suffix when possible."""
        def sort_key(page_id: str):
            try:
                return int(page_id.split("_")[1])
            except Exception:
                return page_id

        return sorted(self.pages.keys(), key=sort_key)

    def get_public_transitions(self, page_id: str) -> List[Tuple[str, str, List[int]]]:
        """Return non-system transitions used for training."""
        return [
            (target, action, bbox)
            for target, action, bbox in self.graph.get(page_id, [])
            if action not in {"back", "home"}
        ]

    def get_path_with_actions(self, start: str, end: str) -> List[Tuple[str, str, List[int]]]:
        """Get shortest path with action and bbox info. Returns [] if no path."""
        try:
            path_nodes = nx.shortest_path(self.nx_graph, start, end)
        except nx.NetworkXNoPath:
            return []

        if len(path_nodes) < 2:
            return []

        path = []
        for i in range(len(path_nodes) - 1):
            from_node = path_nodes[i]
            to_node = path_nodes[i + 1]
            edge_data = self.nx_graph.get_edge_data(from_node, to_node)
            if edge_data:
                path.append((from_node, edge_data["action"], edge_data["bbox"]))

        return path
