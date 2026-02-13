# generate_ood_splits.py
import os
import json
import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Set

SYSTEM_ACTIONS = {"back", "home"}

def read_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def build_graph(pages: Dict) -> Dict[str, List[Tuple[str, str, List[int]]]]:
    graph = {}
    for page_id, page_data in pages.items():
        outs = []
        for t in page_data.get("transitions", []):
            outs.append((t["target_page"], t["action"], t.get("icon_bbox", [0, 0, 0, 0])))
        graph[page_id] = outs
    return graph

def normalize_bbox(bbox: List[int], canvas_w: int, canvas_h: int) -> List[int]:
    if not bbox or bbox == [0, 0, 0, 0]:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * 1000 / canvas_w),
        int(y1 * 1000 / canvas_h),
        int(x2 * 1000 / canvas_w),
        int(y2 * 1000 / canvas_h),
    ]

def bbox_center(norm_bbox: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = norm_bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def subtree_pages_from_layer(ui_layer: Dict, subtree_idx: int) -> Set[str]:
    root = ui_layer.get("root", {})
    subs = root.get("subnodes", [])
    if subtree_idx < 0 or subtree_idx >= len(subs):
        raise ValueError(f"subtree_idx={subtree_idx} out of range. Found {len(subs)} subnodes under root.")
    target_root = subs[subtree_idx]

    pages = set()

    def walk(node):
        img = node.get("image", "")
        if img.endswith(".png"):
            pages.add(img[:-4])
        for ch in node.get("subnodes", []) or []:
            walk(ch)

    walk(target_root)
    return pages

def subtree_pages_fallback(pages: Dict, graph: Dict, subtree_idx: int) -> Set[str]:
    # root children = page_0의 non-system outgoing targets (정렬 후 인덱스로 subtree 매핑)
    root_children = []
    for target, action, _bbox in graph.get("page_0", []):
        if action not in SYSTEM_ACTIONS and target != "page_0":
            root_children.append(target)

    root_children_sorted = sorted(root_children)
    if subtree_idx < 0 or subtree_idx >= len(root_children_sorted):
        raise ValueError(f"subtree_idx={subtree_idx} out of range. Found {len(root_children_sorted)} root children.")

    subtree_root = root_children_sorted[subtree_idx]

    seen = set()
    q = deque([subtree_root])
    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt, act, _bbox in graph.get(cur, []):
            if act in SYSTEM_ACTIONS:
                continue
            if nxt == "page_0":
                continue
            q.append(nxt)
    return seen

def find_shortest_path_edges(
    graph: Dict[str, List[Tuple[str, str, List[int]]]],
    start: str,
    end: str
) -> Optional[List[Tuple[str, str, List[int], str]]]:
    """
    Returns list of edges: (from_page, action, bbox, to_page) on a shortest path.
    """
    if start == end:
        return []

    q = deque([start])
    prev = {start: None}           # node -> previous node
    prev_edge = {}                 # node -> (from, action, bbox)

    while q:
        cur = q.popleft()
        for nxt, act, bbox in graph.get(cur, []):
            if nxt in prev:
                continue
            prev[nxt] = cur
            prev_edge[nxt] = (cur, act, bbox)
            if nxt == end:
                q.clear()
                break
            q.append(nxt)

    if end not in prev:
        return None

    # reconstruct nodes
    nodes = []
    cur = end
    while cur is not None:
        nodes.append(cur)
        cur = prev[cur]
    nodes.reverse()

    # nodes -> edges
    edges = []
    for i in range(1, len(nodes)):
        node = nodes[i]
        frm, act, bbox = prev_edge[node]
        edges.append((frm, act, bbox, node))
    return edges

def format_click(action: str, page_id: str, cx: int, cy: int) -> str:
    return f"Explain:click {action} icon on {page_id}.\tAction: click(start_box='<|box_start|>({cx},{cy})<|box_end|>')"

def format_complete() -> str:
    return "Explain: this is target page.\tAction: complete"

def make_edge_set(
    pages: Dict,
    graph: Dict[str, List[Tuple[str, str, List[int]]]],
    pages_dir: str,
    subtree_pages: List[str],
    canvas_w: int,
    canvas_h: int,
    include_system_actions: bool,
    max_edge_samples: Optional[int],
    seed: int,
) -> List[Dict]:
    random.seed(seed)
    out = []

    for src in subtree_pages:
        for tgt, act, bbox in graph.get(src, []):
            if (not include_system_actions) and (act in SYSTEM_ACTIONS):
                continue
            if tgt not in pages:
                continue

            norm_bbox = normalize_bbox(bbox, canvas_w, canvas_h)
            cx, cy = bbox_center(norm_bbox)

            out.append({
                "idx": len(out),
                "type": "edge",
                "path": 1,
                "task": f"From {src} to {tgt}",
                "bbox": bbox,
                "bbox_norm": norm_bbox,
                "messages": [
                    {"role": "user", "content": f"<image>Instruction: from {src} to {tgt}. History: Null"},
                    {"role": "assistant", "content": format_click(act, src, cx, cy)},
                ],
                "images": [os.path.join(pages_dir, f"{src}.png")],
            })

    if max_edge_samples is not None and len(out) > max_edge_samples:
        out = random.sample(out, k=max_edge_samples)
        for i, s in enumerate(out):
            s["idx"] = i

    return out

def make_path_set(
    pages: Dict,
    graph: Dict[str, List[Tuple[str, str, List[int]]]],
    pages_dir: str,
    subtree_pages: List[str],
    canvas_w: int,
    canvas_h: int,
    include_system_actions: bool,
    max_tasks: int,
    seed: int,
) -> Tuple[List[Dict], int, int]:
    """
    max_tasks: (start,end) task 개수 제한
    return: (samples, made_tasks, attempts)
    """
    random.seed(seed)
    out = []

    pages_list = list(subtree_pages)
    attempted_pairs = set()
    attempts = 0
    made_tasks = 0

    max_attempts = max(max_tasks * 50, 50000)

    while made_tasks < max_tasks and attempts < max_attempts:
        attempts += 1
        start = random.choice(pages_list)
        end = random.choice(pages_list)
        if start == end:
            continue

        key = (start, end)
        if key in attempted_pairs:
            continue
        attempted_pairs.add(key)

        edges = find_shortest_path_edges(graph, start, end)
        if edges is None or len(edges) == 0:
            continue

        # system action 포함 여부
        if not include_system_actions:
            if any(act in SYSTEM_ACTIONS for (_frm, act, _bbox, _to) in edges):
                continue

        path_len = len(edges)
        history_parts = []
        cur_page = start

        # step samples (click)
        for step_idx, (frm, act, bbox, to) in enumerate(edges):
            # frm은 항상 cur_page여야 정상
            cur_page = frm

            norm_bbox = normalize_bbox(bbox, canvas_w, canvas_h)
            cx, cy = bbox_center(norm_bbox)

            history = "; ".join(history_parts) if history_parts else "Null"

            out.append({
                "idx": len(out),
                "type": "path",
                "path": path_len,
                "step": step_idx + 1,
                "task": f"From {start} to {end}",
                "current_page": cur_page,
                "bbox": bbox,
                "bbox_norm": norm_bbox,
                "messages": [
                    {"role": "user", "content": f"<image>Instruction: from {start} to {end}. History: {history}"},
                    {"role": "assistant", "content": format_click(act, cur_page, cx, cy)},
                ],
                "images": [os.path.join(pages_dir, f"{cur_page}.png")],
            })

            history_parts.append(f"step{step_idx + 1}: click {act} icon on {cur_page}")
            cur_page = to

        # final sample (complete) at end
        final_history = "; ".join(history_parts)
        out.append({
            "idx": len(out),
            "type": "path",
            "path": path_len,
            "step": path_len + 1,
            "task": f"From {start} to {end}",
            "current_page": end,
            "bbox": [0, 0, 0, 0],
            "bbox_norm": [0, 0, 0, 0],
            "messages": [
                {"role": "user", "content": f"<image>Instruction: from {start} to {end}. History: {final_history}"},
                {"role": "assistant", "content": format_complete()},
            ],
            "images": [os.path.join(pages_dir, f"{end}.png")],
        })

        made_tasks += 1

    # idx 재정렬(이미 순차지만 안전하게)
    for i, s in enumerate(out):
        s["idx"] = i

    return out, made_tasks, attempts

def make_for_env(
    env_dir: str,
    out_dir: str,
    subtree_idx: int,
    max_path_tasks: int,
    max_edge_samples: Optional[int],
    seed: int,
    include_system_actions: bool,
    tag: str,
):
    ui_structure_path = os.path.join(env_dir, "ui_structure.json")
    ui_layer_path = os.path.join(env_dir, "ui_structure_layer.json")
    cfg_path = os.path.join(env_dir, "config.json")
    pages_dir = os.path.join(env_dir, "pages")

    if not os.path.exists(ui_structure_path):
        raise FileNotFoundError(ui_structure_path)
    if not os.path.isdir(pages_dir):
        raise FileNotFoundError(pages_dir)

    structure = read_json(ui_structure_path)
    pages = structure["pages"]
    graph = build_graph(pages)

    # canvas size
    canvas_w, canvas_h = 1179, 2556
    if os.path.exists(cfg_path):
        cfg = read_json(cfg_path)
        cs = cfg.get("canvas_size")
        if isinstance(cs, list) and len(cs) == 2:
            canvas_w, canvas_h = int(cs[0]), int(cs[1])

    # subtree pages
    if os.path.exists(ui_layer_path):
        ui_layer = read_json(ui_layer_path)
        subtree_set = subtree_pages_from_layer(ui_layer, subtree_idx)
    else:
        subtree_set = subtree_pages_fallback(pages, graph, subtree_idx)

    subtree_set = {p for p in subtree_set if p in pages}
    if not subtree_set:
        raise RuntimeError(f"No pages found for subtree {subtree_idx} in {env_dir}")

    subtree_list = sorted(subtree_set)

    # EDGE
    edge_items = make_edge_set(
        pages=pages,
        graph=graph,
        pages_dir=pages_dir,
        subtree_pages=subtree_list,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        include_system_actions=include_system_actions,
        max_edge_samples=max_edge_samples,
        seed=seed,
    )

    # PATH
    path_items, made_tasks, attempts = make_path_set(
        pages=pages,
        graph=graph,
        pages_dir=pages_dir,
        subtree_pages=subtree_list,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        include_system_actions=include_system_actions,
        max_tasks=max_path_tasks,
        seed=seed,
    )

    os.makedirs(out_dir, exist_ok=True)

    edge_out = os.path.join(out_dir, f"ood_test_{tag}_edge.json")
    path_out = os.path.join(out_dir, f"ood_test_{tag}_path.json")

    with open(edge_out, "w", encoding="utf-8") as f:
        json.dump(edge_items, f, indent=2, ensure_ascii=False)
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(path_items, f, indent=2, ensure_ascii=False)

    print(f"[OK] {os.path.basename(env_dir)}")
    print(f"  EDGE -> {edge_out} | samples={len(edge_items)} | canvas={canvas_w}x{canvas_h} | system_actions={'ON' if include_system_actions else 'OFF'}")
    print(f"  PATH -> {path_out} | samples={len(path_items)} | tasks_made={made_tasks}/{max_path_tasks} | attempts={attempts} | canvas={canvas_w}x{canvas_h} | system_actions={'ON' if include_system_actions else 'OFF'}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood_root", required=True, help="e.g., ui_environment_ood/20260212_164821")
    ap.add_argument("--out_dir", default="datas", help="where to write ood_test_*.json")
    ap.add_argument("--subtree", type=int, default=4, help="subtree index reserved for OOD eval (default: 4)")
    ap.add_argument("--max_path_tasks", type=int, default=2200, help="number of (start,end) tasks for PATH set")
    ap.add_argument("--max_edge_samples", type=int, default=0, help="0 means no limit; otherwise sample down to this many")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_system_actions", action="store_true", help="include back/home in edge/path generation")
    args = ap.parse_args()

    max_edge = None if args.max_edge_samples == 0 else int(args.max_edge_samples)

    env_map = {
        "Base": "Env-Base",
        "Image": "Env-Image",
        "Name": "Env-Name",
        "Position": "Env-Position",
        "Noise": "Env-Noise",
    }

    for tag, folder in env_map.items():
        env_dir = os.path.join(args.ood_root, folder)
        if not os.path.isdir(env_dir):
            print(f"[SKIP] Missing dir: {env_dir}")
            continue

        make_for_env(
            env_dir=env_dir,
            out_dir=args.out_dir,
            subtree_idx=args.subtree,
            max_path_tasks=args.max_path_tasks,
            max_edge_samples=max_edge,
            seed=args.seed,
            include_system_actions=args.include_system_actions,
            tag=tag,
        )

if __name__ == "__main__":
    main()
