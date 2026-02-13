import os
import json
import random
import shutil
import glob
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from PIL import Image, ImageDraw, ImageFont

# ----------------------------
# Utils
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_png(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")

def save_png(img: Image.Image, path: str):
    img.save(path)

def create_system_icon(
    text: str,
    bg_color: Tuple[int,int,int],
    size: Tuple[int,int]
) -> Image.Image:
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)

    font_size = max(10, size[0] // 4)
    try:
        font = ImageFont.truetype("font/helvetica.ttf", font_size)
    except:
        font = None

    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size[0] - w) // 2
    y = (size[1] - h) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return img.convert("RGBA")

def rects_overlap(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def any_overlap(rect, rects) -> bool:
    return any(rects_overlap(rect, r) for r in rects)

# ----------------------------
# Config / Layout
# ----------------------------
@dataclass
class CanvasConfig:
    canvas_size: Tuple[int,int] = (448, 448)
    icon_size: Tuple[int,int] = (50, 50)
    margin: int = 20
    top_margin: int = 50

def generate_grid_positions(cfg: CanvasConfig) -> List[Tuple[int,int]]:
    W, H = cfg.canvas_size
    icon_w, icon_h = cfg.icon_size
    usable_w = W - 2 * cfg.margin
    usable_h = H - cfg.top_margin - cfg.margin

    reserved_top = cfg.top_margin + icon_h + 50
    min_spacing_x = icon_w + 30
    min_spacing_y = icon_h + 30

    num_cols = usable_w // min_spacing_x
    num_rows = (usable_h - reserved_top) // min_spacing_y

    num_cols = min(num_cols, 5)
    num_rows = min(num_rows, 8)

    start_x = cfg.margin + (usable_w - (num_cols - 1) * min_spacing_x) // 2
    start_y = reserved_top + (usable_h - reserved_top - (num_rows - 1) * min_spacing_y) // 2

    pos = []
    for r in range(num_rows):
        for c in range(num_cols):
            x = start_x + c * min_spacing_x
            y = start_y + r * min_spacing_y
            pos.append((int(x), int(y)))
    return pos

# ----------------------------
# Icon sources
# ----------------------------
def load_used_icons_list(base_env_dir: str) -> List[dict]:
    used_path = os.path.join(base_env_dir, "used_icons.json")
    if not os.path.exists(used_path):
        print(f"[WARN] used_icons.json not found at: {used_path}")
        return []
    return read_json(used_path)

def load_used_icons_map(base_env_dir: str) -> Dict[str, str]:
    used = load_used_icons_list(base_env_dir)
    return {x["func_name"]: x["original_path"] for x in used}

def write_used_icons_list(out_dir: str, used_list: List[dict]):
    write_json(os.path.join(out_dir, "used_icons.json"), used_list)

def load_alt_icons_from_dir(alt_icons_dir: str) -> List[str]:
    paths = glob.glob(os.path.join(alt_icons_dir, "**", "*.png"), recursive=True)
    paths = [p for p in paths if os.path.isfile(p)]
    paths.sort()
    if len(paths) == 0:
        raise ValueError(f"No .png found under alt_icons_dir: {alt_icons_dir}")
    return paths

# ----------------------------
# JSON schema helpers
# ----------------------------
def clone_base_structure(base_struct: dict) -> dict:
    return json.loads(json.dumps(base_struct))

def extract_actions_from_page(page_info: dict) -> List[str]:
    return [k for k in page_info["layout"].keys() if k != "page_title"]

def is_system_action(action: str) -> bool:
    return action in ["back", "home"]

def is_noise_action(action: str, noise_prefix="__noise_") -> bool:
    return action.startswith(noise_prefix)

# ----------------------------
# Rendering from structure
# ----------------------------
def render_page_from_structure(
    page_id: str,
    page_info: dict,
    cfg: CanvasConfig,
    action_to_iconimg: Dict[str, Image.Image]
) -> Image.Image:
    W, H = cfg.canvas_size
    icon_w, icon_h = cfg.icon_size

    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_width = cfg.canvas_size[0] // 2
    x1 = (cfg.canvas_size[0] - title_width) // 2
    y1 = cfg.margin
    title_bbox = [x1, y1, x1 + title_width, y1 + cfg.icon_size[1]]
    title_text = page_id
    try:
        font_title = ImageFont.truetype("font/helvetica.ttf", 24)
    except:
        font_title = None

    tb = draw.textbbox((0,0), title_text, font=font_title)
    tw, th = tb[2]-tb[0], tb[3]-tb[1]
    tx = title_bbox[0] + (title_bbox[2]-title_bbox[0]-tw)//2
    ty = title_bbox[1] + (title_bbox[3]-title_bbox[1]-th)//2
    draw.text((tx, ty), title_text, fill=(0,0,0,255), font=font_title)

    for action, info in page_info["layout"].items():
        if action == "page_title":
            continue
        bbox = info["bbox"]
        x1, y1, x2, y2 = bbox
        icon = action_to_iconimg.get(action)
        if icon is None:
            continue
        icon = icon.resize((icon_w, icon_h))

        black_bg_icon = Image.new("RGBA", (icon_w, icon_h), (0, 0, 0, 255))
        black_bg_icon.alpha_composite(icon)
        img.paste(black_bg_icon, (x1, y1))

    return img.convert("RGB")

# ----------------------------
# Layer.json regeneration (match original generator's intent)
# ----------------------------
def build_layered_structure_from_pages(pages: dict) -> dict:
    # parent -> children map (normal transitions only)
    parent_child_map = {pid: [] for pid in pages.keys()}
    for pid, info in pages.items():
        for t in info.get("transitions", []):
            if t["action"] not in ["back", "home"]:
                parent_child_map[pid].append(t["target_page"])

    visited = set()

    def create_node_tree(node_id: str):
        if node_id in visited:
            return None
        visited.add(node_id)

        node_data = json.loads(json.dumps(pages[node_id]))  # deep copy
        node_data["transitions"] = [
            t for t in node_data.get("transitions", [])
            if t["action"] not in ["back", "home"]
        ]

        subnodes = []
        for child_id in parent_child_map.get(node_id, []):
            if child_id != node_id:
                child_tree = create_node_tree(child_id)
                if child_tree is not None:
                    subnodes.append(child_tree)

        node_data["subnodes"] = subnodes
        return node_data

    root_id = "page_0"
    return {
        "root": create_node_tree(root_id),
        "metadata": {
            "total_pages": len(pages),
        }
    }

# ----------------------------
# Name remapping helpers (keys + transitions + used_icons)
# ----------------------------
def apply_name_map(s: str, name_map: Dict[str, str]) -> str:
    out = s
    for k, v in name_map.items():
        if k in out:
            out = out.replace(k, v)
    return out

def remap_actions_in_ui_structure(ood: dict, name_map: Dict[str, str], noise_prefix="__noise_") -> Tuple[dict, Dict[str, str]]:
    """
    실제 func_name(action key)를 바꾼다:
    - pages[*].layout의 key 변경
    - transitions[*].action 변경
    - 충돌(동일 이름 2개가 같은 새 이름으로 매핑) 발생 시 __dupN suffix로 deterministic하게 처리
    반환: (updated_ood, old_to_new_action_map)
    """
    old_to_new: Dict[str, str] = {}

    # 1) 전체 action set 수집 (normal만)
    all_actions = set()
    for _, page_info in ood["pages"].items():
        for action in page_info["layout"].keys():
            if action in ["page_title", "back", "home"]:
                continue
            if is_noise_action(action, noise_prefix=noise_prefix):
                continue
            all_actions.add(action)

    # 2) 1차 매핑(단순 치환)
    proposed = {a: apply_name_map(a, name_map) for a in sorted(all_actions)}

    # 3) 충돌 해결 (stable)
    used_new = {}
    for old in sorted(all_actions):
        base_new = proposed[old]
        if base_new not in used_new:
            used_new[base_new] = 0
            new = base_new
        else:
            used_new[base_new] += 1
            new = f"{base_new}__dup{used_new[base_new]}"
        old_to_new[old] = new

    # 4) layout key 리네임 + display_name도 같이 세팅
    for page_id, page_info in ood["pages"].items():
        layout = page_info["layout"]
        new_layout = {}

        for action, info in layout.items():
            if action == "page_title" or is_system_action(action) or is_noise_action(action, noise_prefix=noise_prefix):
                new_layout[action] = info
                continue

            new_action = old_to_new.get(action, action)
            new_info = json.loads(json.dumps(info))
            # display_name은 "사람용 라벨"이라면, 새 action으로 맞춰두는 게 안전
            new_info["display_name"] = new_action
            new_layout[new_action] = new_info

        page_info["layout"] = new_layout

        # transitions action rename + icon_bbox sync
        for t in page_info.get("transitions", []):
            a = t.get("action")
            if a in old_to_new:
                t["action"] = old_to_new[a]
            # icon_bbox는 "현재 layout의 bbox"로 재동기화
            a2 = t.get("action")
            if a2 in page_info["layout"]:
                t["icon_bbox"] = page_info["layout"][a2]["bbox"]

    return ood, old_to_new

def remap_used_icons_list(used_list: List[dict], old_to_new: Dict[str, str]) -> List[dict]:
    new_list = []
    for item in used_list:
        fn = item.get("func_name")
        new_item = json.loads(json.dumps(item))
        if fn in old_to_new:
            new_item["func_name"] = old_to_new[fn]
        new_list.append(new_item)

    # 혹시 func_name 중복이 생겼다면 마지막 방어(드물지만)
    seen = {}
    fixed = []
    for it in new_list:
        fn = it["func_name"]
        if fn not in seen:
            seen[fn] = 0
            fixed.append(it)
        else:
            seen[fn] += 1
            it2 = json.loads(json.dumps(it))
            it2["func_name"] = f"{fn}__dup{seen[fn]}"
            fixed.append(it2)
    return fixed

# ----------------------------
# Env-Image icon mapping helpers
# ----------------------------
def _extract_category_and_filename(original_path: str) -> tuple[str, str]:
    p = original_path.replace("\\", "/")
    parts = p.split("/")
    filename = parts[-1]
    if len(parts) >= 3 and parts[-2].lower() == "png":
        category = parts[-3]
    else:
        category = parts[-2] if len(parts) >= 2 else "UNKNOWN"
    return category, filename

def _find_matching_alt_icon(alt_root: str, category: str, filename: str) -> str | None:
    cand1 = os.path.join(alt_root, category, filename)
    if os.path.exists(cand1): return cand1
    cand2 = os.path.join(alt_root, category, "PNG", filename)
    if os.path.exists(cand2): return cand2
    pattern = os.path.join(alt_root, category, "**", filename)
    hits = glob.glob(pattern, recursive=True)
    hits = [h for h in hits if os.path.isfile(h)]
    hits.sort()
    if hits: return hits[0]
    return None

# ----------------------------
# OOD builders
# ----------------------------
def copy_config_only(base_env_dir: str, out_dir: str):
    src = os.path.join(base_env_dir, "config.json")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(out_dir, "config.json"))

def build_env_base(base_env_dir: str, out_dir: str):
    ensure_dir(out_dir)
    for name in ["config.json", "ui_structure.json", "ui_structure_layer.json", "used_icons.json"]:
        src = os.path.join(base_env_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))

    src_pages = os.path.join(base_env_dir, "pages")
    dst_pages = os.path.join(out_dir, "pages")
    if os.path.exists(src_pages):
        shutil.copytree(src_pages, dst_pages, dirs_exist_ok=True)

    print(f"[OK] Env-Base saved to {out_dir}")

def build_env_name(base_env_dir: str, out_dir: str, name_map: Dict[str,str], noise_prefix="__noise_"):
    """
    Env-Name (정합 버전):
    - action(func_name) 자체를 바꾼다 (layout key + transitions action)
    - used_icons.json도 func_name에 맞춰 업데이트
    - ui_structure_layer.json은 ui_structure.json으로부터 재생성
    - 이미지에 텍스트 렌더링을 안 한다는 전제면 pages는 복사 가능
    """
    ensure_dir(out_dir)

    base_struct = read_json(os.path.join(base_env_dir, "ui_structure.json"))
    used_list = load_used_icons_list(base_env_dir)

    ood = clone_base_structure(base_struct)
    ood, old_to_new = remap_actions_in_ui_structure(ood, name_map, noise_prefix=noise_prefix)

    # metadata
    ood.setdefault("metadata", {})
    ood["metadata"]["ood_mode"] = "name"
    ood["metadata"]["ood_name_map"] = name_map

    # used_icons 업데이트
    new_used_list = remap_used_icons_list(used_list, old_to_new)

    # config만 복사 (불변)
    copy_config_only(base_env_dir, out_dir)

    # write structures
    write_json(os.path.join(out_dir, "ui_structure.json"), ood)

    layer = build_layered_structure_from_pages(ood["pages"])
    layer["metadata"].update(ood.get("metadata", {}))
    write_json(os.path.join(out_dir, "ui_structure_layer.json"), layer)

    write_used_icons_list(out_dir, new_used_list)

    # pages copy (no text rendering)
    src_pages = os.path.join(base_env_dir, "pages")
    dst_pages = os.path.join(out_dir, "pages")
    if os.path.exists(src_pages):
        shutil.copytree(src_pages, dst_pages, dirs_exist_ok=True)

    print(f"[OK] Env-Name -> {out_dir}")

def build_env_image(base_env_dir: str, out_dir: str, alt_icons_dir: str, seed: int = 0):
    """
    Env-Image (정합 버전):
    - ui_structure.json은 구조/이름/좌표 유지(필요시 metadata만 추가)
    - used_icons.json은 '현재 실제로 쓰는 이미지 경로'로 업데이트 (요구사항 반영)
    - ui_structure_layer.json은 ui_structure.json으로부터 재생성(안전)
    - pages는 alt 아이콘으로 재렌더
    """
    ensure_dir(out_dir)

    base_struct = read_json(os.path.join(base_env_dir, "ui_structure.json"))
    cfg_json = read_json(os.path.join(base_env_dir, "config.json"))
    cfg = CanvasConfig(
        canvas_size=tuple(cfg_json.get("canvas_size", [1179, 2556])),
        icon_size=tuple(cfg_json.get("icon_size", [200, 200])),
        margin=cfg_json.get("margin", 60),
        top_margin=cfg_json.get("top_margin", 150),
    )

    used_list = load_used_icons_list(base_env_dir)
    used_map = {x["func_name"]: x["original_path"] for x in used_list}
    if len(used_map) == 0:
        raise ValueError("Env-Image needs used_icons.json to recover per-action original icon paths.")

    ood = clone_base_structure(base_struct)
    ood.setdefault("metadata", {})
    ood["metadata"]["ood_mode"] = "image"
    ood["metadata"]["alt_icons_dir"] = alt_icons_dir
    ood["metadata"]["seed"] = seed

    action_to_iconimg: Dict[str, Image.Image] = {}
    action_to_iconimg["back"] = create_system_icon("back", (255, 200, 200), size=cfg.icon_size)
    action_to_iconimg["home"] = create_system_icon("home", (200, 255, 200), size=cfg.icon_size)

    rng = random.Random(seed)
    missing = []
    updated_used_list = []

    for item in used_list:
        act = item["func_name"]
        orig_path = item["original_path"]

        category, filename = _extract_category_and_filename(orig_path)
        alt_path = _find_matching_alt_icon(alt_icons_dir, category, filename)

        if alt_path is None:
            pattern = os.path.join(alt_icons_dir, category, "**", "*.png")
            hits = [h for h in glob.glob(pattern, recursive=True) if os.path.isfile(h)]
            hits.sort()
            if hits:
                alt_path = rng.choice(hits)
            else:
                missing.append((act, orig_path))
                alt_path = orig_path  # fallback keep original

        # 렌더용 이미지 등록
        if os.path.exists(alt_path):
            action_to_iconimg[act] = load_png(alt_path)

        # used_icons는 "현재 실제 쓰는 경로"로 업데이트
        updated_used_list.append({
            "func_name": act,
            "original_path": alt_path
        })

    if missing:
        print("[WARN] Some actions had no matching alt icon under alt_icons_dir; kept original paths for them.")

    # config만 복사
    copy_config_only(base_env_dir, out_dir)

    # write ui_structure + regenerated layer + updated used_icons
    write_json(os.path.join(out_dir, "ui_structure.json"), ood)
    layer = build_layered_structure_from_pages(ood["pages"])
    layer["metadata"].update(ood.get("metadata", {}))
    write_json(os.path.join(out_dir, "ui_structure_layer.json"), layer)
    write_used_icons_list(out_dir, updated_used_list)

    # render pages
    dst_pages = os.path.join(out_dir, "pages")
    ensure_dir(dst_pages)
    for page_id, page_info in ood["pages"].items():
        img = render_page_from_structure(page_id, page_info, cfg, action_to_iconimg)
        save_png(img, os.path.join(dst_pages, f"{page_id}.png"))

    print(f"[OK] Env-Image -> {out_dir}")

def build_env_position(base_env_dir: str, out_dir: str, seed: int = 0):
    """
    Env-Position (정합 버전):
    - ui_structure.json bbox + transitions.icon_bbox 업데이트
    - ui_structure_layer.json은 재생성
    - used_icons.json은 그대로(이름/경로 변화 없음)
    - pages 재렌더
    """
    ensure_dir(out_dir)

    base_struct = read_json(os.path.join(base_env_dir, "ui_structure.json"))
    cfg_json = read_json(os.path.join(base_env_dir, "config.json"))
    cfg = CanvasConfig(
        canvas_size=tuple(cfg_json.get("canvas_size", [1179,2556])),
        icon_size=tuple(cfg_json.get("icon_size", [200,200])),
        margin=cfg_json.get("margin", 60),
        top_margin=cfg_json.get("top_margin", 150),
    )

    used_list = load_used_icons_list(base_env_dir)
    used_map = {x["func_name"]: x["original_path"] for x in used_list}
    if len(used_map) == 0:
        raise ValueError("Env-Position needs used_icons.json to re-render icons.")

    rng = random.Random(seed)
    grid = generate_grid_positions(cfg)

    action_to_iconimg: Dict[str, Image.Image] = {}
    action_to_iconimg["back"] = create_system_icon("back", (255,200,200), size=cfg.icon_size)
    action_to_iconimg["home"] = create_system_icon("home", (200,255,200), size=cfg.icon_size)

    for func, p in used_map.items():
        if os.path.exists(p):
            action_to_iconimg[func] = load_png(p)

    ood = clone_base_structure(base_struct)
    ood.setdefault("metadata", {})
    ood["metadata"]["ood_mode"] = "position"
    ood["metadata"]["seed"] = seed

    for _, page_info in ood["pages"].items():
        actions = [
            a for a in extract_actions_from_page(page_info)
            if (not is_system_action(a))
            and a != "page_title"
            and (not a.startswith("__noise_"))
        ]
        rng.shuffle(grid)
        selected = rng.sample(grid, k=len(actions)) if len(grid) >= len(actions) else [rng.choice(grid) for _ in range(len(actions))]

        icon_w, icon_h = cfg.icon_size
        for act, (x, y) in zip(actions, selected):
            page_info["layout"][act]["bbox"] = [x, y, x + icon_w, y + icon_h]

        for t in page_info.get("transitions", []):
            a = t["action"]
            if a in page_info["layout"]:
                t["icon_bbox"] = page_info["layout"][a]["bbox"]

    # config만 복사 + used_icons는 그대로 복사(동일)
    copy_config_only(base_env_dir, out_dir)
    write_used_icons_list(out_dir, used_list)

    # write structures
    write_json(os.path.join(out_dir, "ui_structure.json"), ood)
    layer = build_layered_structure_from_pages(ood["pages"])
    layer["metadata"].update(ood.get("metadata", {}))
    write_json(os.path.join(out_dir, "ui_structure_layer.json"), layer)

    # render pages
    dst_pages = os.path.join(out_dir, "pages")
    ensure_dir(dst_pages)
    for page_id, page_info in ood["pages"].items():
        img = render_page_from_structure(page_id, page_info, cfg, action_to_iconimg)
        save_png(img, os.path.join(dst_pages, f"{page_id}.png"))

    print(f"[OK] Env-Position -> {out_dir}")

def build_env_noise(
    base_env_dir: str,
    out_dir: str,
    noise_icons_dir: str,
    noise_k: int = 2,
    seed: int = 0,
    noise_prefix: str = "__noise_"
):
    """
    Env-Noise (정합 버전):
    - ui_structure.json에 noise action 추가
    - ui_structure_layer.json 재생성(최소한 layout/bbox 정합)
    - used_icons.json은 기본 유지 + noise도 기록(재현 가능/렌더 가능)
    - pages 재렌더
    """
    ensure_dir(out_dir)

    base_struct = read_json(os.path.join(base_env_dir, "ui_structure.json"))
    cfg_json = read_json(os.path.join(base_env_dir, "config.json"))
    cfg = CanvasConfig(
        canvas_size=tuple(cfg_json.get("canvas_size", [1179,2556])),
        icon_size=tuple(cfg_json.get("icon_size", [200,200])),
        margin=cfg_json.get("margin", 60),
        top_margin=cfg_json.get("top_margin", 150),
    )

    used_list = load_used_icons_list(base_env_dir)
    used_map = {x["func_name"]: x["original_path"] for x in used_list}
    if len(used_map) == 0:
        raise ValueError("Env-Noise needs used_icons.json to re-render icons.")

    rng = random.Random(seed)
    grid = generate_grid_positions(cfg)

    noise_paths = load_alt_icons_from_dir(noise_icons_dir)
    rng.shuffle(noise_paths)
    if len(noise_paths) < noise_k:
        print(f"[WARN] noise icons < k, will recycle. noise={len(noise_paths)}, k={noise_k}")

    action_to_iconimg: Dict[str, Image.Image] = {}
    action_to_iconimg["back"] = create_system_icon("back", (255,200,200), size=cfg.icon_size)
    action_to_iconimg["home"] = create_system_icon("home", (200,255,200), size=cfg.icon_size)

    for func, p in used_map.items():
        if os.path.exists(p):
            action_to_iconimg[func] = load_png(p)

    noise_actions = [f"{noise_prefix}{i}" for i in range(noise_k)]
    noise_used_entries = []
    for i, na in enumerate(noise_actions):
        p = noise_paths[i % len(noise_paths)]
        action_to_iconimg[na] = load_png(p)
        noise_used_entries.append({"func_name": na, "original_path": p})

    ood = clone_base_structure(base_struct)
    ood.setdefault("metadata", {})
    ood["metadata"]["ood_mode"] = "noise"
    ood["metadata"]["seed"] = seed
    ood["metadata"]["noise_k"] = noise_k
    ood["metadata"]["noise_icons_dir"] = noise_icons_dir

    icon_w, icon_h = cfg.icon_size

    for _, page_info in ood["pages"].items():
        occupied = []
        for action, info in page_info["layout"].items():
            if action == "page_title":
                continue
            occupied.append(info["bbox"])

        candidates = grid[:]
        rng.shuffle(candidates)

        placed = []
        for _ in range(noise_k):
            found = None
            for (x, y) in candidates:
                rect = [x, y, x + icon_w, y + icon_h]
                if not any_overlap(rect, occupied) and not any_overlap(rect, placed):
                    found = rect
                    break
            if found is None:
                x, y = rng.choice(candidates)
                found = [x, y, x + icon_w, y + icon_h]
            placed.append(found)

        for i, na in enumerate(noise_actions):
            page_info["layout"][na] = {
                "bbox": placed[i],
                "type": "noise",
                "display_name": na  # func_name과 동일하게 두는 게 혼란이 적음
            }

        # transitions.icon_bbox 재동기화(안전)
        for t in page_info.get("transitions", []):
            a = t.get("action")
            if a in page_info["layout"]:
                t["icon_bbox"] = page_info["layout"][a]["bbox"]

    # config만 복사
    copy_config_only(base_env_dir, out_dir)

    # write ui_structure + regenerated layer
    write_json(os.path.join(out_dir, "ui_structure.json"), ood)
    layer = build_layered_structure_from_pages(ood["pages"])
    layer["metadata"].update(ood.get("metadata", {}))
    write_json(os.path.join(out_dir, "ui_structure_layer.json"), layer)

    # used_icons = base used + noise entries (정합/재현성)
    write_used_icons_list(out_dir, used_list + noise_used_entries)

    # render pages
    dst_pages = os.path.join(out_dir, "pages")
    ensure_dir(dst_pages)
    for page_id, page_info in ood["pages"].items():
        img = render_page_from_structure(page_id, page_info, cfg, action_to_iconimg)
        save_png(img, os.path.join(dst_pages, f"{page_id}.png"))

    print(f"[OK] Env-Noise -> {out_dir}")

# ----------------------------
# Main Entrypoint
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="기존 ui_environment run directory")
    parser.add_argument("--out", required=True, help="OOD output root directory")
    parser.add_argument("--alt_icons", default="", help="Env-Image용 아이콘 디렉토리")
    parser.add_argument("--noise_icons", default="", help="Env-Noise용 노이즈 아이콘 디렉토리")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_k", type=int, default=2)
    args = parser.parse_args()

    base_dir = args.base
    out_root = args.out
    ensure_dir(out_root)

    # 1) Env-Base (원본 복제는 그대로 OK)
    build_env_base(base_dir, os.path.join(out_root, "Env-Base"))

    # 2) Env-Image
    if args.alt_icons:
        build_env_image(base_dir, os.path.join(out_root, "Env-Image"), args.alt_icons, seed=args.seed)
    else:
        print("[SKIP] Env-Image: --alt_icons not provided")

    # 3) Env-Name
    name_map = {"Animals": "Creatures", "Business": "Jobs"}
    build_env_name(base_dir, os.path.join(out_root, "Env-Name"), name_map=name_map)

    # 4) Env-Position
    build_env_position(base_dir, os.path.join(out_root, "Env-Position"), seed=args.seed)

    # 5) Env-Noise
    if args.noise_icons:
        build_env_noise(base_dir, os.path.join(out_root, "Env-Noise"), args.noise_icons, noise_k=args.noise_k, seed=args.seed)
    else:
        print("[SKIP] Env-Noise: --noise_icons not provided")
