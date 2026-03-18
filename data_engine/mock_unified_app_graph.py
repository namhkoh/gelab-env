"""
Unified launcher-to-app graph builder for the mock drawer + AMEX trajectories.

This script:
1. Renders the synthetic launcher home/app-drawer pages from `mock_simulator.py`
2. Reads app labels placed in the app drawer
3. Scans AMEX instruction annotations for trajectories that mention those apps
4. Starts each matched trajectory from the first step whose package_name matches the app
5. Reuses `amex_sim2real_compose_action_coord.py` to detect/crop/compose in-app pages
6. Connects the drawer app icons to per-app hub pages and then to the matched trajectories
7. Saves unified `ui_structure.json`, `ui_structure_layer.json`, `ui_topology.png`,
   per-page debug overlays, extracted assets, and GPT-generated page code

Notes:
- By default, only the contiguous in-app segment is kept after the first matching
  package_name step. Use `--include_post_app_steps` to keep the remainder.
- `action` is stored as a lowercase semantic action (`tap`, `swipe`, `press_home`, ...)
  while `raw_action` preserves the original AMEX/raw action string.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw

import mock_simulator as launcher_mock


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LAYOUT_CONFIG_PATH = SCRIPT_DIR / "mock_simulator_layout_config.json"

action_compose = None
merged_compose = None

HOME_PAGE_ID = launcher_mock.HOME_PAGE_ID
DRAWER_PAGE_ID = launcher_mock.DRAWER_PAGE_ID
OUTPUT_CANVAS_SIZE = launcher_mock.CANVAS_SIZE
VISIBLE_VIEWPORT_HEIGHT = launcher_mock.VISIBLE_VIEWPORT_HEIGHT

HUB_PAGE_SIZE = 6
HUB_CONTENT_TOP = 180
HUB_CARD_GAP_X = 36
HUB_CARD_GAP_Y = 42
HUB_CARD_W = 466
HUB_CARD_H = 320
HUB_MARGIN_X = 56
HUB_TITLE_Y = 118
HUB_SUBTITLE_Y = 210
HUB_GRID_TOP = 320
HUB_HANDLE_BBOX = [446, 2340, 634, 2374]


@dataclass
class DrawerAppSpec:
    label: str
    asset: str
    slug: str
    layout_key: str
    bbox: List[int]
    match_tokens: set[str]


@dataclass
class MatchedTrajectory:
    app_slug: str
    app_label: str
    annotation_path: str
    episode_id: str
    instruction: str
    start_step_idx: int
    end_step_idx: int
    matched_package: str
    total_steps: int


def _normalize_compact(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(text or "").lower())


def _full_trajectory_id(match: MatchedTrajectory) -> str:
    return Path(match.annotation_path).stem


def _ensure_compose_modules(load_topology: bool = False) -> None:
    global action_compose, merged_compose

    if action_compose is None:
        try:
            import amex_sim2real_compose_action_coord as action_compose_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Full unified graph composition requires the dependencies used by "
                "`amex_sim2real_compose_action_coord.py` (for example `openai`, "
                "`ultralytics`, and EasyOCR runtime dependencies)."
            ) from exc
        action_compose = action_compose_module

    if load_topology and merged_compose is None:
        try:
            import amex_sim2real_compose as merged_compose_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Topology export requires the dependencies used by "
                "`amex_sim2real_compose.py` (for example `openai` and plotting deps)."
            ) from exc
        merged_compose = merged_compose_module


def _tokenize_words(text: str) -> List[str]:
    return [token for token in re.split(r"[^0-9a-z]+", str(text or "").lower()) if token]


def _build_match_tokens(label: str, asset: str) -> set[str]:
    tokens: set[str] = set()
    for candidate in (label, asset, str(asset).replace("_real", "")):
        compact = _normalize_compact(candidate)
        if len(compact) >= 3:
            tokens.add(compact)
        for token in _tokenize_words(candidate):
            if len(token) >= 3:
                tokens.add(token)
    return tokens


def _matches_text(text: str, tokens: Iterable[str]) -> bool:
    text_lower = str(text or "").lower()
    text_compact = _normalize_compact(text_lower)
    for token in tokens:
        if token in text_lower or token in text_compact:
            return True
    return False


def _typed_layout_to_plain(layout: Dict[str, Any]) -> Dict[str, List[int]]:
    plain: Dict[str, List[int]] = {}
    for key, value in (layout or {}).items():
        if isinstance(value, dict) and "bbox" in value:
            plain[key] = [int(v) for v in value["bbox"]]
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            plain[key] = [int(v) for v in value]
    return plain


def _resolve_spec_label(spec: dict) -> str:
    label = str(spec.get("label") or "").strip()
    if label:
        return label
    asset = str(spec.get("asset") or "").strip().lower()
    if asset in launcher_mock.REAL_ICON_LIBRARY:
        return str(launcher_mock.REAL_ICON_LIBRARY[asset]["label"])
    return asset or "app"


def _find_layout_key(layout: Dict[str, List[int]], preferred_label: str, asset: str) -> Optional[str]:
    candidates = [
        str(preferred_label or "").strip(),
        launcher_mock._safe_key(str(preferred_label or "").strip()),
        str(asset or "").strip(),
        launcher_mock._safe_key(str(asset or "").strip()),
    ]
    lowered = {key.lower(): key for key in layout}
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in layout:
            return candidate
        lowered_match = lowered.get(candidate.lower())
        if lowered_match:
            return lowered_match
    return None


def _extract_drawer_apps(layout_config: dict, drawer_layout: Dict[str, List[int]]) -> List[DrawerAppSpec]:
    apps: List[DrawerAppSpec] = []
    drawer_icons = layout_config.get("pages", {}).get("app_drawer", {}).get("icons", [])
    for spec in drawer_icons:
        if not isinstance(spec, dict):
            continue
        label = _resolve_spec_label(spec)
        asset = str(spec.get("asset") or label)
        layout_key = _find_layout_key(drawer_layout, label, asset)
        if layout_key is None:
            continue
        slug = _normalize_compact(label or asset)
        if not slug:
            continue
        apps.append(DrawerAppSpec(
            label=label,
            asset=asset,
            slug=slug,
            layout_key=layout_key,
            bbox=[int(v) for v in drawer_layout[layout_key]],
            match_tokens=_build_match_tokens(label, asset),
        ))
    return apps


def _find_first_matching_step_index(steps: List[dict], tokens: set[str]) -> Tuple[Optional[int], str]:
    for idx, step in enumerate(steps):
        package_name = str(step.get("package_name", "") or "")
        if _matches_text(package_name, tokens):
            return idx, package_name
    return None, ""


def _find_matching_segment_end(steps: List[dict], start_idx: int, tokens: set[str]) -> int:
    end_idx = start_idx + 1
    while end_idx < len(steps):
        package_name = str(steps[end_idx].get("package_name", "") or "")
        if not _matches_text(package_name, tokens):
            break
        end_idx += 1
    return end_idx


def _scan_matching_annotations(apps: List[DrawerAppSpec],
                               annotations_dir: str,
                               include_post_app_steps: bool) -> Dict[str, List[MatchedTrajectory]]:
    matches: Dict[str, List[MatchedTrajectory]] = defaultdict(list)
    annotations_path = Path(annotations_dir)
    for annot_path in sorted(annotations_path.glob("*.json")):
        try:
            trajectory = json.loads(annot_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        instruction = str(trajectory.get("instruction", "") or "")
        steps = trajectory.get("steps", []) or []
        packages = [str(step.get("package_name", "") or "") for step in steps]

        for app in apps:
            if not (_matches_text(instruction, app.match_tokens) or any(_matches_text(pkg, app.match_tokens) for pkg in packages)):
                continue

            start_idx, matched_package = _find_first_matching_step_index(steps, app.match_tokens)
            if start_idx is None:
                continue

            end_idx = len(steps) if include_post_app_steps else _find_matching_segment_end(steps, start_idx, app.match_tokens)
            matches[app.slug].append(MatchedTrajectory(
                app_slug=app.slug,
                app_label=app.label,
                annotation_path=str(annot_path),
                episode_id=str(trajectory.get("episode_id") or annot_path.stem),
                instruction=instruction,
                start_step_idx=int(start_idx),
                end_step_idx=int(end_idx),
                matched_package=matched_package,
                total_steps=len(steps),
            ))

    for app_slug in matches:
        matches[app_slug].sort(key=lambda item: (item.episode_id, item.start_step_idx))
    return matches


def _semantic_action(raw_action: str, fallback: str = "") -> str:
    normalized = str(raw_action or "").strip().upper()
    semantic_map = {
        "TAP": "tap",
        "CLICK": "tap",
        "SWIPE": "swipe",
        "SCROLL": "swipe",
        "TYPE": "type",
        "TEXT": "type",
        "PRESS_ENTER": "press_enter",
        "PRESS_BACK": "press_back",
        "PRESS_HOME": "press_home",
        "TASK_COMPLETE": "task_complete",
        "TASK_IMPOSSIBLE": "task_impossible",
    }
    if normalized in semantic_map:
        return semantic_map[normalized]
    return str(fallback or normalized or "navigate").strip().lower()


def _bbox_center(bbox: List[int]) -> List[int]:
    return [int(round((bbox[0] + bbox[2]) / 2.0)), int(round((bbox[1] + bbox[3]) / 2.0))]


def _valid_point(point: List[int]) -> bool:
    return isinstance(point, (list, tuple)) and len(point) == 2 and not (int(point[0]) == 0 and int(point[1]) == 0)


def _valid_bbox(bbox: List[int]) -> bool:
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4 and any(int(v) != 0 for v in bbox)


def _build_tap_transition(target_page: str,
                          target_element: str,
                          icon_bbox: List[int],
                          raw_action: str = "TAP") -> dict:
    return {
        "raw_action": raw_action,
        "action": target_element,
        "action_kind": _semantic_action(raw_action),
        "target_page": target_page,
        "canvas_action_bbox": [int(v) for v in icon_bbox],
        "canvas_action_point": _bbox_center(icon_bbox),
        "canvas_lift_coord": [0, 0],
        "icon_bbox": [int(v) for v in icon_bbox],
        "type_text": "",
        "gesture_direction": "",
    }


def _build_swipe_transition(target_page: str,
                            action_coord: List[int],
                            lift_coord: List[int],
                            direction: str,
                            icon_bbox: Optional[List[int]] = None) -> dict:
    start = [int(action_coord[0]), int(action_coord[1])]
    end = [int(lift_coord[0]), int(lift_coord[1])]
    bbox = icon_bbox or [
        min(start[0], end[0]),
        min(start[1], end[1]),
        max(start[0], end[0]),
        max(start[1], end[1]),
    ]
    return {
        "raw_action": "SWIPE",
        "action": "swipe",
        "action_kind": "swipe",
        "target_page": target_page,
        "canvas_action_bbox": [int(v) for v in bbox],
        "canvas_action_point": start,
        "canvas_lift_coord": end,
        "icon_bbox": [int(v) for v in bbox],
        "type_text": "",
        "gesture_direction": direction,
    }


def _serialize_layout(layout: Dict[str, List[int]]) -> Dict[str, dict]:
    return {
        key: {
            "bbox": [int(v) for v in bbox],
            "type": "system" if key in ("back", "home") else "normal",
        }
        for key, bbox in (layout or {}).items()
    }


def _serialize_transition(transition: dict) -> dict:
    raw_action = str(transition.get("raw_action", "") or "").strip().upper()
    semantic_action = _semantic_action(raw_action, transition.get("action", ""))
    item = {
        "action": semantic_action,
        "target_page": transition.get("target_page", ""),
        "raw_action": raw_action,
    }

    target_element = str(transition.get("action", "") or "").strip()
    if target_element and target_element.lower() not in {
        semantic_action,
        "tap",
        "swipe",
        "type",
        "press_enter",
        "press_back",
        "press_home",
        "task_complete",
        "task_impossible",
    }:
        item["target_element"] = target_element

    action_coord = action_compose._stored_transition_action_coord(transition)
    lift_coord = action_compose._stored_transition_lift_coord(transition)
    if _valid_point(action_coord):
        item["action_coord"] = [int(action_coord[0]), int(action_coord[1])]
    if _valid_point(lift_coord):
        item["lift_coord"] = [int(lift_coord[0]), int(lift_coord[1])]

    icon_bbox = transition.get("icon_bbox") or [0, 0, 0, 0]
    if raw_action in ("SWIPE", "SCROLL"):
        icon_bbox = transition.get("canvas_action_bbox") or icon_bbox
    if _valid_bbox(icon_bbox):
        item["icon_bbox"] = [int(v) for v in icon_bbox]

    type_text = str(transition.get("type_text", "") or "").strip()
    if type_text:
        item["type_text"] = type_text
    gesture_direction = str(transition.get("gesture_direction", "") or "").strip()
    if gesture_direction:
        item["gesture_direction"] = gesture_direction
    return item


def _build_ui_layer(pages: Dict[str, dict], root_page_id: str) -> dict:
    visited: set[str] = set()

    def build_node(page_id: str) -> Optional[dict]:
        if page_id in visited or page_id not in pages:
            return None
        visited.add(page_id)
        page = pages[page_id]
        serialized_transitions = [_serialize_transition(t) for t in page.get("transitions", [])]
        non_system = [t for t in serialized_transitions if t.get("action") not in ("press_back", "press_home")]

        subnodes = []
        for transition in non_system:
            child_id = transition.get("target_page", "")
            child = build_node(child_id)
            if child is not None:
                subnodes.append(child)
        return {
            "image": page.get("image", f"{page_id}.png"),
            "depth": int(page.get("depth", 0)),
            "layout": _serialize_layout(page.get("layout", {})),
            "transitions": non_system,
            "subnodes": subnodes,
        }

    root = build_node(root_page_id)
    for page_id in sorted(pages):
        if page_id in visited:
            continue
        child = build_node(page_id)
        if child is not None and root is not None:
            root["subnodes"].append(child)
    return {
        "root": root,
        "metadata": {
            "source": "mock_unified_app_graph",
            "canvas_size": list(OUTPUT_CANVAS_SIZE),
            "total_pages": len(pages),
        },
    }


def _save_ui_structure(output_dir: Path,
                       pages: Dict[str, dict],
                       root_page_id: str,
                       metadata: dict) -> None:
    serialized_pages = {}
    for page_id, page in sorted(pages.items(), key=lambda item: (int(item[1].get("depth", 0)), str(item[0]))):
        serialized_pages[page_id] = {
            "image": page.get("image", f"{page_id}.png"),
            "depth": int(page.get("depth", 0)),
            "layout": _serialize_layout(page.get("layout", {})),
            "transitions": [_serialize_transition(t) for t in page.get("transitions", [])],
        }

    ui_structure = {
        "pages": serialized_pages,
        "metadata": {
            **metadata,
            "source": "mock_unified_app_graph",
            "canvas_size": list(OUTPUT_CANVAS_SIZE),
            "root_page_id": root_page_id,
            "total_pages": len(serialized_pages),
        },
    }
    ui_layer = _build_ui_layer(pages, root_page_id)

    (output_dir / "ui_structure.json").write_text(json.dumps(ui_structure, indent=2), encoding="utf-8")
    (output_dir / "ui_structure_layer.json").write_text(json.dumps(ui_layer, indent=2), encoding="utf-8")


def _make_page_summary(page_name: str,
                       application_id: str,
                       application_name: str,
                       layout: Dict[str, List[int]],
                       page_family: str) -> dict:
    icons = sorted([key for key in layout if key not in ("back", "home")])
    return {
        "page_family": page_family,
        "semantic_page_name": page_name,
        "logical_page_name": page_name,
        "application_id": application_id,
        "application_name": application_name,
        "icons": icons,
        "additional_icons": [],
        "components": icons,
        "additional_components": [],
    }


def _render_hub_page(app: DrawerAppSpec,
                     app_matches: List[MatchedTrajectory],
                     page_idx: int,
                     total_pages: int) -> Tuple[Image.Image, Dict[str, List[int]]]:
    canvas = Image.new("RGB", OUTPUT_CANVAS_SIZE, (247, 249, 252))
    draw = ImageDraw.Draw(canvas)
    layout: Dict[str, List[int]] = {}

    title_font = launcher_mock.FONT_LG
    subtitle_font = launcher_mock.FONT_MD
    card_title_font = launcher_mock.FONT_MD
    card_body_font = launcher_mock.FONT_SM

    draw.rounded_rectangle([36, 96, 1044, 2308], radius=68, fill=(255, 255, 255), outline=(224, 228, 237), width=2)
    draw.text((76, HUB_TITLE_Y), f"{app.label} Trajectories", fill=(26, 31, 41), font=title_font)
    draw.text((76, HUB_SUBTITLE_Y), f"Page {page_idx + 1}/{total_pages}", fill=(104, 110, 124), font=subtitle_font)

    start = page_idx * HUB_PAGE_SIZE
    chunk = app_matches[start:start + HUB_PAGE_SIZE]
    for local_idx, match in enumerate(chunk):
        row = local_idx // 2
        col = local_idx % 2
        x1 = HUB_MARGIN_X + col * (HUB_CARD_W + HUB_CARD_GAP_X)
        y1 = HUB_GRID_TOP + row * (HUB_CARD_H + HUB_CARD_GAP_Y)
        x2 = x1 + HUB_CARD_W
        y2 = y1 + HUB_CARD_H
        bbox = [x1, y1, x2, y2]
        card_key = f"trajectory_{page_idx:02d}_{local_idx:02d}"
        layout[card_key] = bbox

        draw.rounded_rectangle(bbox, radius=38, fill=(248, 250, 254), outline=(215, 222, 233), width=2)
        draw.text((x1 + 26, y1 + 22), match.episode_id[:12], fill=(88, 95, 110), font=card_body_font)

        title = f"Start step {match.start_step_idx + 1}/{match.total_steps}"
        draw.text((x1 + 26, y1 + 62), title, fill=(25, 31, 44), font=card_title_font)

        instruction = re.sub(r"\s+", " ", match.instruction).strip()
        wrapped = _wrap_text(instruction, width=34)
        visible_lines = wrapped[:6]
        text_y = y1 + 116
        for line in visible_lines:
            draw.text((x1 + 26, text_y), line, fill=(66, 72, 84), font=card_body_font)
            text_y += 34

    draw.rounded_rectangle(HUB_HANDLE_BBOX, radius=18, fill=(74, 81, 94))
    layout["drawer_handle"] = list(HUB_HANDLE_BBOX)

    canvas, layout = action_compose._ensure_system_nav_controls(canvas, layout)
    return canvas, layout


def _wrap_text(text: str, width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _launcher_transition_from_mock(swipe_transition: dict, target_page: str) -> dict:
    return {
        "raw_action": "SWIPE",
        "action": "swipe",
        "action_kind": "swipe",
        "target_page": target_page,
        "canvas_action_bbox": [int(v) for v in swipe_transition.get("icon_bbox", [0, 0, 0, 0])],
        "canvas_action_point": [int(v) for v in swipe_transition.get("action_coord", [0, 0])],
        "canvas_lift_coord": [int(v) for v in swipe_transition.get("lift_coord", [0, 0])],
        "icon_bbox": [int(v) for v in swipe_transition.get("icon_bbox", [0, 0, 0, 0])],
        "type_text": "",
        "gesture_direction": str(swipe_transition.get("gesture_direction", "") or ""),
    }


def _save_mock_pages(output_dir: Path,
                     layout_config: dict) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, Any]]:
    render_state: Dict[str, Any] = {
        "rng": launcher_mock._build_rng(layout_config),
        "used_random_assets": set(),
        "resolved_icons": [],
    }
    home_img, home_layout_typed, swipe_transition = launcher_mock._draw_home_page(layout_config, render_state)
    drawer_img, drawer_layout_typed = launcher_mock._draw_app_drawer_page(layout_config, render_state)

    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    home_img.save(pages_dir / f"{HOME_PAGE_ID}.png")
    drawer_img.save(pages_dir / f"{DRAWER_PAGE_ID}.png")

    return (
        _typed_layout_to_plain(home_layout_typed),
        _typed_layout_to_plain(drawer_layout_typed),
        {
            "swipe_transition": swipe_transition,
            "resolved_icons": render_state["resolved_icons"],
        },
    )


def _segment_steps(trajectory: dict,
                   start_step_idx: int,
                   end_step_idx: int) -> dict:
    segment = dict(trajectory)
    steps = trajectory.get("steps", []) or []
    selected_steps = []
    for global_idx in range(start_step_idx, end_step_idx):
        step = dict(steps[global_idx])
        step["source_step_index"] = global_idx + 1
        selected_steps.append(step)
    segment["steps"] = selected_steps
    segment["segment_start_step_index"] = start_step_idx + 1
    segment["segment_end_step_index"] = end_step_idx
    return segment


def _step_context_for_segment(segment_trajectory: dict, local_idx: int) -> dict:
    context = action_compose._build_step_context(segment_trajectory, local_idx)
    context["task"] = segment_trajectory.get("instruction", "")
    context["task_instruction"] = segment_trajectory.get("instruction", "")
    context["source_episode_id"] = segment_trajectory.get("episode_id", "")
    context["global_step_index"] = segment_trajectory["steps"][local_idx].get("source_step_index", local_idx + 1)
    return context


def _compose_segment_pages(match: MatchedTrajectory,
                           app: DrawerAppSpec,
                           app_root_page_id: str,
                           home_page_id: str,
                           args,
                           client,
                           model_name: str,
                           yolo_model,
                           ocr_reader,
                           output_dir: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    with open(match.annotation_path, "r", encoding="utf-8") as f:
        trajectory = json.load(f)

    segment_trajectory = _segment_steps(trajectory, match.start_step_idx, match.end_step_idx)
    steps = segment_trajectory.get("steps", []) or []
    if not steps:
        return [], [], []

    print(
        f"    composing trajectory_id={_full_trajectory_id(match)} "
        f"episode={match.episode_id} "
        f"steps={match.start_step_idx + 1}-{match.end_step_idx} "
        f"({len(steps)} pages)"
    )

    pages_dir = output_dir / "pages"
    code_dir = output_dir / "generated_code"
    assets_dir = output_dir / "extracted_assets" / app.slug / match.episode_id
    pages_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    page_rows: List[dict] = []
    manifest_rows: List[dict] = []
    matched_rows: List[dict] = []

    for local_idx, step in enumerate(steps):
        screenshot_name, screenshot_path = action_compose._resolve_step_screenshot(
            step, args.screenshots_dir, match.episode_id, local_idx
        )
        if not os.path.exists(screenshot_path):
            print(
                f"      [{local_idx + 1}/{len(steps)}] "
                f"missing screenshot={screenshot_name} "
                f"source_step={step.get('source_step_index', local_idx + 1)}"
            )
            continue

        step_context = _step_context_for_segment(segment_trajectory, local_idx)
        page_id = f"page_{app.slug}_{match.episode_id}_{local_idx + 1:02d}"
        print(
            f"      [{local_idx + 1}/{len(steps)}] "
            f"source_step={step.get('source_step_index', local_idx + 1)} "
            f"screenshot={screenshot_name}"
        )

        elements, orig_size = action_compose.detect_and_crop(screenshot_path, yolo_model, ocr_reader)
        elements, anno_stats = action_compose._prioritize_element_anno_bboxes(
            elements,
            screenshot_path,
            screenshot_name,
            args.element_anno_dir,
        )
        asset_elements = action_compose._persist_extracted_assets(
            elements,
            screenshot_name,
            str(assets_dir),
            step_context,
        )

        page_img, layout, code_artifact = action_compose.compose_page(
            client,
            model_name,
            asset_elements,
            orig_size,
            screenshot_path,
            step_context,
        )
        page_img, layout = action_compose._ensure_system_nav_controls(page_img, layout)
        page_img.save(pages_dir / f"{page_id}.png")
        action_compose._save_page_code(
            str(code_dir),
            page_id,
            screenshot_name,
            step_context,
            code_artifact,
        )
        print(
            f"        -> page_id={page_id} "
            f"detected={len(elements)} "
            f"anno_loaded={anno_stats.get('loaded', 0)} "
            f"layout={len(layout)}"
        )

        layout = action_compose._ensure_system_layout(layout)
        page_rows.append({
            "page_id": page_id,
            "image": f"{page_id}.png",
            "depth": 3 + local_idx,
            "layout": layout,
            "orig_size": tuple(orig_size),
            "step": step,
            "step_context": step_context,
            "episode_id": match.episode_id,
            "page_name": f"{app.label} {match.episode_id[:8]} step {local_idx + 1}",
            "application_id": app.slug,
            "application_name": app.label,
            "trajectory_ids": [match.episode_id],
            "trace_steps": [step.get("source_step_index", local_idx + 1)],
            "anno_stats": anno_stats,
        })

        for elem in asset_elements:
            manifest_rows.append({
                "page_id": page_id,
                "episode_id": match.episode_id,
                "app_label": app.label,
                "step_index": step.get("source_step_index", local_idx + 1),
                "screenshot": screenshot_name,
                "type": elem.get("type"),
                "label": elem.get("label"),
                "bbox": elem.get("bbox"),
                "asset_path": elem.get("asset_path"),
                "asset_source": elem.get("asset_source"),
            })

        matched_rows.append({
            "page_id": page_id,
            "trajectory_id_full": _full_trajectory_id(match),
            "episode_id": match.episode_id,
            "app_label": app.label,
            "instruction": match.instruction,
            "step_index": step.get("source_step_index", local_idx + 1),
            "screenshot": screenshot_name,
            "package_name": step.get("package_name", ""),
        })

    if not page_rows:
        return [], manifest_rows, matched_rows

    for idx, page in enumerate(page_rows):
        next_page_id = page_rows[idx + 1]["page_id"] if idx + 1 < len(page_rows) else page["page_id"]
        raw_action = str(page["step"].get("action", "") or "").strip().upper()
        if idx + 1 >= len(page_rows):
            if raw_action == "PRESS_HOME":
                next_page_id = home_page_id
            elif raw_action == "PRESS_BACK":
                next_page_id = app_root_page_id

        transition = action_compose._resolve_transition(
            page["step"],
            page["layout"],
            tuple(page["orig_size"]),
            next_page_id,
        )
        transition["source_trace_page"] = page["page_id"]
        transition["source_trajectory_id"] = match.episode_id
        transition["source_step_indices"] = [page["step"].get("source_step_index", idx + 1)]
        transition["action_kind"] = _semantic_action(transition.get("raw_action", ""), transition.get("action", ""))

        transitions = [transition]
        used_system_targets = {transition.get("action")} if transition.get("action") in ("back", "home") else set()
        back_target = page_rows[idx - 1]["page_id"] if idx > 0 else app_root_page_id
        if "back" not in used_system_targets:
            back_transition = action_compose._build_system_transition(
                raw_action="PRESS_BACK",
                action="back",
                target_page=back_target,
                icon_bbox=page["layout"].get("back", action_compose.GELAB_BACK_BBOX),
            )
            back_transition["action_kind"] = "press_back"
            transitions.append(back_transition)
        if "home" not in used_system_targets:
            home_transition = action_compose._build_system_transition(
                raw_action="PRESS_HOME",
                action="home",
                target_page=home_page_id,
                icon_bbox=page["layout"].get("home", action_compose.GELAB_HOME_BBOX),
            )
            home_transition["action_kind"] = "press_home"
            transitions.append(home_transition)

        page["transitions"] = transitions
        page["page_summary"] = _make_page_summary(
            page["page_name"],
            page["application_id"],
            page["application_name"],
            page["layout"],
            page_family="content_page",
        )

    return page_rows, manifest_rows, matched_rows


def _build_rich_launcher_pages(home_layout: Dict[str, List[int]],
                               drawer_layout: Dict[str, List[int]],
                               swipe_transition: dict,
                               drawer_apps: List[DrawerAppSpec],
                               app_first_pages: Dict[str, str]) -> Dict[str, dict]:
    pages: Dict[str, dict] = {}

    home_transitions = [
        _launcher_transition_from_mock(swipe_transition, DRAWER_PAGE_ID),
        {
            **action_compose._build_system_transition(
                raw_action="PRESS_HOME",
                action="home",
                target_page=HOME_PAGE_ID,
                icon_bbox=home_layout.get("home", [0, 0, 0, 0]),
            ),
            "action_kind": "press_home",
        },
    ]
    pages[HOME_PAGE_ID] = {
        "page_id": HOME_PAGE_ID,
        "image": f"{HOME_PAGE_ID}.png",
        "depth": 0,
        "layout": home_layout,
        "transitions": home_transitions,
        "page_name": "Home",
        "application_id": "launcher",
        "application_name": "Launcher",
        "trajectory_ids": [],
        "trace_steps": [],
        "page_summary": _make_page_summary("Home", "launcher", "Launcher", home_layout, page_family="home"),
    }

    drawer_transitions: List[dict] = [
        {
            **action_compose._build_system_transition(
                raw_action="PRESS_BACK",
                action="back",
                target_page=HOME_PAGE_ID,
                icon_bbox=drawer_layout.get("back", [0, 0, 0, 0]),
            ),
            "action_kind": "press_back",
        },
        {
            **action_compose._build_system_transition(
                raw_action="PRESS_HOME",
                action="home",
                target_page=HOME_PAGE_ID,
                icon_bbox=drawer_layout.get("home", [0, 0, 0, 0]),
            ),
            "action_kind": "press_home",
        },
    ]
    for app in drawer_apps:
        first_page = app_first_pages.get(app.slug)
        if not first_page:
            continue
        transition = _build_tap_transition(first_page, app.layout_key, app.bbox, raw_action="TAP")
        drawer_transitions.append(transition)

    pages[DRAWER_PAGE_ID] = {
        "page_id": DRAWER_PAGE_ID,
        "image": f"{DRAWER_PAGE_ID}.png",
        "depth": 1,
        "layout": drawer_layout,
        "transitions": drawer_transitions,
        "page_name": "App Drawer",
        "application_id": "launcher",
        "application_name": "Launcher",
        "trajectory_ids": [],
        "trace_steps": [],
        "page_summary": _make_page_summary("App Drawer", "launcher", "Launcher", drawer_layout, page_family="app_drawer"),
    }
    return pages


def _build_app_hub_pages(app: DrawerAppSpec,
                         app_matches: List[MatchedTrajectory],
                         first_page_by_episode: Dict[str, str],
                         drawer_page_id: str,
                         home_page_id: str,
                         output_dir: Path) -> List[dict]:
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    hub_pages: List[dict] = []
    total_hub_pages = max(1, math.ceil(len(app_matches) / HUB_PAGE_SIZE))
    for page_idx in range(total_hub_pages):
        page_id = f"page_{app.slug}_hub_{page_idx + 1:02d}"
        image, layout = _render_hub_page(app, app_matches, page_idx, total_hub_pages)
        image.save(pages_dir / f"{page_id}.png")

        transitions: List[dict] = []
        start = page_idx * HUB_PAGE_SIZE
        chunk = app_matches[start:start + HUB_PAGE_SIZE]
        for local_idx, match in enumerate(chunk):
            card_key = f"trajectory_{page_idx:02d}_{local_idx:02d}"
            bbox = layout.get(card_key, [0, 0, 0, 0])
            first_page_id = first_page_by_episode.get(match.episode_id)
            if not first_page_id or not _valid_bbox(bbox):
                continue
            transition = _build_tap_transition(first_page_id, card_key, bbox, raw_action="TAP")
            transition["source_trajectory_id"] = match.episode_id
            transition["source_step_indices"] = [match.start_step_idx + 1]
            transition["action_kind"] = "tap"
            transitions.append(transition)

        if page_idx + 1 < total_hub_pages:
            transitions.append(_build_swipe_transition(
                target_page=f"page_{app.slug}_hub_{page_idx + 2:02d}",
                action_coord=[540, 2306],
                lift_coord=[540, 1560],
                direction="up",
                icon_bbox=[500, 1560, 580, 2306],
            ))
        if page_idx > 0:
            transitions.append(_build_swipe_transition(
                target_page=f"page_{app.slug}_hub_{page_idx:02d}",
                action_coord=[540, 1480],
                lift_coord=[540, 2280],
                direction="down",
                icon_bbox=[500, 1480, 580, 2280],
            ))

        back_transition = action_compose._build_system_transition(
            raw_action="PRESS_BACK",
            action="back",
            target_page=drawer_page_id,
            icon_bbox=layout.get("back", action_compose.GELAB_BACK_BBOX),
        )
        back_transition["action_kind"] = "press_back"
        home_transition = action_compose._build_system_transition(
            raw_action="PRESS_HOME",
            action="home",
            target_page=home_page_id,
            icon_bbox=layout.get("home", action_compose.GELAB_HOME_BBOX),
        )
        home_transition["action_kind"] = "press_home"
        transitions.extend([back_transition, home_transition])

        hub_pages.append({
            "page_id": page_id,
            "image": f"{page_id}.png",
            "depth": 2 + page_idx,
            "layout": layout,
            "transitions": transitions,
            "page_name": f"{app.label} Hub {page_idx + 1}",
            "application_id": app.slug,
            "application_name": app.label,
            "trajectory_ids": [match.episode_id for match in chunk],
            "trace_steps": [match.start_step_idx + 1 for match in chunk],
            "page_summary": _make_page_summary(
                f"{app.label} Hub {page_idx + 1}",
                app.slug,
                app.label,
                layout,
                page_family="app_hub",
            ),
        })

    return hub_pages


def _write_topology_artifacts(output_dir: Path, pages: Dict[str, dict], root_page_id: str) -> None:
    _ensure_compose_modules(load_topology=True)
    topology_tree = merged_compose._build_topology_tree(pages, root_page_id=root_page_id)
    topology_graph = merged_compose._build_full_topology_graph(pages, root_page_id=root_page_id)
    (output_dir / "ui_topology_tree.json").write_text(json.dumps(topology_tree, indent=2), encoding="utf-8")
    (output_dir / "ui_topology.json").write_text(json.dumps(topology_graph, indent=2), encoding="utf-8")
    merged_compose._save_topology_visualization(pages, str(output_dir / "ui_topology.png"))


def _save_action_debug_overlays(output_dir: Path, pages: Dict[str, dict]) -> None:
    _ensure_compose_modules(load_topology=False)
    debug_dir = output_dir / "action_coord_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for page_id, page in pages.items():
        image_path = output_dir / "pages" / f"{page_id}.png"
        overlay_path = debug_dir / f"{page_id}.png"
        action_compose._save_action_debug_overlay(str(image_path), str(overlay_path), page.get("transitions", []))


def _write_match_report(output_dir: Path,
                        drawer_apps: List[DrawerAppSpec],
                        matches_by_app: Dict[str, List[MatchedTrajectory]]) -> None:
    payload = {
        "apps": [
            {
                "label": app.label,
                "asset": app.asset,
                "slug": app.slug,
                "layout_key": app.layout_key,
                "bbox": app.bbox,
                "matched_annotation_count": len(matches_by_app.get(app.slug, [])),
                "matches": [
                    {
                        "trajectory_id_full": _full_trajectory_id(match),
                        "episode_id": match.episode_id,
                        "annotation_path": match.annotation_path,
                        "instruction": match.instruction,
                        "start_step_idx": match.start_step_idx,
                        "end_step_idx": match.end_step_idx,
                        "matched_package": match.matched_package,
                        "total_steps": match.total_steps,
                    }
                    for match in matches_by_app.get(app.slug, [])
                ],
            }
            for app in drawer_apps
        ]
    }
    (output_dir / "matched_annotations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_match_summary(drawer_apps: List[DrawerAppSpec],
                         matches_by_app: Dict[str, List[MatchedTrajectory]],
                         max_trajectories_per_app: Optional[int]) -> None:
    print(f"Drawer apps detected: {len(drawer_apps)}")
    for app in drawer_apps:
        all_matches = matches_by_app.get(app.slug, [])
        shown_matches = all_matches[:max_trajectories_per_app] if max_trajectories_per_app is not None else all_matches
        print(f"[{app.label}] matched {len(all_matches)} trajectories")
        if max_trajectories_per_app is not None and len(all_matches) > len(shown_matches):
            print(f"  using first {len(shown_matches)} trajectories due to --max_trajectories_per_app")
        for match in shown_matches:
            print(
                "  - "
                f"trajectory_id={_full_trajectory_id(match)} "
                f"episode={match.episode_id} "
                f"start_step={match.start_step_idx + 1} "
                f"end_step={match.end_step_idx} "
                f"package={match.matched_package}"
            )


def _renumber_page_outputs(output_dir: Path,
                           pages: Dict[str, dict],
                           asset_manifest: List[dict],
                           matched_step_rows: List[dict],
                           root_page_id: str) -> Tuple[Dict[str, dict], str, Dict[str, str]]:
    ordered_old_ids = list(pages.keys())
    page_id_map = {old_page_id: f"page_{idx}" for idx, old_page_id in enumerate(ordered_old_ids)}

    def rename_outputs(base_dir: Path, suffix: str) -> None:
        if not base_dir.exists():
            return
        staged_paths: List[Tuple[Path, Path]] = []
        for old_page_id, new_page_id in page_id_map.items():
            old_path = base_dir / f"{old_page_id}{suffix}"
            if not old_path.exists():
                continue
            staged_path = base_dir / f"__tmp__{old_page_id}{suffix}"
            if staged_path.exists():
                staged_path.unlink()
            old_path.rename(staged_path)
            staged_paths.append((staged_path, base_dir / f"{new_page_id}{suffix}"))

        for staged_path, final_path in staged_paths:
            if final_path.exists():
                final_path.unlink()
            staged_path.rename(final_path)

    rename_outputs(output_dir / "pages", ".png")
    rename_outputs(output_dir / "generated_code", ".py")

    renumbered_pages: Dict[str, dict] = {}
    for old_page_id in ordered_old_ids:
        page = dict(pages[old_page_id])
        new_page_id = page_id_map[old_page_id]
        page["page_id"] = new_page_id
        page["image"] = f"{new_page_id}.png"

        remapped_transitions = []
        for transition in page.get("transitions", []):
            remapped = dict(transition)
            target_page = remapped.get("target_page")
            source_trace_page = remapped.get("source_trace_page")
            if target_page in page_id_map:
                remapped["target_page"] = page_id_map[target_page]
            if source_trace_page in page_id_map:
                remapped["source_trace_page"] = page_id_map[source_trace_page]
            remapped_transitions.append(remapped)
        page["transitions"] = remapped_transitions
        renumbered_pages[new_page_id] = page

    for row in asset_manifest:
        page_id = row.get("page_id")
        if page_id in page_id_map:
            row["page_id"] = page_id_map[page_id]

    for row in matched_step_rows:
        page_id = row.get("page_id")
        if page_id in page_id_map:
            row["page_id"] = page_id_map[page_id]

    return renumbered_pages, page_id_map.get(root_page_id, root_page_id), page_id_map


def build_unified_graph(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_config = launcher_mock._load_layout_config(Path(args.layout_config))
    home_layout, drawer_layout, launcher_meta = _save_mock_pages(output_dir, layout_config)
    drawer_apps = _extract_drawer_apps(layout_config, drawer_layout)
    matches_by_app = _scan_matching_annotations(drawer_apps, args.annotations_dir, args.include_post_app_steps)
    _write_match_report(output_dir, drawer_apps, matches_by_app)
    _print_match_summary(drawer_apps, matches_by_app, args.max_trajectories_per_app)

    if args.scan_only:
        total_matches = sum(len(matches_by_app.get(app.slug, [])) for app in drawer_apps)
        print(f"scan_only: apps={len(drawer_apps)} matched_trajectories={total_matches}")
        return

    _ensure_compose_modules(load_topology=True)
    client = action_compose.load_api_client()
    model_name = args.model_name
    yolo_model, ocr_reader = action_compose.load_detection_models(args.weights_dir, args.gpu)

    pages: Dict[str, dict] = {}
    first_page_by_app: Dict[str, str] = {}
    asset_manifest: List[dict] = []
    matched_step_rows: List[dict] = []

    all_hub_pages: List[dict] = []
    all_app_pages: List[dict] = []

    for app in drawer_apps:
        app_matches = matches_by_app.get(app.slug, [])
        if args.max_trajectories_per_app is not None:
            app_matches = app_matches[:args.max_trajectories_per_app]
        if not app_matches:
            print(f"\n[App] {app.label}: no matched trajectories to compose")
            continue

        print(f"\n[App] {app.label}: composing {len(app_matches)} trajectories")

        first_page_by_episode: Dict[str, str] = {}
        composed_rows_by_episode: Dict[str, List[dict]] = {}
        for match_idx, match in enumerate(app_matches, start=1):
            print(
                f"  [{match_idx}/{len(app_matches)}] "
                f"trajectory_id={_full_trajectory_id(match)} "
                f"episode={match.episode_id} "
                f"start_step={match.start_step_idx + 1} "
                f"end_step={match.end_step_idx}"
            )
            app_root_page_id = f"page_{app.slug}_hub_01"
            page_rows, manifest_rows, matched_rows = _compose_segment_pages(
                match=match,
                app=app,
                app_root_page_id=app_root_page_id,
                home_page_id=HOME_PAGE_ID,
                args=args,
                client=client,
                model_name=model_name,
                yolo_model=yolo_model,
                ocr_reader=ocr_reader,
                output_dir=output_dir,
            )
            if not page_rows:
                print(
                    f"    -> skipped trajectory_id={_full_trajectory_id(match)} "
                    f"episode={match.episode_id} (no composed pages)"
                )
                continue
            composed_rows_by_episode[match.episode_id] = page_rows
            first_page_by_episode[match.episode_id] = page_rows[0]["page_id"]
            asset_manifest.extend(manifest_rows)
            matched_step_rows.extend(matched_rows)
            print(
                f"    -> completed trajectory_id={_full_trajectory_id(match)} "
                f"episode={match.episode_id} pages={len(page_rows)}"
            )

        if not first_page_by_episode:
            print(f"  -> no usable trajectories remained for {app.label}")
            continue

        hub_pages = _build_app_hub_pages(
            app=app,
            app_matches=[match for match in app_matches if match.episode_id in first_page_by_episode],
            first_page_by_episode=first_page_by_episode,
            drawer_page_id=DRAWER_PAGE_ID,
            home_page_id=HOME_PAGE_ID,
            output_dir=output_dir,
        )
        if not hub_pages:
            print(f"  -> failed to create hub pages for {app.label}")
            continue

        first_page_by_app[app.slug] = hub_pages[0]["page_id"]
        all_hub_pages.extend(hub_pages)
        for episode_id in sorted(composed_rows_by_episode):
            all_app_pages.extend(composed_rows_by_episode[episode_id])
        print(f"  -> hub_pages={len(hub_pages)} trajectory_pages={sum(len(v) for v in composed_rows_by_episode.values())}")

    pages.update(_build_rich_launcher_pages(
        home_layout=home_layout,
        drawer_layout=drawer_layout,
        swipe_transition=launcher_meta["swipe_transition"],
        drawer_apps=drawer_apps,
        app_first_pages=first_page_by_app,
    ))
    for page in all_hub_pages + all_app_pages:
        page["depth"] = max(2, int(page.get("depth", 2)))
        pages[page["page_id"]] = page

    metadata = {
        "launcher_layout_config": str(args.layout_config),
        "annotations_dir": str(args.annotations_dir),
        "screenshots_dir": str(args.screenshots_dir),
        "element_anno_dir": str(args.element_anno_dir),
        "model_name": model_name,
        "matched_app_count": sum(1 for app in drawer_apps if first_page_by_app.get(app.slug)),
        "matched_trajectory_count": len({row["episode_id"] for row in matched_step_rows}),
        "resolved_icons": launcher_meta.get("resolved_icons", []),
    }

    pages, root_page_id, page_id_map = _renumber_page_outputs(
        output_dir=output_dir,
        pages=pages,
        asset_manifest=asset_manifest,
        matched_step_rows=matched_step_rows,
        root_page_id=HOME_PAGE_ID,
    )
    metadata["page_id_map"] = page_id_map

    _save_ui_structure(output_dir, pages, root_page_id, metadata)
    _write_topology_artifacts(output_dir, pages, root_page_id)
    _save_action_debug_overlays(output_dir, pages)
    (output_dir / "trajectory_assets_manifest.json").write_text(json.dumps(asset_manifest, indent=2), encoding="utf-8")
    (output_dir / "matched_steps.json").write_text(json.dumps(matched_step_rows, indent=2), encoding="utf-8")

    print(f"Done: {output_dir}")
    print(f"  pages: {len(pages)}")
    print(f"  assets: {len(asset_manifest)}")
    print(f"  ui_structure.json")
    print(f"  ui_structure_layer.json")
    print(f"  ui_topology.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Build one unified launcher-to-app graph from mock drawer apps + AMEX trajectories")
    parser.add_argument("--layout_config", type=str, default=str(DEFAULT_LAYOUT_CONFIG_PATH),
                        help="Mock launcher layout config used to read app drawer labels/positions")
    parser.add_argument("--annotations_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/instruction_anno",
                        help="Directory containing AMEX instruction annotation JSONs")
    parser.add_argument("--screenshots_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/screenshot",
                        help="Directory containing AMEX screenshots")
    parser.add_argument("--element_anno_dir", type=str,
                        default="/ext_hdd2/tsyou/AMEX_dataset/AMEX/element_anno",
                        help="Directory containing AMEX element annotation JSONs")
    parser.add_argument("--weights_dir", type=str,
                        default="/ext_hdd2/nhkoh/OmniParser/weights",
                        help="OmniParser weights directory")
    parser.add_argument("--output_dir", type=str,
                        default="data_engine/mock_unified_app_graph",
                        help="Directory where the unified environment will be saved")
    parser.add_argument("--model_name", type=str, default="gpt-5-mini-2025-08-07",
                        help="OpenAI model used for page styling composition")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for OmniParser/EasyOCR")
    parser.add_argument("--max_trajectories_per_app", type=int, default=None,
                        help="Optional limit for matched trajectories per app")
    parser.add_argument("--include_post_app_steps", action="store_true",
                        help="Keep the full remainder of a trajectory after the app first appears")
    parser.add_argument("--scan_only", action="store_true",
                        help="Only scan annotations and write matched_annotations.json without GPT/detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_unified_graph(args)


if __name__ == "__main__":
    main()
