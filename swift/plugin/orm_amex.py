import re
from typing import List, Optional, Tuple

from swift.plugin.orm import ORM, orms


def _strip_reasoning(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def _extract_action(text: str) -> Optional[str]:
    match = re.search(r"\tAction:\s*([a-z_]+)", text)
    if not match:
        return None
    return match.group(1).strip().lower()


def _extract_explain(text: str) -> str:
    match = re.search(r"[Ee]xplain:(.*?)\tAction:", text, re.DOTALL)
    if not match:
        return ""
    explain = match.group(1).strip()
    explain = re.sub(r"\s+", " ", explain)
    return explain.rstrip(". ").lower()


def _extract_points(text: str) -> List[Tuple[int, int]]:
    matches = re.findall(r"\((\d+),\s*(\d+)\)", text)
    return [(int(x), int(y)) for x, y in matches]


def _extract_type_text(text: str) -> Optional[str]:
    match = re.search(r"text='((?:\\\\|\\'|[^'])*)'", text)
    if not match:
        return None
    value = match.group(1)
    value = value.replace("\\'", "'").replace("\\\\", "\\")
    return value


def _valid_box(box) -> bool:
    return isinstance(box, (list, tuple)) and len(box) == 4 and any(int(v) != 0 for v in box)


def _point_in_box(point: Tuple[int, int], box) -> bool:
    x, y = point
    x1, y1, x2, y2 = [int(v) for v in box]
    return x1 <= x <= x2 and y1 <= y <= y2


def _coalesce_boxes(primary, fallback, size: int) -> List:
    if primary is not None:
        return primary
    if fallback is not None:
        return fallback
    return [[0, 0, 0, 0]] * size


def _normalize_action_name(action: str) -> str:
    mapping = {
        "TAP": "tap",
        "CLICK": "tap",
        "SWIPE": "swipe",
        "SCROLL": "swipe",
        "TYPE": "type",
        "TEXT": "type",
        "PRESS_ENTER": "press_enter",
        "PRESS_BACK": "press_back",
        "PRESS_HOME": "press_home",
        "TASK_COMPLETE": "complete",
        "TASK_IMPOSSIBLE": "impossible",
        "COMPLETE": "complete",
        "IMPOSSIBLE": "impossible",
    }
    return mapping.get(str(action).strip().upper(), str(action).strip().lower())


def _same_action(pred_action: Optional[str], gold_action: Optional[str], action_type: Optional[str]) -> bool:
    gold = gold_action or _normalize_action_name(action_type or "")
    return pred_action is not None and gold is not None and pred_action == gold


def _requires_coordinate(
    action: Optional[str],
    solution: str,
    bbox,
    start_box,
    end_box,
) -> bool:
    if action in {"press_enter", "complete", "impossible"}:
        return False
    if _extract_points(solution):
        return True
    if action == "tap":
        return _valid_box(bbox)
    if action in {"type", "press_back", "press_home"}:
        return _valid_box(bbox)
    if action == "swipe":
        return _valid_box(start_box) or _valid_box(end_box)
    return False


class AMEXActionMatchORM(ORM):
    def __call__(self, completions, solution, action_type=None, **kwargs) -> List[float]:
        rewards = []
        action_type = action_type or [None] * len(completions)
        for content, sol, gold_action_type in zip(completions, solution, action_type):
            content = _strip_reasoning(content)
            sol = _strip_reasoning(sol)
            pred_action = _extract_action(content)
            sol_action = _extract_action(sol)
            rewards.append(float(_same_action(pred_action, sol_action, gold_action_type)))
        return rewards


class AMEXCoordinateMatchORM(ORM):
    def __call__(
        self,
        completions,
        solution,
        bbox_px=None,
        start_box_px=None,
        end_box_px=None,
        bbox_norm=None,
        start_box_norm=None,
        end_box_norm=None,
        action_type=None,
        **kwargs,
    ) -> List[float]:
        rewards = []
        bbox_boxes = _coalesce_boxes(bbox_px, bbox_norm, len(completions))
        start_boxes = _coalesce_boxes(start_box_px, start_box_norm, len(completions))
        end_boxes = _coalesce_boxes(end_box_px, end_box_norm, len(completions))
        action_type = action_type or [None] * len(completions)

        for content, sol, bbox, start_box, end_box, gold_action_type in zip(
                completions, solution, bbox_boxes, start_boxes, end_boxes, action_type):
            content = _strip_reasoning(content)
            sol = _strip_reasoning(sol)
            pred_action = _extract_action(content)
            sol_action = _extract_action(sol)

            if not _same_action(pred_action, sol_action, gold_action_type):
                rewards.append(0.0)
                continue

            points = _extract_points(content)
            requires_coord = _requires_coordinate(pred_action, sol, bbox, start_box, end_box)

            if not requires_coord:
                rewards.append(float(len(points) == 0))
                continue

            if pred_action == "tap":
                if not points:
                    rewards.append(0.0)
                    continue
                rewards.append(float(_valid_box(bbox) and _point_in_box(points[0], bbox)))
                continue

            if pred_action in {"type", "press_back", "press_home"}:
                if not points:
                    rewards.append(0.0)
                    continue
                rewards.append(float(_valid_box(bbox) and _point_in_box(points[0], bbox)))
                continue

            if pred_action == "swipe":
                if len(points) < 2:
                    rewards.append(0.0)
                    continue
                start_ok = not _valid_box(start_box) or _point_in_box(points[0], start_box)
                end_ok = not _valid_box(end_box) or _point_in_box(points[1], end_box)
                rewards.append(float(start_ok and end_ok))
                continue

            rewards.append(0.0)

        return rewards


def _extract_tap_target(explain: str) -> Optional[str]:
    """Extract element name from 'tap X' explain."""
    match = re.match(r"tap\s+(.+)", explain)
    if match:
        return match.group(1).strip().rstrip(".")
    return None


def _extract_swipe_direction(explain: str) -> Optional[str]:
    """Extract direction from 'swipe up/down/left/right' explain."""
    match = re.search(r"swipe\s+(up|down|left|right)", explain)
    if match:
        return match.group(1)
    return None


def _extract_type_content(explain: str) -> Optional[str]:
    """Extract quoted text from 'type "X" into ...' explain."""
    match = re.search(r'type\s+["\u201c](.+?)["\u201d]', explain)
    if match:
        return match.group(1)
    return None


class AMEXIntentMatchORM(ORM):
    def __call__(self, completions, solution, action_type=None, type_text=None, **kwargs) -> List[float]:
        rewards = []
        action_type = action_type or [None] * len(completions)
        type_text = type_text or [""] * len(completions)

        for content, sol, gold_action_type, gold_type_text in zip(completions, solution, action_type, type_text):
            content = _strip_reasoning(content)
            sol = _strip_reasoning(sol)
            pred_action = _extract_action(content)
            sol_action = _extract_action(sol)

            if not _same_action(pred_action, sol_action, gold_action_type):
                rewards.append(0.0)
                continue

            pred_explain = _extract_explain(content)
            sol_explain = _extract_explain(sol)

            if pred_action == "tap":
                pred_target = _extract_tap_target(pred_explain)
                sol_target = _extract_tap_target(sol_explain)
                if pred_target and sol_target:
                    rewards.append(float(pred_target == sol_target))
                else:
                    rewards.append(0.0)

            elif pred_action == "swipe":
                pred_dir = _extract_swipe_direction(pred_explain)
                sol_dir = _extract_swipe_direction(sol_explain)
                if pred_dir and sol_dir:
                    rewards.append(float(pred_dir == sol_dir))
                else:
                    rewards.append(0.0)

            elif pred_action == "type":
                pred_text = _extract_type_content(pred_explain)
                if pred_text and gold_type_text:
                    rewards.append(float(pred_text.lower() == gold_type_text.lower()))
                elif gold_type_text:
                    pred_text2 = _extract_type_text(content)
                    rewards.append(float(pred_text2 is not None and pred_text2.lower() == gold_type_text.lower()))
                else:
                    rewards.append(0.0)

            elif pred_action in ("complete", "task"):
                rewards.append(float("complete" in pred_explain or "target" in pred_explain or "finish" in pred_explain))

            elif pred_action == "impossible":
                rewards.append(float("cannot" in pred_explain or "impossible" in pred_explain))

            elif pred_action in ("press_enter", "press_back", "press_home"):
                rewards.append(1.0)

            else:
                rewards.append(0.0)

        return rewards


class AMEXFormatConstraintORM(ORM):
    def __init__(self):
        prefix = r"^(?:<think>.*?</think>\s*)?[Ee]xplain:.*?\tAction:\s*"
        point = r"'<\|box_start\|>\(\d+,\s*\d+\)<\|box_end\|>'"
        self.patterns = [
            re.compile(prefix + rf"tap\(start_box={point}\)$"),
            re.compile(prefix + rf"swipe\(start_box={point},\s*end_box={point}\)$"),
            re.compile(prefix + rf"type\(start_box={point},\s*text='(?:\\\\|\\'|[^'])*'\)$"),
            re.compile(prefix + r"press_enter\(\)$"),
            re.compile(prefix + rf"press_back\(\)$"),
            re.compile(prefix + rf"press_back\(start_box={point}\)$"),
            re.compile(prefix + rf"press_home\(\)$"),
            re.compile(prefix + rf"press_home\(start_box={point}\)$"),
            re.compile(prefix + r"complete$"),
            re.compile(prefix + r"impossible$"),
        ]

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            text = completion.strip()
            rewards.append(float(any(pattern.fullmatch(text) for pattern in self.patterns)))
        return rewards


orms["amex_action_match"] = AMEXActionMatchORM
orms["amex_coordinate_match"] = AMEXCoordinateMatchORM
orms["amex_intent_match"] = AMEXIntentMatchORM
orms["amex_format_constraint"] = AMEXFormatConstraintORM
