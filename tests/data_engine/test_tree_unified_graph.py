"""
Tests for Item #1: state fingerprint and spine merge (unified graph).
Run from repo root: python -m pytest tests/data_engine/test_tree_unified_graph.py -v
"""
import json
import sys
import unittest
from typing import Dict, List

# Allow importing data_engine from repo root
import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_engine.tree import (
    SYSTEM_LAYOUT_KEYS,
    state_fingerprint_layout,
    state_fingerprint_page,
    _build_spine_with_merge,
    UIPage,
)


class TestStateFingerprint(unittest.TestCase):
    def test_same_layout_same_fingerprint(self):
        layout = {"icon_a": [10, 20, 50, 60], "icon_b": [100, 100, 150, 150]}
        fp1 = state_fingerprint_layout(layout)
        fp2 = state_fingerprint_layout(dict(layout))
        self.assertEqual(fp1, fp2)

    def test_different_layout_different_fingerprint(self):
        layout_a = {"icon_a": [10, 20, 50, 60]}
        layout_b = {"icon_a": [10, 20, 51, 60]}
        self.assertNotEqual(state_fingerprint_layout(layout_a), state_fingerprint_layout(layout_b))

    def test_system_keys_ignored(self):
        layout_with_system = {"back": [0, 0, 10, 10], "home": [0, 0, 10, 10], "icon_a": [10, 10, 20, 20]}
        layout_content_only = {"icon_a": [10, 10, 20, 20]}
        self.assertEqual(
            state_fingerprint_layout(layout_with_system),
            state_fingerprint_layout(layout_content_only),
        )

    def test_order_independent(self):
        layout1 = {"z": [1, 2, 3, 4], "a": [5, 6, 7, 8]}
        layout2 = {"a": [5, 6, 7, 8], "z": [1, 2, 3, 4]}
        self.assertEqual(state_fingerprint_layout(layout1), state_fingerprint_layout(layout2))

    def test_state_fingerprint_page(self):
        # UIPage has layout: Dict[str, Tuple[int,int]] — actually it's name -> bbox; bbox can be list/tuple
        from data_engine.tree import UIPage
        page = UIPage("p1", [], {"icon_a": (10, 20, 50, 60), "icon_b": (100, 100, 150, 150)}, None)
        fp = state_fingerprint_page(page)
        expected = state_fingerprint_layout({"icon_a": [10, 20, 50, 60], "icon_b": [100, 100, 150, 150]})
        self.assertEqual(fp, expected)


class TestBuildSpineWithMerge(unittest.TestCase):
    def _make_family(
        self,
        page_id: str,
        layout: Dict[str, List[int]],
        next_exists: bool,
        action_name: str = "icon_a",
        action_bbox: List[int] = None,
    ) -> dict:
        action_bbox = action_bbox or (list(layout.values())[0] if layout else [0, 0, 10, 10])
        return {
            "page_id": page_id,
            "layout": layout,
            "canonical_action_name": action_name if next_exists else None,
            "canonical_action_bbox": action_bbox if next_exists else None,
            "page_family_id": f"family_{page_id.replace('page_', '').zfill(3)}",
        }

    def test_no_merge_three_steps(self):
        families = [
            self._make_family("page_0", {"icon_a": [0, 0, 10, 10]}, next_exists=True),
            self._make_family("page_1", {"icon_b": [20, 20, 30, 30]}, next_exists=True, action_name="icon_b", action_bbox=[20, 20, 30, 30]),
            self._make_family("page_2", {"icon_c": [40, 40, 50, 50]}, next_exists=False, action_name="icon_c", action_bbox=[40, 40, 50, 50]),
        ]
        spine_page_ids = [f["page_id"] for f in families]
        pages, tree_children, effective_spine_id = _build_spine_with_merge(families, spine_page_ids)
        self.assertEqual(len(pages), 3)
        self.assertEqual(effective_spine_id, ["page_0", "page_1", "page_2"])
        self.assertIn("page_0", pages)
        self.assertIn("page_1", pages)
        self.assertIn("page_2", pages)
        self.assertEqual(pages["page_0"]["transitions"][0]["target_page"], "page_1")
        self.assertEqual(pages["page_1"]["transitions"][0]["target_page"], "page_2")

    def test_merge_duplicate_middle_returns_to_first(self):
        # Layout A -> B -> A (same as first). Should yield 2 nodes: page_0, page_1; spine 0->1->0
        layout_a = {"icon_a": [0, 0, 10, 10]}
        layout_b = {"icon_b": [20, 20, 30, 30]}
        families = [
            self._make_family("page_0", layout_a, next_exists=True, action_name="icon_a", action_bbox=[0, 0, 10, 10]),
            self._make_family("page_1", layout_b, next_exists=True, action_name="icon_b", action_bbox=[20, 20, 30, 30]),
            self._make_family("page_2", dict(layout_a), next_exists=False, action_name="icon_a", action_bbox=[0, 0, 10, 10]),
        ]
        spine_page_ids = [f["page_id"] for f in families]
        pages, tree_children, effective_spine_id = _build_spine_with_merge(families, spine_page_ids)
        self.assertEqual(len(pages), 2, "Should merge page_2 into page_0")
        self.assertIn("page_0", pages)
        self.assertIn("page_1", pages)
        self.assertNotIn("page_2", pages)
        self.assertEqual(effective_spine_id, ["page_0", "page_1", "page_0"])
        # page_0 -> page_1, page_1 -> page_0
        self.assertEqual(pages["page_0"]["transitions"][0]["target_page"], "page_1")
        self.assertEqual(pages["page_1"]["transitions"][0]["target_page"], "page_0")
        self.assertEqual(tree_children.get("page_0"), ["page_1"])
        self.assertEqual(tree_children.get("page_1"), ["page_0"])

    def test_merge_consecutive_duplicates(self):
        # A -> A -> A: all same layout, should yield 1 node
        layout_a = {"icon_a": [0, 0, 10, 10]}
        families = [
            self._make_family("page_0", layout_a, next_exists=True),
            self._make_family("page_1", dict(layout_a), next_exists=True),
            self._make_family("page_2", dict(layout_a), next_exists=False),
        ]
        spine_page_ids = [f["page_id"] for f in families]
        pages, tree_children, effective_spine_id = _build_spine_with_merge(families, spine_page_ids)
        self.assertEqual(len(pages), 1)
        self.assertEqual(effective_spine_id, ["page_0", "page_0", "page_0"])
        self.assertIn("page_0", pages)
        self.assertNotIn("page_1", pages)
        self.assertNotIn("page_2", pages)
        # page_0 should have two self-transitions (to page_1 and page_2, both merged to page_0)
        self.assertEqual(len(pages["page_0"]["transitions"]), 2)
        self.assertEqual(pages["page_0"]["transitions"][0]["target_page"], "page_0")
        self.assertEqual(pages["page_0"]["transitions"][1]["target_page"], "page_0")

    def test_merge_uses_original_layout_when_set(self):
        """Merge must use _original_layout (pre-mutation) so _choose_click_target adding keys does not break dedup."""
        layout_b = {"click_0": [30, 30, 40, 40]}
        families = [
            self._make_family("page_0", {"click_0": [10, 10, 20, 20], "launcher_button": [0, 0, 10, 10]}, next_exists=True, action_name="launcher_button", action_bbox=[0, 0, 10, 10]),
            self._make_family("page_1", layout_b, next_exists=True, action_name="click_0", action_bbox=[30, 30, 40, 40]),
            self._make_family("page_2", {"click_0": [10, 10, 20, 20]}, next_exists=False, action_name="click_0", action_bbox=[10, 10, 20, 20]),
        ]
        for f in families:
            f["_original_layout"] = {"click_0": [10, 10, 20, 20]} if f["page_id"] in ("page_0", "page_2") else dict(layout_b)
        spine_page_ids = [f["page_id"] for f in families]
        pages, tree_children, effective_spine_id = _build_spine_with_merge(families, spine_page_ids)
        self.assertEqual(len(pages), 2, "page_2 should merge into page_0 via _original_layout")
        self.assertEqual(effective_spine_id, ["page_0", "page_1", "page_0"])


def _layout_from_ui_structure_page(page: dict) -> Dict[str, List[int]]:
    """Convert ui_structure.json page layout (name -> {bbox, type}) to Dict[str, List[int]]."""
    out = {}
    for name, obj in page.get("layout", {}).items():
        bbox = obj.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            out[name] = [int(x) for x in bbox[:4]]
    return out


def _first_spine_action(page: dict) -> tuple:
    """Return (action_name, icon_bbox) from first non-merge transition, or (None, None)."""
    for t in page.get("transitions", []):
        if t.get("transition_role") != "merge":
            return t.get("action"), t.get("icon_bbox", [0, 0, 10, 10])
    return None, [0, 0, 10, 10]


class TestMergeWithRealLayoutData(unittest.TestCase):
    """Integration test: use layout data from datas/ui_structure.json to verify merge."""

    @classmethod
    def _load_ui_structure(cls):
        path = os.path.join(_REPO_ROOT, "datas", "ui_structure.json")
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_merge_with_ui_structure_layouts(self):
        """Build a spine from real page layouts; duplicate one layout so merge happens."""
        data = self._load_ui_structure()
        if data is None:
            self.skipTest("datas/ui_structure.json not found")
        pages_data = data.get("pages", {})
        if len(pages_data) < 3:
            self.skipTest("ui_structure.json has fewer than 3 pages")
        # Use page_0, page_1, then again layout of page_0 so step 2 merges into page_0
        ids = ["page_0", "page_1", "page_2"]
        p0 = pages_data.get("page_0")
        p1 = pages_data.get("page_1")
        p2 = pages_data.get("page_2")
        if not all((p0, p1, p2)):
            self.skipTest("page_0, page_1, page_2 not in ui_structure.json")
        layout0 = _layout_from_ui_structure_page(p0)
        layout1 = _layout_from_ui_structure_page(p1)
        layout2 = _layout_from_ui_structure_page(p2)
        if not layout0 or not layout1:
            self.skipTest("page_0 or page_1 has no layout")
        act0, bbox0 = _first_spine_action(p0)
        act1, bbox1 = _first_spine_action(p1)
        act2, bbox2 = _first_spine_action(p2)
        # Spine: page_0 -> page_1 -> page_2. Force merge by using layout0 for step 2.
        families = [
            {
                "page_id": "page_0",
                "layout": layout0,
                "canonical_action_name": act0,
                "canonical_action_bbox": bbox0 or [0, 0, 10, 10],
                "page_family_id": "family_000",
            },
            {
                "page_id": "page_1",
                "layout": layout1,
                "canonical_action_name": act1,
                "canonical_action_bbox": bbox1 or [0, 0, 10, 10],
                "page_family_id": "family_001",
            },
            {
                "page_id": "page_2",
                "layout": dict(layout0),  # same as page_0 -> merge into page_0
                "canonical_action_name": act2,
                "canonical_action_bbox": bbox2 or [0, 0, 10, 10],
                "page_family_id": "family_002",
            },
        ]
        spine_page_ids = [f["page_id"] for f in families]
        pages, tree_children, effective_spine_id = _build_spine_with_merge(families, spine_page_ids)
        self.assertEqual(len(pages), 2, "page_2 should merge into page_0 (same layout)")
        self.assertIn("page_0", pages)
        self.assertIn("page_1", pages)
        self.assertNotIn("page_2", pages)
        self.assertEqual(effective_spine_id, ["page_0", "page_1", "page_0"])
        self.assertIn("page_1", tree_children.get("page_0", []))
        self.assertIn("page_0", tree_children.get("page_1", []))


if __name__ == "__main__":
    unittest.main()
