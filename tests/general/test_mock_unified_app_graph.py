from pathlib import Path
import sys
import time
from types import SimpleNamespace

from PIL import Image


sys.path.insert(0, "/home/tsyou/gelab-env/data_engine")

import mock_unified_app_graph as mug


def _page(page_id: str, depth: int, targets: list[str]) -> dict:
    transitions = []
    for idx, target_page in enumerate(targets):
        transitions.append({
            "action": "tap",
            "raw_action": "TAP",
            "target_page": target_page,
            "icon_bbox": [32 + idx * 40, 32, 64 + idx * 40, 64],
        })
    return {
        "page_id": page_id,
        "page_name": page_id,
        "image": f"{page_id}.png",
        "depth": depth,
        "layout": {"back": [0, 0, 16, 16], "home": [100, 0, 116, 16]},
        "transitions": transitions,
        "application_id": "novelship",
        "application_name": "Novelship",
        "page_summary": {
            "application_id": "novelship",
            "application_name": "Novelship",
            "page_family": "content_page",
        },
    }


def test_future_path_split_starts_from_page_4_transition(tmp_path: Path):
    pages = {
        "page_4": _page("page_4", 3, ["page_9"]),
        "page_9": _page("page_9", 4, ["page_15"]),
        "page_15": _page("page_15", 5, ["page_20"]),
        "page_20": _page("page_20", 6, ["page_26", "page_27"]),
        "page_26": _page("page_26", 7, []),
        "page_27": _page("page_27", 7, []),
    }

    matched_rows = [
        {
            "page_id": "page_4",
            "trajectory_id_full": "traj_a",
            "episode_id": "ep_a",
            "step_index": 1,
            "screenshot": "a_1.png",
        },
        {
            "page_id": "page_9",
            "trajectory_id_full": "traj_a",
            "episode_id": "ep_a",
            "step_index": 2,
            "screenshot": "a_2.png",
        },
        {
            "page_id": "page_15",
            "trajectory_id_full": "traj_a",
            "episode_id": "ep_a",
            "step_index": 3,
            "screenshot": "a_3.png",
        },
        {
            "page_id": "page_20",
            "trajectory_id_full": "traj_a",
            "episode_id": "ep_a",
            "step_index": 4,
            "screenshot": "a_4.png",
        },
        {
            "page_id": "page_26",
            "trajectory_id_full": "traj_a",
            "episode_id": "ep_a",
            "step_index": 5,
            "screenshot": "a_5.png",
        },
        {
            "page_id": "page_4",
            "trajectory_id_full": "traj_b",
            "episode_id": "ep_b",
            "step_index": 1,
            "screenshot": "b_1.png",
        },
        {
            "page_id": "page_9",
            "trajectory_id_full": "traj_b",
            "episode_id": "ep_b",
            "step_index": 2,
            "screenshot": "b_2.png",
        },
        {
            "page_id": "page_15",
            "trajectory_id_full": "traj_b",
            "episode_id": "ep_b",
            "step_index": 3,
            "screenshot": "b_3.png",
        },
        {
            "page_id": "page_20",
            "trajectory_id_full": "traj_b",
            "episode_id": "ep_b",
            "step_index": 4,
            "screenshot": "b_4.png",
        },
        {
            "page_id": "page_27",
            "trajectory_id_full": "traj_b",
            "episode_id": "ep_b",
            "step_index": 5,
            "screenshot": "b_5.png",
        },
    ]
    asset_rows = [
        {
            "page_id": row["page_id"],
            "episode_id": row["episode_id"],
            "screenshot": row["screenshot"],
            "type": "icon",
            "label": "shared",
        }
        for row in matched_rows
    ]

    expanded = mug._expand_topology_pages_by_trajectory_asset_clusters(
        output_dir=tmp_path,
        pages=pages,
        matched_rows=matched_rows,
        asset_rows=asset_rows,
    )

    page_4_targets = [
        transition["target_page"]
        for transition in expanded["page_4"]["transitions"]
    ]

    assert len(expanded) > len(pages)
    assert "page_9" in page_4_targets
    assert len(page_4_targets) == 2
    assert any(target_page != "page_9" for target_page in page_4_targets)


def test_renumber_copies_assets_for_split_clone_pages(tmp_path: Path):
    pages = {
        "page_4": _page("page_4", 3, ["page_9"]),
        "page_9": _page("page_9", 4, ["page_15"]),
        "page_15": _page("page_15", 5, ["page_20"]),
        "page_20": _page("page_20", 6, ["page_26", "page_27"]),
        "page_26": _page("page_26", 7, []),
        "page_27": _page("page_27", 7, []),
    }
    matched_rows = [
        {"page_id": "page_4", "trajectory_id_full": "traj_a", "episode_id": "ep_a", "step_index": 1, "screenshot": "a_1.png"},
        {"page_id": "page_9", "trajectory_id_full": "traj_a", "episode_id": "ep_a", "step_index": 2, "screenshot": "a_2.png"},
        {"page_id": "page_15", "trajectory_id_full": "traj_a", "episode_id": "ep_a", "step_index": 3, "screenshot": "a_3.png"},
        {"page_id": "page_20", "trajectory_id_full": "traj_a", "episode_id": "ep_a", "step_index": 4, "screenshot": "a_4.png"},
        {"page_id": "page_26", "trajectory_id_full": "traj_a", "episode_id": "ep_a", "step_index": 5, "screenshot": "a_5.png"},
        {"page_id": "page_4", "trajectory_id_full": "traj_b", "episode_id": "ep_b", "step_index": 1, "screenshot": "b_1.png"},
        {"page_id": "page_9", "trajectory_id_full": "traj_b", "episode_id": "ep_b", "step_index": 2, "screenshot": "b_2.png"},
        {"page_id": "page_15", "trajectory_id_full": "traj_b", "episode_id": "ep_b", "step_index": 3, "screenshot": "b_3.png"},
        {"page_id": "page_20", "trajectory_id_full": "traj_b", "episode_id": "ep_b", "step_index": 4, "screenshot": "b_4.png"},
        {"page_id": "page_27", "trajectory_id_full": "traj_b", "episode_id": "ep_b", "step_index": 5, "screenshot": "b_5.png"},
    ]
    asset_rows = [
        {"page_id": row["page_id"], "episode_id": row["episode_id"], "screenshot": row["screenshot"], "type": "icon", "label": "shared"}
        for row in matched_rows
    ]
    expanded = mug._expand_topology_pages_by_trajectory_asset_clusters(
        output_dir=tmp_path,
        pages=pages,
        matched_rows=matched_rows,
        asset_rows=asset_rows,
    )

    pages_dir = tmp_path / "pages"
    code_dir = tmp_path / "generated_code"
    pages_dir.mkdir()
    code_dir.mkdir()
    for page_id in pages:
        (pages_dir / f"{page_id}.png").write_bytes(b"png")
        (code_dir / f"{page_id}.py").write_text(f"# {page_id}\n", encoding="utf-8")

    renumbered_pages, _, _ = mug._renumber_page_outputs(
        output_dir=tmp_path,
        pages=expanded,
        asset_manifest=[],
        matched_step_rows=[],
        root_page_id="page_4",
    )

    for page_id in renumbered_pages:
        assert (pages_dir / f"{page_id}.png").exists()
        assert (code_dir / f"{page_id}.py").exists()


def test_topology_contracts_same_page_split_group(tmp_path: Path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    for page_id, color in {
        "page_4": (255, 0, 0),
        "page_9": (0, 255, 0),
        "page_18": (0, 255, 0),
        "page_17": (0, 0, 255),
        "page_16": (255, 255, 0),
    }.items():
        Image.new("RGB", (32, 32), color).save(pages_dir / f"{page_id}.png")

    pages = {
        "page_4": _page("page_4", 3, ["page_9", "page_18"]),
        "page_9": _page("page_9", 4, ["page_17"]),
        "page_18": _page("page_18", 4, ["page_16"]),
        "page_17": _page("page_17", 5, []),
        "page_16": _page("page_16", 5, []),
    }
    pages["page_18"]["topology_split_from_page_id"] = "page_9"

    contracted = mug._contract_same_page_split_groups_for_topology(tmp_path, pages)
    recomputed = mug._recompute_topology_depths(contracted, root_page_id="page_4")

    assert set(contracted) == {"page_4", "page_16", "page_17"}
    assert [t["target_page"] for t in contracted["page_4"]["transitions"]] == ["page_17", "page_16"]
    assert recomputed["page_4"]["depth"] == 0
    assert recomputed["page_16"]["depth"] == 1
    assert recomputed["page_17"]["depth"] == 1


def test_split_clone_preserves_source_trace_and_parent_after_renumber(tmp_path: Path):
    pages = {
        "page_4": _page("page_4", 3, ["page_9", "page_9"]),
        "page_9": _page("page_9", 4, ["page_16", "page_17"]),
        "page_16": _page("page_16", 5, []),
        "page_17": _page("page_17", 5, []),
    }
    matched_rows = [
        {"page_id": "page_4", "trajectory_id_full": "traj_a", "episode_id": "ep_a", "step_index": 1, "screenshot": "a_1.png"},
        {"page_id": "page_9", "trajectory_id_full": "traj_a", "episode_id": "ep_a", "step_index": 2, "screenshot": "a_2.png"},
        {"page_id": "page_16", "trajectory_id_full": "traj_a", "episode_id": "ep_a", "step_index": 3, "screenshot": "a_3.png"},
        {"page_id": "page_4", "trajectory_id_full": "traj_b", "episode_id": "ep_b", "step_index": 1, "screenshot": "b_1.png"},
        {"page_id": "page_9", "trajectory_id_full": "traj_b", "episode_id": "ep_b", "step_index": 2, "screenshot": "b_2.png"},
        {"page_id": "page_17", "trajectory_id_full": "traj_b", "episode_id": "ep_b", "step_index": 3, "screenshot": "b_3.png"},
    ]
    asset_rows = [
        {
            "page_id": row["page_id"],
            "episode_id": row["episode_id"],
            "screenshot": row["screenshot"],
            "type": "icon",
            "label": "shared",
        }
        for row in matched_rows
    ]

    expanded = mug._expand_topology_pages_by_trajectory_asset_clusters(
        output_dir=tmp_path,
        pages=pages,
        matched_rows=matched_rows,
        asset_rows=asset_rows,
    )
    split_page_id = next(
        page_id
        for page_id, page in expanded.items()
        if page_id != "page_9" and page.get("topology_split_from_page_id") == "page_9"
    )
    assert expanded[split_page_id]["transitions"][0]["source_trace_page"] == split_page_id

    pages_dir = tmp_path / "pages"
    code_dir = tmp_path / "generated_code"
    pages_dir.mkdir()
    code_dir.mkdir()
    for page_id in pages:
        Image.new("RGB", (16, 16), (255, 255, 255)).save(pages_dir / f"{page_id}.png")
        (code_dir / f"{page_id}.py").write_text("# x\n", encoding="utf-8")

    renumbered_pages, _, page_id_map = mug._renumber_page_outputs(
        output_dir=tmp_path,
        pages=expanded,
        asset_manifest=[],
        matched_step_rows=[],
        root_page_id="page_4",
    )
    renumbered_split_page = next(
        page
        for page in renumbered_pages.values()
        if page.get("topology_split_from_page_id")
    )
    assert renumbered_split_page["topology_split_from_page_id"] == page_id_map["page_9"]


def test_topology_same_page_collapse_keeps_branch_pages_with_different_future_states(
    tmp_path: Path,
    monkeypatch,
):
    pages = {
        "page_4": _page("page_4", 3, ["page_9", "page_18"]),
        "page_9": _page("page_9", 4, ["page_17"]),
        "page_18": _page("page_18", 4, ["page_16"]),
        "page_16": _page("page_16", 5, []),
        "page_17": _page("page_17", 5, []),
    }
    pages["page_9"]["transitions"][0]["target_element"] = "Search"
    pages["page_18"]["transitions"][0]["target_element"] = "Search"
    pages["page_16"]["layout"]["content"] = [32, 96, 320, 192]
    pages["page_17"]["layout"]["content"] = [32, 96, 320, 320]

    monkeypatch.setattr(mug, "_looks_like_same_page", lambda *args, **kwargs: True)

    same_page_state = mug._build_topology_same_page_state(
        pages,
        output_dir=tmp_path,
        metadata=None,
    )
    collapsed_pages, _ = mug._collapse_topology_pages(
        pages,
        root_page_id="page_4",
        same_page_state=same_page_state,
        metadata=None,
    )

    page_4_targets = [
        transition["target_page"]
        for transition in collapsed_pages["page_4"]["transitions"]
    ]
    grouped_pairs = {
        tuple(group["page_ids"])
        for group in same_page_state["groups"]
    }

    assert ("page_9", "page_18") not in grouped_pairs
    assert set(page_4_targets) == {"page_9", "page_18"}


def test_recompute_output_page_depths_compacts_sparse_step_based_depths():
    pages = {
        mug.HOME_PAGE_ID: {
            "page_id": mug.HOME_PAGE_ID,
            "page_name": "Home",
            "image": f"{mug.HOME_PAGE_ID}.png",
            "depth": 0,
            "layout": {},
            "transitions": [{"action": "swipe", "raw_action": "SWIPE", "target_page": mug.DRAWER_PAGE_ID}],
            "application_id": "launcher",
            "application_name": "Launcher",
            "page_summary": {"application_id": "launcher", "application_name": "Launcher"},
        },
        mug.DRAWER_PAGE_ID: {
            "page_id": mug.DRAWER_PAGE_ID,
            "page_name": "Drawer",
            "image": f"{mug.DRAWER_PAGE_ID}.png",
            "depth": 1,
            "layout": {},
            "transitions": [{"action": "tap", "raw_action": "TAP", "target_page": "page_4"}],
            "application_id": "launcher",
            "application_name": "Launcher",
            "page_summary": {"application_id": "launcher", "application_name": "Launcher"},
        },
        "page_4": _page("page_4", 2, ["page_9"]),
        "page_9": _page("page_9", 13, ["page_15"]),
        "page_15": _page("page_15", 14, ["page_20"]),
        "page_20": _page("page_20", 15, []),
    }

    normalized = mug._recompute_output_page_depths(pages, mug.HOME_PAGE_ID)

    assert normalized[mug.HOME_PAGE_ID]["depth"] == 0
    assert normalized[mug.DRAWER_PAGE_ID]["depth"] == 1
    assert normalized["page_4"]["depth"] == 2
    assert normalized["page_9"]["depth"] == 3
    assert normalized["page_15"]["depth"] == 4
    assert normalized["page_20"]["depth"] == 5


def test_extract_drawer_apps_keeps_launcher_page_id_for_second_drawer():
    launcher_pages = {
        mug.DRAWER_PAGE_ID: {
            "layout": {"SeatGeek": [10, 10, 70, 70]},
        },
        "page_2_app_drawer": {
            "layout": {"Music": [120, 120, 240, 240]},
        },
    }
    drawer_page_specs = [
        {
            "page_id": mug.DRAWER_PAGE_ID,
            "icons": [{"label": "SeatGeek", "asset": "seatgeek"}],
        },
        {
            "page_id": "page_2_app_drawer",
            "icons": [{"label": "Music", "asset": "music_real"}],
        },
    ]

    apps = mug._extract_drawer_apps(drawer_page_specs, launcher_pages)

    assert [(app.slug, app.launcher_page_id) for app in apps] == [
        ("seatgeek", mug.DRAWER_PAGE_ID),
        ("music", "page_2_app_drawer"),
    ]
    assert apps[1].bbox == [120, 120, 240, 240]


def test_build_rich_launcher_pages_routes_tap_from_second_drawer_page():
    mug._ensure_compose_modules()

    launcher_pages = {
        mug.HOME_PAGE_ID: {
            "page_id": mug.HOME_PAGE_ID,
            "image": f"{mug.HOME_PAGE_ID}.png",
            "depth": 0,
            "layout": {"home": [940, 20, 1020, 100]},
            "transitions": [
                {
                    "action": "swipe",
                    "target_page": mug.DRAWER_PAGE_ID,
                    "action_coord": [540, 2343],
                    "lift_coord": [540, 1460],
                    "icon_bbox": [500, 1460, 580, 2343],
                    "gesture_direction": "up",
                },
                {
                    "action": "PRESS_HOME",
                    "target_page": mug.HOME_PAGE_ID,
                    "action_coord": [980, 60],
                    "icon_bbox": [940, 20, 1020, 100],
                },
            ],
        },
        mug.DRAWER_PAGE_ID: {
            "page_id": mug.DRAWER_PAGE_ID,
            "image": f"{mug.DRAWER_PAGE_ID}.png",
            "depth": 1,
            "layout": {"back": [60, 20, 140, 100], "home": [940, 20, 1020, 100]},
            "transitions": [
                {
                    "action": "PRESS_BACK",
                    "target_page": mug.HOME_PAGE_ID,
                    "action_coord": [100, 60],
                    "icon_bbox": [60, 20, 140, 100],
                },
                {
                    "action": "PRESS_HOME",
                    "target_page": mug.HOME_PAGE_ID,
                    "action_coord": [980, 60],
                    "icon_bbox": [940, 20, 1020, 100],
                },
                {
                    "action": "swipe",
                    "target_page": "page_2_app_drawer",
                    "action_coord": [540, 2140],
                    "lift_coord": [540, 980],
                    "icon_bbox": [500, 980, 580, 2140],
                    "gesture_direction": "up",
                },
            ],
        },
        "page_2_app_drawer": {
            "page_id": "page_2_app_drawer",
            "image": "page_2_app_drawer.png",
            "depth": 2,
            "layout": {
                "back": [60, 20, 140, 100],
                "home": [940, 20, 1020, 100],
                "Music": [120, 120, 240, 240],
            },
            "transitions": [
                {
                    "action": "PRESS_BACK",
                    "target_page": mug.HOME_PAGE_ID,
                    "action_coord": [100, 60],
                    "icon_bbox": [60, 20, 140, 100],
                },
                {
                    "action": "PRESS_HOME",
                    "target_page": mug.HOME_PAGE_ID,
                    "action_coord": [980, 60],
                    "icon_bbox": [940, 20, 1020, 100],
                },
                {
                    "action": "swipe",
                    "target_page": mug.DRAWER_PAGE_ID,
                    "action_coord": [540, 980],
                    "lift_coord": [540, 2140],
                    "icon_bbox": [500, 980, 580, 2140],
                    "gesture_direction": "down",
                },
            ],
        },
    }
    drawer_apps = [
        mug.DrawerAppSpec(
            label="Music",
            asset="music_real",
            slug="music",
            layout_key="Music",
            bbox=[120, 120, 240, 240],
            launcher_page_id="page_2_app_drawer",
            match_tokens={"music"},
        )
    ]

    pages = mug._build_rich_launcher_pages(
        launcher_pages=launcher_pages,
        drawer_apps=drawer_apps,
        app_entry_pages={"music": ["page_music_entry"]},
    )

    page_2_transitions = pages["page_2_app_drawer"]["transitions"]
    swipe_targets = [
        transition["target_page"]
        for transition in page_2_transitions
        if transition["action_kind"] == "swipe"
    ]
    tap_targets = [
        transition["target_page"]
        for transition in page_2_transitions
        if transition["action_kind"] == "tap"
    ]

    assert mug.DRAWER_PAGE_ID in swipe_targets
    assert tap_targets == ["page_music_entry"]
    assert page_2_transitions[-1]["canvas_action_point"] == [180, 180]


def test_serialize_transition_keeps_type_spatial_anchor():
    mug._ensure_compose_modules()

    serialized = mug._serialize_transition({
        "raw_action": "TYPE",
        "action": "Search_Field",
        "target_page": "page_b",
        "canvas_action_bbox": [0, 0, 0, 0],
        "canvas_action_point": [0, 0],
        "canvas_lift_coord": [0, 0],
        "icon_bbox": [20, 40, 60, 80],
        "type_text": "hello",
        "gesture_direction": "",
        "source_trace_page": "page_src",
        "source_trajectory_id": "traj_1",
        "source_step_indices": [3],
    })

    assert serialized["action"] == "type"
    assert serialized["target_element"] == "Search_Field"
    assert serialized["action_coord"] == [40, 60]
    assert serialized["icon_bbox"] == [20, 40, 60, 80]
    assert serialized["source_trace_page"] == "page_src"
    assert serialized["source_trajectory_id"] == "traj_1"
    assert serialized["source_step_indices"] == [3]


def test_action_debug_overlay_draws_swipe_path_and_type_bbox(tmp_path: Path):
    image_path = tmp_path / "page.png"
    overlay_path = tmp_path / "overlay.png"
    Image.new("RGB", (120, 120), (255, 255, 255)).save(image_path)

    mug._ac_save_action_debug_overlay(
        str(image_path),
        str(overlay_path),
        [
            {
                "raw_action": "SWIPE",
                "action": "swipe",
                "target_page": "page_b",
                "canvas_action_bbox": [16, 20, 24, 100],
                "canvas_action_point": [20, 20],
                "canvas_lift_coord": [20, 100],
                "icon_bbox": [16, 20, 24, 100],
                "type_text": "",
                "gesture_direction": "down",
            },
            {
                "raw_action": "TYPE",
                "action": "Search_Field",
                "target_page": "page_c",
                "canvas_action_bbox": [0, 0, 0, 0],
                "canvas_action_point": [0, 0],
                "canvas_lift_coord": [0, 0],
                "icon_bbox": [60, 30, 100, 60],
                "type_text": "hello",
                "gesture_direction": "",
            },
        ],
    )

    overlay = Image.open(overlay_path).convert("RGB")
    assert overlay.getpixel((20, 60)) != (255, 255, 255)
    assert overlay.getpixel((60, 30)) != (255, 255, 255)


def test_build_topology_graph_payload_includes_spatial_transition_metadata():
    mug._ensure_compose_modules()
    pages = {
        "page_a": {
            "page_id": "page_a",
            "page_name": "A",
            "image": "page_a.png",
            "depth": 0,
            "layout": {"Search_Field": [20, 40, 60, 80]},
            "transitions": [
                {
                    "raw_action": "TYPE",
                    "action": "Search_Field",
                    "target_page": "page_b",
                    "canvas_action_bbox": [0, 0, 0, 0],
                    "canvas_action_point": [0, 0],
                    "canvas_lift_coord": [0, 0],
                    "icon_bbox": [20, 40, 60, 80],
                    "type_text": "hello",
                    "gesture_direction": "",
                    "source_trace_page": "page_src",
                    "source_trajectory_id": "traj_1",
                    "source_step_indices": [3],
                }
            ],
            "application_id": "novelship",
            "application_name": "Novelship",
            "page_summary": {"application_id": "novelship", "application_name": "Novelship"},
        },
        "page_b": {
            "page_id": "page_b",
            "page_name": "B",
            "image": "page_b.png",
            "depth": 1,
            "layout": {},
            "transitions": [],
            "application_id": "novelship",
            "application_name": "Novelship",
            "page_summary": {"application_id": "novelship", "application_name": "Novelship"},
        },
    }

    graph = mug._build_topology_graph_payload(pages, root_page_id="page_a")
    edge = graph["edges"][0]

    assert edge["action"] == "type"
    assert edge["spatial_anchor_valid"] is True
    assert edge["spatial_anchor_type"] == "point"
    assert edge["spatial_path_valid"] is True
    assert edge["action_coord"] == [40, 60]
    assert edge["icon_bbox"] == [20, 40, 60, 80]
    assert edge["source_trace_page"] == "page_src"
    assert edge["source_trajectory_id"] == "traj_1"
    assert edge["source_step_indices"] == [3]


def test_parse_args_accepts_api_concurrency(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["mock_unified_app_graph.py", "--api_concurrency", "4"],
    )

    args = mug.parse_args()

    assert args.api_concurrency == 4


def test_compose_segment_pages_parallel_branch_preserves_page_order(
    tmp_path: Path,
    monkeypatch,
):
    annotation_path = tmp_path / "trajectory.json"
    annotation_path.write_text(
        """
        {
          "episode_id": "ep_parallel",
          "instruction": "Open Novelship",
          "steps": [
            {"action": "TAP", "package_name": "com.novelship", "touch_coord": [10, 10], "lift_coord": [10, 10]},
            {"action": "TAP", "package_name": "com.novelship", "touch_coord": [20, 20], "lift_coord": [20, 20]},
            {"action": "PRESS_HOME", "package_name": "com.novelship", "touch_coord": [0, 0], "lift_coord": [0, 0]}
          ]
        }
        """,
        encoding="utf-8",
    )

    screenshots_dir = tmp_path / "screens"
    screenshots_dir.mkdir()
    for idx in range(3):
        (screenshots_dir / f"ep_parallel_{idx}.png").write_bytes(b"png")

    monkeypatch.setattr(
        mug,
        "_step_context_for_segment",
        lambda segment, idx: {"step_index": idx + 1},
    )

    class StubActionCompose:
        GELAB_BACK_BBOX = [0, 0, 8, 8]
        GELAB_HOME_BBOX = [12, 0, 20, 8]

        @staticmethod
        def _resolve_step_screenshot(step, screenshots_dir, episode_id, local_idx):
            path = Path(screenshots_dir) / f"{episode_id}_{local_idx}.png"
            return path.name, str(path)

        @staticmethod
        def detect_and_crop(screenshot_path, yolo_model, ocr_reader):
            return [], (100, 200)

        @staticmethod
        def _prioritize_element_anno_bboxes(elements, screenshot_path, screenshot_name, element_anno_dir):
            return elements, {"loaded": 0, "matched": 0, "added": 0}

        @staticmethod
        def _persist_extracted_assets(elements, screenshot_name, assets_dir, step_info):
            return []

        @staticmethod
        def _resolve_transition(step, layout, orig_size, target_page):
            return {
                "raw_action": str(step.get("action", "")),
                "action": "tap",
                "target_page": target_page,
                "canvas_action_bbox": [1, 1, 2, 2],
                "canvas_action_point": [1, 1],
                "canvas_lift_coord": [0, 0],
                "icon_bbox": [1, 1, 2, 2],
                "type_text": "",
                "gesture_direction": "",
            }

        @staticmethod
        def _build_system_transition(raw_action, action, target_page, icon_bbox):
            return {
                "raw_action": raw_action,
                "action": action,
                "target_page": target_page,
                "canvas_action_bbox": [0, 0, 0, 0],
                "canvas_action_point": [0, 0],
                "canvas_lift_coord": [0, 0],
                "icon_bbox": icon_bbox,
                "type_text": "",
                "gesture_direction": "",
            }

    monkeypatch.setattr(mug, "action_compose", StubActionCompose)

    call_flags = []
    delays = {"01": 0.04, "02": 0.01, "03": 0.02}

    def fake_compose_segment_page_record(page_job, pages_dir, code_dir, model_name, client=None, use_thread_client=False):
        time.sleep(delays[page_job["page_id"][-2:]])
        call_flags.append(use_thread_client)
        return {
            "message": f"done {page_job['page_id']}",
            "page_row": {
                "page_id": page_job["page_id"],
                "image": f"{page_job['page_id']}.png",
                "depth": page_job["depth"],
                "layout": {
                    "content": [10, 10, 30, 30],
                    "back": [0, 0, 8, 8],
                    "home": [12, 0, 20, 8],
                },
                "orig_size": tuple(page_job["orig_size"]),
                "step": page_job["step"],
                "step_context": page_job["step_context"],
                "episode_id": page_job["episode_id"],
                "page_name": page_job["page_name"],
                "application_id": page_job["application_id"],
                "application_name": page_job["application_name"],
                "trajectory_ids": [page_job["episode_id"]],
                "trajectory_ids_full": [page_job["trajectory_id_full"]],
                "trace_steps": [page_job["trace_step"]],
                "anno_stats": page_job["anno_stats"],
            },
        }

    monkeypatch.setattr(mug, "_compose_segment_page_record", fake_compose_segment_page_record)

    match = mug.MatchedTrajectory(
        app_slug="novelship",
        app_label="Novelship",
        annotation_path=str(annotation_path),
        episode_id="ep_parallel",
        instruction="Open Novelship",
        start_step_idx=0,
        end_step_idx=3,
        matched_package="com.novelship",
        total_steps=3,
    )
    app = mug.DrawerAppSpec(
        label="Novelship",
        asset="novelship",
        slug="novelship",
        layout_key="novelship",
        bbox=[0, 0, 32, 32],
        launcher_page_id=mug.DRAWER_PAGE_ID,
        match_tokens={"novelship"},
    )
    args = SimpleNamespace(
        screenshots_dir=str(screenshots_dir),
        element_anno_dir=str(tmp_path / "element_anno"),
        api_concurrency=3,
    )

    page_rows, manifest_rows, matched_rows = mug._compose_segment_pages(
        match=match,
        app=app,
        app_root_page_id=mug.DRAWER_PAGE_ID,
        home_page_id=mug.HOME_PAGE_ID,
        args=args,
        client=object(),
        model_name="gpt-test",
        yolo_model=object(),
        ocr_reader=object(),
        output_dir=tmp_path,
    )

    assert [page["page_id"] for page in page_rows] == [
        "page_novelship_ep_parallel_01",
        "page_novelship_ep_parallel_02",
        "page_novelship_ep_parallel_03",
    ]
    assert page_rows[0]["transitions"][0]["target_page"] == "page_novelship_ep_parallel_02"
    assert page_rows[1]["transitions"][0]["target_page"] == "page_novelship_ep_parallel_03"
    assert page_rows[2]["transitions"][0]["target_page"] == mug.HOME_PAGE_ID
    assert call_flags == [True, True, True]
    assert manifest_rows == []
    assert len(matched_rows) == 3
