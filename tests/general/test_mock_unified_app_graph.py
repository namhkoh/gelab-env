from pathlib import Path
import sys


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
