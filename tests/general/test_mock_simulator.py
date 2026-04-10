import json
import sys
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, "/home/tsyou/gelab-env/data_engine")

import mock_unified_app_graph as mug


def _write_annotation(annotation_path: Path, package_name: str) -> None:
    annotation_path.write_text(
        json.dumps(
            {
                "instruction": f"Open {package_name}",
                "episode_id": annotation_path.stem,
                "steps": [
                    {"package_name": "com.android.launcher3", "action": "SWIPE"},
                    {"package_name": package_name, "action": "TAP"},
                    {"package_name": package_name, "action": "TASK_COMPLETE"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_layout_config_populates_drawer_from_annotations_and_paginates(tmp_path: Path):
    annotations_dir = tmp_path / "instruction_anno"
    annotations_dir.mkdir()

    packages = ["com.seatgeek.android"] + [f"com.example.app{idx}" for idx in range(1, 25)]
    for idx, package_name in enumerate(packages):
        _write_annotation(annotations_dir / f"traj_{idx:02d}.json", package_name)

    layout_config = mug.launcher_mock._load_layout_config(None, annotations_dir=annotations_dir)
    drawer_icons = layout_config["pages"]["app_drawer"]["icons"]

    assert len(drawer_icons) == len(packages)
    assert {icon["package_name"] for icon in drawer_icons} == set(packages)
    assert any(
        icon["package_name"] == "com.seatgeek.android" and icon["asset"] == "seatgeek"
        for icon in drawer_icons
    )

    launcher_bundle = mug.launcher_mock._render_launcher_bundle(layout_config)

    assert set(launcher_bundle["page_images"]) == {
        mug.HOME_PAGE_ID,
        mug.DRAWER_PAGE_ID,
        "page_2_app_drawer",
    }
    assert len(launcher_bundle["drawer_page_specs"]) == 2
    assert len(launcher_bundle["drawer_page_specs"][1]["icons"]) == 1
    assert launcher_bundle["drawer_page_specs"][1]["icons"][0]["slot"] == "drawer_slot_1"

    second_page_icon = launcher_bundle["drawer_page_specs"][1]["icons"][0]
    second_page_layout = launcher_bundle["ui_structure"]["pages"]["page_2_app_drawer"]["layout"]
    assert second_page_layout[second_page_icon["layout_key"]]["bbox"] == [132, 560, 288, 716]


def test_build_unified_graph_passes_annotations_dir_to_launcher_layout_loader(
    tmp_path: Path,
    monkeypatch,
):
    captured: dict[str, Path | None] = {}

    def fake_load_layout_config(config_path: Path | None, annotations_dir: Path | None = None) -> dict:
        captured["annotations_dir"] = annotations_dir
        return {
            "metadata": {},
            "slots": {},
            "pages": {
                "home": {"icons": [], "dock_icons": []},
                "app_drawer": {"icons": []},
            },
        }

    monkeypatch.setattr(mug.launcher_mock, "_load_layout_config", fake_load_layout_config)
    monkeypatch.setattr(mug, "_save_mock_pages", lambda output_dir, layout_config: ({}, [], {"resolved_icons": []}))
    monkeypatch.setattr(mug, "_extract_drawer_apps", lambda drawer_page_specs, launcher_pages: [])
    monkeypatch.setattr(mug, "_scan_matching_annotations", lambda *args, **kwargs: {})
    monkeypatch.setattr(mug, "_write_match_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(mug, "_print_match_summary", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        max_trajectories_per_app=0,
        layout_config=str(tmp_path / "layout.json"),
        annotations_dir=str(tmp_path / "instruction_anno"),
        include_post_app_steps=False,
        scan_only=True,
    )

    mug.build_unified_graph(args)

    assert captured["annotations_dir"] == Path(args.annotations_dir)


def test_build_unified_graph_keeps_trying_matches_until_success_limit(
    tmp_path: Path,
    monkeypatch,
):
    app = mug.DrawerAppSpec(
        label="Example",
        asset="com.example.app",
        slug="example",
        layout_key="app_example",
        bbox=[12, 34, 56, 78],
        launcher_page_id=mug.DRAWER_PAGE_ID,
        match_tokens={"example"},
    )
    matches = [
        mug.MatchedTrajectory(
            app_slug="example",
            app_label="Example",
            annotation_path=str(tmp_path / f"traj_{idx}.json"),
            episode_id=f"episode_{idx}",
            instruction="Open Example",
            start_step_idx=0,
            end_step_idx=0,
            matched_package="com.example.app",
            total_steps=1,
        )
        for idx in range(1, 5)
    ]
    for match in matches:
        Path(match.annotation_path).write_text(
            json.dumps(
                {
                    "instruction": match.instruction,
                    "episode_id": match.episode_id,
                    "steps": [{"package_name": match.matched_package, "action": "TASK_COMPLETE"}],
                }
            ),
            encoding="utf-8",
        )

    compose_calls: list[str] = []

    def fake_compose_segment_pages(**kwargs):
        match = kwargs["match"]
        compose_calls.append(match.episode_id)
        if match.episode_id == "episode_1":
            return [], [], []
        return (
            [
                {
                    "page_id": f"page_{match.episode_id}",
                    "image": f"page_{match.episode_id}.png",
                    "depth": 2,
                    "layout": {},
                    "transitions": [],
                    "page_name": match.episode_id,
                    "application_id": "example",
                    "application_name": "Example",
                    "trajectory_ids": [match.episode_id],
                    "trajectory_ids_full": [match.episode_id],
                    "trace_steps": [1],
                    "page_summary": {},
                }
            ],
            [{"page_id": f"page_{match.episode_id}", "episode_id": match.episode_id}],
            [{"page_id": f"page_{match.episode_id}", "episode_id": match.episode_id}],
        )

    monkeypatch.setattr(
        mug.launcher_mock,
        "_load_layout_config",
        lambda config_path, annotations_dir=None: {"metadata": {}, "slots": {}, "pages": {}},
    )
    monkeypatch.setattr(mug, "_save_mock_pages", lambda output_dir, layout_config: ({}, [], {"resolved_icons": []}))
    monkeypatch.setattr(mug, "_extract_drawer_apps", lambda drawer_page_specs, launcher_pages: [app])
    monkeypatch.setattr(mug, "_scan_matching_annotations", lambda *args, **kwargs: {"example": matches})
    monkeypatch.setattr(mug, "_write_match_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(mug, "_print_match_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(mug, "_ensure_compose_modules", lambda: None)
    monkeypatch.setattr(
        mug,
        "action_compose",
        SimpleNamespace(
            load_api_client=lambda: None,
            load_detection_models=lambda *args, **kwargs: (None, None),
        ),
    )
    monkeypatch.setattr(mug, "_compose_segment_pages", fake_compose_segment_pages)
    monkeypatch.setattr(
        mug,
        "_merge_duplicate_content_pages",
        lambda output_dir, pages, asset_manifest, matched_step_rows, merge_candidate_page_ids: (
            pages,
            {},
            {
                "original_content_pages": len(pages),
                "merged_content_pages": len(pages),
                "merge_candidate_pages": len(merge_candidate_page_ids),
                "collapsed_duplicate_pages": 0,
                "collapsed_entry_pages": 0,
                "dropped_cross_app_or_invalid_transitions": 0,
            },
        ),
    )
    monkeypatch.setattr(mug, "_build_rich_launcher_pages", lambda launcher_pages, drawer_apps, app_entry_pages: {})
    monkeypatch.setattr(
        mug,
        "_merge_pages_by_deterministic_action_targets",
        lambda output_dir, pages, asset_manifest, matched_step_rows: (
            pages,
            {
                "original_pages": len(pages),
                "merged_pages": len(pages),
                "collapsed_duplicate_pages": 0,
                "duplicate_transitions_removed": 0,
            },
        ),
    )
    monkeypatch.setattr(
        mug,
        "_expand_topology_pages_by_trajectory_asset_clusters",
        lambda output_dir, pages, matched_rows, asset_rows: pages,
    )
    monkeypatch.setattr(
        mug,
        "_renumber_page_outputs",
        lambda output_dir, pages, asset_manifest, matched_step_rows, root_page_id: (pages, root_page_id, {}),
    )
    monkeypatch.setattr(mug, "_recompute_output_page_depths", lambda pages, root_page_id: pages)
    monkeypatch.setattr(mug, "_save_ui_structure", lambda *args, **kwargs: None)
    monkeypatch.setattr(mug, "_write_topology_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(mug, "_save_action_debug_overlays", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        max_trajectories_per_app=2,
        layout_config=str(tmp_path / "layout.json"),
        annotations_dir=str(tmp_path / "instruction_anno"),
        include_post_app_steps=False,
        scan_only=False,
        model_name="test-model",
        api_concurrency=1,
        weights_dir=str(tmp_path / "weights"),
        gpu=0,
        screenshots_dir=str(tmp_path / "screenshots"),
        element_anno_dir=str(tmp_path / "element_anno"),
    )

    mug.build_unified_graph(args)

    assert compose_calls == ["episode_1", "episode_2", "episode_3"]


def test_extract_drawer_apps_uses_explicit_layout_key():
    launcher_pages = {
        mug.DRAWER_PAGE_ID: {
            "layout": {"app_com_example_app1": [12, 34, 56, 78]},
        },
    }
    drawer_page_specs = [
        {
            "page_id": mug.DRAWER_PAGE_ID,
            "icons": [
                {
                    "label": "Example",
                    "asset": "com.example.app1",
                    "layout_key": "app_com_example_app1",
                }
            ],
        }
    ]

    apps = mug._extract_drawer_apps(drawer_page_specs, launcher_pages)

    assert len(apps) == 1
    assert apps[0].layout_key == "app_com_example_app1"
    assert apps[0].bbox == [12, 34, 56, 78]


def test_annotation_icon_spec_discovers_real_icon_path(tmp_path: Path, monkeypatch):
    element_root = tmp_path / "elements"
    target_dir = element_root / "traj_01"
    target_dir.mkdir(parents=True)
    icon_path = target_dir / "traj_01-2_spotify_elem0.png"
    icon_path.write_bytes(b"png")

    monkeypatch.setattr(mug.launcher_mock, "REAL_ICON_SEARCH_ROOTS", [element_root])
    monkeypatch.setattr(mug.launcher_mock, "REAL_ICON_LIBRARY", dict(mug.launcher_mock.REAL_ICON_LIBRARY))

    spec = mug.launcher_mock._annotation_app_icon_spec("com.spotify.music", annotation_id="traj_01")

    assert spec["asset"] == "com.spotify.music"
    assert mug.launcher_mock.REAL_ICON_LIBRARY[spec["asset"]]["path"] == icon_path


def test_package_match_tokens_do_not_match_nexuslauncher_generics():
    tokens = mug._build_match_tokens("Tasks", "com.google.android.apps.tasks")

    assert mug._matches_text("com.google.android.apps.tasks", tokens) is True
    assert mug._matches_text("com.google.android.apps.nexuslauncher", tokens) is False


def test_newsbreak_tokens_do_not_match_smartnews_package():
    tokens = mug._build_match_tokens("NewsBreak", "news_real")

    assert mug._matches_text("com.particlenews.newsbreak", tokens) is True
    assert mug._matches_text("jp.gocro.smartnews.android", tokens) is False


def test_generic_podcast_label_relies_on_package_specific_asset():
    tokens = mug._build_match_tokens("Podcast", "com.google.android.apps.podcasts")

    assert mug._matches_text("com.google.android.apps.podcasts", tokens) is True
    assert mug._matches_text("fm.castbox.audiobook.radio.podcast", tokens) is False
