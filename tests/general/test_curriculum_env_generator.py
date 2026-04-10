from pathlib import Path
import json
import sys

from PIL import Image, ImageDraw


sys.path.insert(0, "/home/tsyou/gelab-env/data_engine")

import curriculum_env_generator as ceg


def _make_chain_env(env_dir: Path, total_pages: int = 4) -> None:
    pages_dir = env_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages = {}
    for idx in range(total_pages):
        page_id = f"page_{idx}"
        image = Image.new("RGB", (1080, 2400), (248, 248, 248))
        draw = ImageDraw.Draw(image)

        target_bbox = [120, 340, 520, 640]
        distractor_a = [580, 340, 960, 640]
        distractor_b = [120, 760, 520, 1060]

        draw.rounded_rectangle(target_bbox, radius=30, fill=(120, 190, 255))
        draw.rounded_rectangle(distractor_a, radius=30, fill=(255, 205, 140))
        draw.rounded_rectangle(distractor_b, radius=30, fill=(160, 225, 170))
        draw.rounded_rectangle([10, 8, 138, 60], radius=14, fill=(255, 220, 220))
        draw.rounded_rectangle([942, 8, 1070, 60], radius=14, fill=(220, 248, 220))
        image.save(pages_dir / f"{page_id}.png")

        transitions = []
        if idx < total_pages - 1:
            transitions.append(
                {
                    "action": f"target_{idx}",
                    "target_page": f"page_{idx + 1}",
                    "icon_bbox": target_bbox,
                    "action_coord": [(target_bbox[0] + target_bbox[2]) // 2, (target_bbox[1] + target_bbox[3]) // 2],
                }
            )
        transitions.append(
            {
                "action": "back",
                "target_page": f"page_{max(0, idx - 1)}",
                "icon_bbox": [10, 8, 138, 60],
                "action_coord": [74, 34],
            }
        )
        transitions.append(
            {
                "action": "home",
                "target_page": "page_0",
                "icon_bbox": [942, 8, 1070, 60],
                "action_coord": [1006, 34],
            }
        )

        pages[page_id] = {
            "image": f"{page_id}.png",
            "depth": idx,
            "layout": {
                f"target_{idx}": {"bbox": target_bbox, "type": "normal"},
                f"distractor_{idx}_a": {"bbox": distractor_a, "type": "normal"},
                f"distractor_{idx}_b": {"bbox": distractor_b, "type": "normal"},
                "back": {"bbox": [10, 8, 138, 60], "type": "system"},
                "home": {"bbox": [942, 8, 1070, 60], "type": "system"},
            },
            "transitions": transitions,
        }

    layer = {
        "root": {
            "image": "page_0.png",
            "depth": 0,
            "layout": pages["page_0"]["layout"],
            "transitions": [pages["page_0"]["transitions"][0]],
            "subnodes": [],
        }
    }
    current = layer["root"]
    for idx in range(1, total_pages):
        page_id = f"page_{idx}"
        node = {
            "image": f"{page_id}.png",
            "depth": idx,
            "layout": pages[page_id]["layout"],
            "transitions": [t for t in pages[page_id]["transitions"] if t["action"] not in {"back", "home"}],
            "subnodes": [],
        }
        current["subnodes"].append(node)
        current = node

    ui = {
        "pages": pages,
        "metadata": {
            "source": "synthetic_test_env",
            "episode_id": env_dir.name,
            "instruction": "Open the target item and keep navigating forward.",
            "canvas_size": [1080, 2400],
            "phone_canvas_size": [1080, 2328],
            "nav_strip_height": 72,
        },
    }

    with (env_dir / "ui_structure.json").open("w", encoding="utf-8") as f:
        json.dump(ui, f, indent=2)
    with (env_dir / "ui_structure_layer.json").open("w", encoding="utf-8") as f:
        json.dump(layer, f, indent=2)


def test_generate_curriculum_keeps_original_trajectory_length(tmp_path: Path):
    env_dir = tmp_path / "env_a"
    output_root = tmp_path / "out"
    _make_chain_env(env_dir, total_pages=4)

    spec = ceg.LevelSpec(
        level=1,
        trajectory_length_ratio=0.5,
        ui_icon_number=3,
        action_space=1,
        target_scale=1.2,
        layout_variation=0.0,
        style_variation=0.0,
        popup_probability=0.0,
    )
    manifest = ceg.generate_curriculum_for_env(env_dir, output_root, [spec], seed=7)

    level_ui_path = output_root / "env_a" / "level_01" / "ui_structure.json"
    with level_ui_path.open("r", encoding="utf-8") as f:
        generated = json.load(f)

    assert manifest["levels"][0]["main_pages"] == 4
    assert generated["metadata"]["trajectory_length"] == 4
    assert generated["metadata"]["trajectory_length_policy"] == "kept_original"
    assert generated["metadata"]["main_page_ids"] == ["page_0", "page_1", "page_2", "page_3"]
    assert set(generated["pages"].keys()) == {"page_0", "page_1", "page_2", "page_3"}
    assert (output_root / "env_a" / "level_01" / "action_coord" / "page_0.png").exists()


def test_generate_curriculum_adds_popup_contamination(tmp_path: Path):
    env_dir = tmp_path / "env_b"
    output_root = tmp_path / "out"
    _make_chain_env(env_dir, total_pages=3)

    spec = ceg.LevelSpec(
        level=4,
        trajectory_length_ratio=1.0,
        ui_icon_number=6,
        action_space=3,
        target_scale=0.9,
        layout_variation=0.8,
        style_variation=1.0,
        popup_probability=1.0,
    )
    ceg.generate_curriculum_for_env(env_dir, output_root, [spec], seed=11)

    level_ui_path = output_root / "env_b" / "level_04" / "ui_structure.json"
    with level_ui_path.open("r", encoding="utf-8") as f:
        generated = json.load(f)

    popup_pages = [page_id for page_id in generated["pages"] if "_popup_" in page_id]
    page_0_targets = {
        trans["target_page"]
        for trans in generated["pages"]["page_0"]["transitions"]
    }

    assert popup_pages
    assert any(target in popup_pages for target in page_0_targets)
    assert generated["metadata"]["popup_pages"] >= 1
    assert (output_root / "env_b" / "level_04" / "action_coord" / "page_0.png").exists()


def test_variation_pages_do_not_visually_mark_ground_truth_slot(tmp_path: Path):
    env_dir = tmp_path / "env_clean"
    output_root = tmp_path / "out"
    _make_chain_env(env_dir, total_pages=3)

    spec = ceg.LevelSpec(
        level=2,
        trajectory_length_ratio=0.7,
        ui_icon_number=3,
        action_space=2,
        target_scale=1.0,
        layout_variation=0.0,
        style_variation=0.0,
        popup_probability=0.0,
    )
    ceg.generate_curriculum_for_env(env_dir, output_root, [spec], seed=13)

    level_dir = output_root / "env_clean" / "level_02"
    with (level_dir / "ui_structure.json").open("r", encoding="utf-8") as f:
        generated = json.load(f)

    page_info = generated["pages"]["page_0"]
    gt_action = next(
        trans["action"] for trans in page_info["transitions"] if trans.get("transition_role") == "ground_truth"
    )
    distractor_action = next(
        label for label in page_info["layout"] if label not in {gt_action, "back", "home"}
    )
    gt_bbox = page_info["layout"][gt_action]["bbox"]
    distractor_bbox = page_info["layout"][distractor_action]["bbox"]

    with Image.open(env_dir / "pages" / "page_0.png") as source_image:
        source_gt_pixel = source_image.getpixel((gt_bbox[0], (gt_bbox[1] + gt_bbox[3]) // 2))
        source_distractor_pixel = source_image.getpixel((distractor_bbox[0], (distractor_bbox[1] + distractor_bbox[3]) // 2))
    with Image.open(level_dir / "pages" / "page_0.png") as image:
        gt_border_pixel = image.getpixel((gt_bbox[0], (gt_bbox[1] + gt_bbox[3]) // 2))
        distractor_border_pixel = image.getpixel((distractor_bbox[0], (distractor_bbox[1] + distractor_bbox[3]) // 2))

    assert gt_border_pixel == source_gt_pixel
    assert distractor_border_pixel == source_distractor_pixel


def test_variation_pages_keep_original_page_background(tmp_path: Path):
    env_dir = tmp_path / "env_base"
    output_root = tmp_path / "out"
    _make_chain_env(env_dir, total_pages=3)

    spec = ceg.LevelSpec(
        level=2,
        trajectory_length_ratio=0.7,
        ui_icon_number=2,
        action_space=1,
        target_scale=1.0,
        layout_variation=0.0,
        style_variation=0.0,
        popup_probability=0.0,
    )
    ceg.generate_curriculum_for_env(env_dir, output_root, [spec], seed=17)

    with Image.open(env_dir / "pages" / "page_0.png") as source_image:
        source_pixel = source_image.getpixel((40, 1800))
    with Image.open(output_root / "env_base" / "level_02" / "pages" / "page_0.png") as generated_image:
        generated_pixel = generated_image.getpixel((40, 1800))

    assert generated_pixel == source_pixel


def test_generate_curriculum_suite_keeps_envs_independent(tmp_path: Path):
    input_root = tmp_path / "suite"
    output_root = tmp_path / "out"
    env_a = input_root / "env_a"
    env_b = input_root / "env_b"
    _make_chain_env(env_a, total_pages=2)
    _make_chain_env(env_b, total_pages=3)

    specs = ceg.default_level_specs(1)
    manifest = ceg.generate_curriculum_suite(input_root, output_root, specs, seed=19)

    assert len(manifest["environments"]) == 2
    assert (output_root / "env_a" / "level_01" / "ui_structure.json").exists()
    assert (output_root / "env_b" / "level_01" / "ui_structure.json").exists()
    assert manifest["environments"][0]["source_env_id"] != manifest["environments"][1]["source_env_id"]


def test_select_level_specs_returns_only_requested_level():
    specs = ceg.default_level_specs(3)

    selected = ceg.select_level_specs(specs, level=2)

    assert [spec.level for spec in selected] == [2]


def test_default_five_level_suite_anchors_original_base_in_middle(tmp_path: Path):
    env_dir = tmp_path / "env_c"
    output_root = tmp_path / "out"
    _make_chain_env(env_dir, total_pages=3)

    specs = ceg.default_level_specs(5)
    manifest = ceg.generate_curriculum_for_env(env_dir, output_root, specs, seed=5)

    level_three_ui = output_root / "env_c" / "level_03" / "ui_structure.json"
    with level_three_ui.open("r", encoding="utf-8") as f:
        generated = json.load(f)
    with (env_dir / "ui_structure.json").open("r", encoding="utf-8") as f:
        source = json.load(f)

    level_modes = {entry["level"]: entry["level_mode"] for entry in manifest["levels"]}
    assert level_modes[3] == "original_base"
    assert level_modes[1] == "variation"
    assert level_modes[5] == "variation"
    assert generated["metadata"]["level_mode"] == "original_base"
    assert generated["metadata"]["level"] == 3
    assert generated["pages"] == source["pages"]
    assert (output_root / "env_c" / "level_03" / "pages" / "page_0.png").exists()


def test_default_level_specs_rejects_unknown_base_level():
    try:
        ceg.default_level_specs(5, base_level=8)
    except ValueError as exc:
        assert "Base level 8 is not defined" in str(exc)
    else:
        raise AssertionError("Expected invalid base_level to raise ValueError")
