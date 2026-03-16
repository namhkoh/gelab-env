# Running trajectory mode with real GUIOdyssey data

## What is already prepared

1. **Annotations**
   - Per‑episode JSON can be downloaded from Hugging Face via `scripts/run_trajectory_real_guiodyssey.py` (or with `--no_run` to prepare only the annotations).
   - Already prepared locally:
     - `datas/guiodyssey_2493102722960871/annotations/2493102722960871.json` (2‑step episode)
     - `datas/guiodyssey_7872483543119388/annotations/7872483543119388.json` (18‑step episode)

2. **Screenshots**
   - On HF, screenshots are provided as **split zip archives** (`screenshots.zip` + `.z01`~`.z08`, about 90GB total), so downloading only a small subset is not straightforward.
   - Options:
     - **A.** On the shared server, unzip everything under ` /ext_hdd/nhkoh/dataset/GUIOdyssey/screenshots/` and run with that path.
     - **B.** Use `--use_placeholder_screenshots`: build layouts from real annotations (`sam2_bbox`) but use placeholder images + family cache so the pipeline can be validated **without OmniParser**.

## How to run

### 1) Placeholder images + real annotations (no OmniParser)

```bash
cd /home/sychoe/gelab-env
python3 scripts/run_trajectory_real_guiodyssey.py \
  --episode_id 2493102722960871 \
  --use_placeholder_screenshots
```

- For the 2‑step episode `2493102722960871`, this creates placeholder screenshots and family cache, then runs `tree.py`.
- Output: `datas/guiodyssey_real_2493102722960871/out/ui_structure.json`, `effective_spine_page_ids`, etc.

### 2) One‑shot run for the same 2‑step episode

```bash
cd /home/sychoe/gelab-env
python3 scripts/run_real_one_shot.py
```

- Assumes `datas/guiodyssey_2493102722960871/annotations/` already exists. It builds placeholders + cache and runs `tree.py` in a single step.

### 3) When real screenshots are available (e.g., on the server)

If PNG screenshots already exist in some directory, just point the script there:

```bash
# Example: after extracting zip files on the server
export GUIODYSSEY_SCREENSHOTS_DIR=/ext_hdd/nhkoh/dataset/GUIOdyssey/screenshots
python3 scripts/run_trajectory_real_guiodyssey.py --episode_id 2493102722960871
```

- In this case detection must run, so you need a valid **OmniParser weights** path (via `--omniparser_weights` or the default `/ext_hdd/nhkoh/OmniParser/weights`).

## Summary

- To **experiment with real GUIOdyssey data**:
  - Annotations are already usable from HF.
  - For images, you can either:
    - Extract the HF screenshot zips on the server, or
    - Use `--use_placeholder_screenshots` / `run_real_one_shot.py` to first validate the pipeline with “real annotations + placeholder images”.
- To use **actual pixels**, prepare a screenshots directory and configure the OmniParser weights path.

---

## Data paths on the shared server

Code defaults use `/ext_hdd2/nhkoh/...`; on the shared server only `/ext_hdd` is mounted.

| Purpose | Verified path |
|--------|----------------|
| nhkoh home | `/ext_hdd/nhkoh` |
| GUIOdyssey data | `/ext_hdd/nhkoh/dataset/GUIOdyssey/` |
| Screenshots | `/ext_hdd/nhkoh/dataset/GUIOdyssey/screenshots/` (split archives; extract for PNGs) |
| OmniParser weights | Under `/ext_hdd/nhkoh` (confirm path) |

Example for trajectory mode:

```bash
--screenshots_dir /ext_hdd/nhkoh/dataset/GUIOdyssey/screenshots
--annotations_dir /ext_hdd/nhkoh/dataset/GUIOdyssey/annotations
--omniparser_weights /ext_hdd/nhkoh/OmniParser/weights
```
