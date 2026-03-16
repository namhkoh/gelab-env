# Data paths on the shared server (findings)

The code defaults assume `/ext_hdd2/nhkoh/...`, but **on this server there is no `/ext_hdd2` — only `/ext_hdd` is mounted.**

## Paths that actually exist (when sharing this server)

| Purpose | Code / docs default | Verified path on this server | Notes |
|--------|---------------------|------------------------------|-------|
| nhkoh home (shared disk) | `/ext_hdd2/nhkoh` | **`/ext_hdd/nhkoh`** | ✅ Present |
| gelab project | `/ext_hdd2/nhkoh/gelab-env` | **`/ext_hdd/nhkoh/gelab-engine`** | Repo name is `gelab-engine` (may differ) |
| GUIOdyssey data | `/ext_hdd2/nhkoh/GUI-Odyssey/` | **`/ext_hdd/nhkoh/dataset/GUIOdyssey/`** | ✅ Present |
| GUIOdyssey screenshots | `.../screenshots` | **`/ext_hdd/nhkoh/dataset/GUIOdyssey/screenshots/`** | Only split archives (`screenshots.z05`~`.z08`) currently; PNGs must be extracted |
| OmniParser weights | `/ext_hdd2/nhkoh/OmniParser/weights` | (not yet confirmed) | Likely somewhere under `/ext_hdd/nhkoh` |

## Summary of what was checked

- **`/ext_hdd/nhkoh`** — exists and is reachable when sharing this server.
- **`/ext_hdd/nhkoh/dataset/GUIOdyssey`** — exists. Under `screenshots/`, only `screenshots.z05` ~ `screenshots.z08` (zip split) were found; **PNGs must be extracted before use**.
- **`/ext_hdd/nhkoh/gelab-engine/datas/images`** — contains a small number of page images such as `page_0.png`, `page_1.png` (similar to this repo’s `datas/images`).
- **Annotations** — an `annotations` folder was not immediately found under GUIOdyssey during this scan; it may live somewhere under `dataset/GUIOdyssey` or in a Hugging Face cache.
- **OmniParser weights** — exact location under `/ext_hdd/nhkoh` is still unknown.

## Using these paths for trajectory mode on this server

Instead of relying on the hard‑coded defaults, you can pass explicit paths like:

```bash
# Example (after extracting archives)
--screenshots_dir /ext_hdd/nhkoh/dataset/GUIOdyssey/screenshots
--annotations_dir /ext_hdd/nhkoh/dataset/GUIOdyssey/annotations   # verify existence first
--omniparser_weights /ext_hdd/nhkoh/OmniParser/weights            # update to the actual OmniParser location
```

- You may need to extract the GUIOdyssey screenshots from `screenshots.z*` before they can be used.
- It is a good idea to double‑check the `annotations` directory with `ls /ext_hdd/nhkoh/dataset/GUIOdyssey`.
