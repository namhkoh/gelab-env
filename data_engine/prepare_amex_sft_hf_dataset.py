from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class TrajectoryEntry:
    source_root: Path
    trajectory_dir: Path

    @property
    def source_name(self) -> str:
        return self.source_root.name

    @property
    def trajectory_id(self) -> str:
        return self.trajectory_dir.name

    @property
    def archive_prefix(self) -> Path:
        return Path(self.source_name) / self.trajectory_id


def _iter_source_roots(source_dir: Path, dataset_glob: str) -> List[Path]:
    candidate_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    matched_children = [path for path in candidate_dirs if fnmatch(path.name, dataset_glob)]
    if matched_children:
        return matched_children
    return [source_dir]


def _iter_trajectory_dirs(source_roots: List[Path]) -> List[TrajectoryEntry]:
    trajectory_entries: List[TrajectoryEntry] = []
    for source_root in sorted(source_roots, key=lambda path: path.name):
        trajectory_entries.extend(
            TrajectoryEntry(source_root=source_root, trajectory_dir=path)
            for path in sorted(path for path in source_root.iterdir() if path.is_dir())
        )
    return trajectory_entries


def _batched(items: List[TrajectoryEntry], batch_size: int) -> Iterable[List[TrajectoryEntry]]:
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _write_dataset_card(
    output_dir: Path,
    source_roots: List[Path],
    source_counts: dict[str, int],
    shard_count: int,
    trajectory_count: int,
    trajectories_per_shard: int,
) -> None:
    source_root_lines = "\n".join(
        f"- `{source_root}` ({source_counts[source_root.name]} trajectories)"
        for source_root in source_roots
    )
    readme = f"""---
pretty_name: AMEX SFT
task_categories:
- image-text-to-text
- text-generation
tags:
- gui
- mobile
- navigation
- multimodal
- ge-lab
size_categories:
- 10K<n<100K
---

# AMEX SFT

This dataset is a packaged export of the local `amex_sft` directory for uploading to the Hugging Face Hub as a dataset repository.

## Source

- Source dataset roots:
{source_root_lines}
- Number of trajectory folders: `{trajectory_count}`
- Number of tar shards: `{shard_count}`
- Trajectories per shard: `{trajectories_per_shard}`

## Layout

- `shards/*.tar`: tar shards containing trajectory folders
- `manifest.jsonl`: trajectory-to-shard index
- `dataset_info.json`: high-level metadata

Each tar shard preserves the original trajectory folder layout. For example:

```text
<source_root>/<trajectory_id>/
  ui_structure.json
  ui_structure_layer.json
  trajectory_assets_manifest.json
  action_coord/...
  extracted_assets/...
```

## Why tar shards?

The raw source contains a very large number of small PNG files. Packaging them into tar shards makes Hub upload and downstream download much more reliable for large-scale storage.

## Notes

- This repository is intended for storage and reuse of the packaged dataset.
- If you want Hub-native previews and lighter access patterns, consider a future Parquet/WebDataset conversion.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def _remove_existing_shards(shards_dir: Path) -> int:
    removed = 0
    for shard_path in sorted(shards_dir.glob("amex_sft_shard_*.tar")):
        shard_path.unlink()
        removed += 1
    return removed


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    dataset_glob: str,
    trajectories_per_shard: int,
) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    source_roots = _iter_source_roots(source_dir, dataset_glob)
    if not source_roots:
        raise SystemExit(
            f"No source dataset roots found in {source_dir} matching {dataset_glob!r}"
        )

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    removed_shards = _remove_existing_shards(shards_dir)
    if removed_shards:
        print(f"Removed {removed_shards} existing shard(s) from {shards_dir}")

    trajectory_entries = _iter_trajectory_dirs(source_roots)
    if not trajectory_entries:
        raise SystemExit(f"No trajectory folders found under {source_dir}")

    source_counts = {}
    for source_root in source_roots:
        source_counts[source_root.name] = sum(
            1 for path in source_root.iterdir() if path.is_dir()
        )

    print(f"Discovered {len(source_roots)} source dataset root(s):")
    for source_root in source_roots:
        print(f"  - {source_root.name}: {source_counts[source_root.name]} trajectories")

    manifest_rows = []
    total_bytes = 0
    total_files = 0

    for shard_index, batch in enumerate(_batched(trajectory_entries, trajectories_per_shard)):
        shard_name = f"amex_sft_shard_{shard_index:04d}.tar"
        shard_path = shards_dir / shard_name
        shard_bytes = 0
        shard_files = 0

        print(f"[{shard_index + 1}] writing {shard_name} with {len(batch)} trajectories")
        with tarfile.open(shard_path, "w") as tar_handle:
            for entry in batch:
                trajectory_bytes = 0
                trajectory_files = 0
                for file_path in _iter_files(entry.trajectory_dir):
                    arcname = entry.archive_prefix / file_path.relative_to(entry.trajectory_dir)
                    tar_handle.add(file_path, arcname=str(arcname))
                    file_size = file_path.stat().st_size
                    trajectory_bytes += file_size
                    trajectory_files += 1

                manifest_rows.append({
                    "source_root": entry.source_name,
                    "source_dir": str(entry.source_root),
                    "trajectory_id": entry.trajectory_id,
                    "shard": f"shards/{shard_name}",
                    "member_prefix": f"{entry.archive_prefix.as_posix()}/",
                    "file_count": trajectory_files,
                    "bytes": trajectory_bytes,
                })
                shard_bytes += trajectory_bytes
                shard_files += trajectory_files

        total_bytes += shard_bytes
        total_files += shard_files
        print(
            f"    finished {shard_name}: {len(batch)} trajectories, "
            f"{shard_files} files, {shard_bytes / (1024 ** 3):.2f} GiB"
        )

    dataset_info = {
        "dataset_name": "amex_sft",
        "source_dir": str(source_dir),
        "source_root_count": len(source_roots),
        "source_roots": [
            {
                "name": source_root.name,
                "path": str(source_root),
                "trajectory_count": source_counts[source_root.name],
            }
            for source_root in source_roots
        ],
        "trajectory_count": len(trajectory_entries),
        "shard_count": len(list(shards_dir.glob("*.tar"))),
        "trajectories_per_shard": trajectories_per_shard,
        "file_count": total_files,
        "bytes": total_bytes,
    }

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2),
        encoding="utf-8",
    )
    _write_dataset_card(
        output_dir=output_dir,
        source_roots=source_roots,
        source_counts=source_counts,
        shard_count=dataset_info["shard_count"],
        trajectory_count=dataset_info["trajectory_count"],
        trajectories_per_shard=trajectories_per_shard,
    )

    print()
    print(f"Prepared dataset at: {output_dir}")
    print(f"Shards           : {dataset_info['shard_count']}")
    print(f"Trajectories     : {dataset_info['trajectory_count']}")
    print(f"Files packaged   : {dataset_info['file_count']}")
    print(f"Total size       : {dataset_info['bytes'] / (1024 ** 3):.2f} GiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package amex_sft into tar shards suitable for a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("/ext_hdd2/tsyou/gelab-env/data_engine"),
        help=(
            "Either a single AMEX source root (for example .../amex_sft) or a parent "
            "directory that contains multiple AMEX roots"
        ),
    )
    parser.add_argument(
        "--dataset_glob",
        type=str,
        default="amex_sft*",
        help="When --source_dir points at a parent directory, include child directories matching this glob",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the packaged HF-ready dataset will be written",
    )
    parser.add_argument(
        "--trajectories_per_shard",
        type=int,
        default=50,
        help="How many trajectory folders to store in each tar shard",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        dataset_glob=args.dataset_glob,
        trajectories_per_shard=max(1, int(args.trajectories_per_shard)),
    )


if __name__ == "__main__":
    main()
