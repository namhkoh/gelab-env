from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _load_dataset_info(local_dir: Path) -> dict:
    dataset_info_path = local_dir / "dataset_info.json"
    if not dataset_info_path.exists():
        raise SystemExit(f"dataset_info.json not found in {local_dir}")
    return json.loads(dataset_info_path.read_text(encoding="utf-8"))


def _count_manifest_rows(local_dir: Path) -> int:
    manifest_path = local_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.jsonl not found in {local_dir}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _summarize_prepared_dataset(local_dir: Path) -> dict:
    dataset_info = _load_dataset_info(local_dir)
    shards_dir = local_dir / "shards"
    if not shards_dir.exists():
        raise SystemExit(f"shards directory not found in {local_dir}")

    shard_count = len(list(shards_dir.glob("*.tar")))
    manifest_rows = _count_manifest_rows(local_dir)
    expected_shards = int(dataset_info.get("shard_count", -1))
    expected_trajectories = int(dataset_info.get("trajectory_count", -1))

    if expected_shards != shard_count:
        raise SystemExit(
            "Prepared dataset looks incomplete: "
            f"dataset_info.json expects {expected_shards} shard(s), found {shard_count}"
        )
    if expected_trajectories != manifest_rows:
        raise SystemExit(
            "Prepared dataset looks incomplete: "
            f"dataset_info.json expects {expected_trajectories} manifest row(s), found {manifest_rows}"
        )

    return {
        "dataset_info": dataset_info,
        "shard_count": shard_count,
        "manifest_rows": manifest_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a prepared AMEX SFT folder to a Hugging Face dataset repository."
    )
    parser.add_argument(
        "--local_dir",
        type=Path,
        required=True,
        help="Prepared local dataset directory produced by prepare_amex_sft_hf_dataset.py",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target dataset repo id, e.g. username/amex-sft",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="HF token. If omitted, HF_TOKEN or HUGGINGFACE_HUB_TOKEN will be used.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate the prepared dataset folder and print a summary without uploading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = args.local_dir.resolve()
    if not local_dir.exists():
        raise SystemExit(f"Local directory does not exist: {local_dir}")

    summary = _summarize_prepared_dataset(local_dir)
    dataset_info = summary["dataset_info"]
    source_roots = dataset_info.get("source_roots", [])
    print(f"Validated prepared dataset: {local_dir}")
    print(f"Source roots      : {len(source_roots)}")
    for source_root in source_roots:
        print(
            "  - "
            f"{source_root['name']}: {source_root['trajectory_count']} trajectories"
        )
    print(f"Trajectories      : {dataset_info['trajectory_count']}")
    print(f"Manifest rows     : {summary['manifest_rows']}")
    print(f"Tar shards        : {summary['shard_count']}")
    print(f"Prepared size     : {dataset_info['bytes'] / (1024 ** 3):.2f} GiB")

    if args.dry_run:
        print("Dry run complete. Skipping Hugging Face upload.")
        return

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF token not found. Pass --token or set HF_TOKEN.")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run: pip install 'huggingface_hub>=0.30.0'"
        ) from exc

    api = HfApi(token=token)
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=bool(args.private),
        exist_ok=True,
    )
    print(f"Dataset repo ready: {repo_url}")
    print(f"Uploading folder with upload_large_folder(): {local_dir}")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
    )
    print(f"Upload complete: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
