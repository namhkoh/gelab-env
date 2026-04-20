"""Upload a local Qwen2.5-VL-7B SFT checkpoint to a HuggingFace repo.

Usage:
    python scripts/hf_upload_checkpoint.py \
        --checkpoint checkpoint/gui_exp/sft_amex_t1a_aug_full/v0-.../checkpoint-4984 \
        --repo namhokaist/qwen25vl-7b-augexp-t1a_aug_full \
        --private  # optional
"""
import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", required=True, help="namespace/name")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--commit-message", default=None)
    args = ap.parse_args()

    if not args.token:
        print("ERROR: HF_TOKEN not provided"); return 2
    if not os.path.isdir(args.checkpoint):
        print(f"ERROR: checkpoint dir not found: {args.checkpoint}"); return 2

    api = HfApi(token=args.token)
    # Create the repo if it doesn't exist (idempotent)
    try:
        create_repo(args.repo, token=args.token, private=args.private, exist_ok=True)
        print(f"[hf] repo ready: {args.repo}")
    except Exception as e:
        print(f"WARN: create_repo failed: {e}")

    commit = args.commit_message or f"Upload SFT checkpoint from {os.path.basename(args.checkpoint)}"

    # Filter: upload model/tokenizer/processor weights+config. Skip the bulky optimizer state
    # and rng state files — user specified save_only_model=true so those shouldn't exist anyway.
    ignore = [
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "rng_state*.pth",
        "global_step*",  # deepspeed
        "zero_to_fp32.py",
        "latest",
    ]
    print(f"[hf] uploading {args.checkpoint} -> {args.repo}")
    api.upload_folder(
        folder_path=args.checkpoint,
        repo_id=args.repo,
        repo_type="model",
        commit_message=commit,
        ignore_patterns=ignore,
    )
    print(f"[hf] done: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
