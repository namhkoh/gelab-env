"""Resume HF dataset snapshot with fewer workers and 429 backoff."""
import os
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

if not os.environ.get("HF_TOKEN"):
    raise SystemExit("HF_TOKEN env var required")

LOCAL_DIR = "/workspace/gelab-env/datas_amex/amex-augmented-sft"
REPO = "namhokaist/amex-augmented-sft"

MAX_ATTEMPTS = 40
WORKERS = 4
BACKOFF = 60


def current_count() -> int:
    imgs = os.path.join(LOCAL_DIR, "images")
    return len(os.listdir(imgs)) if os.path.isdir(imgs) else 0


for attempt in range(1, MAX_ATTEMPTS + 1):
    start = current_count()
    print(f"[attempt {attempt}] starting (images so far: {start}/10000)", flush=True)
    try:
        path = snapshot_download(
            repo_id=REPO,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            max_workers=WORKERS,
        )
        done = current_count()
        print(f"SNAPSHOT_DONE {path} ({done} images)", flush=True)
        break
    except HfHubHTTPError as e:
        msg = str(e)
        print(f"[attempt {attempt}] HTTP error: {msg[:200]}", flush=True)
        if "429" in msg or "Too Many Requests" in msg:
            print(f"[attempt {attempt}] sleeping {BACKOFF}s for rate limit", flush=True)
            time.sleep(BACKOFF)
        else:
            time.sleep(30)
    except Exception as e:
        print(f"[attempt {attempt}] error: {e}", flush=True)
        time.sleep(30)
else:
    raise SystemExit(f"Failed after {MAX_ATTEMPTS} attempts")
