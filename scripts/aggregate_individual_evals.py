"""Aggregate per-benchmark individual-eval JSONs into one table.

Writes:
  - eval_results/individual/summary.json        (structured)
  - results_individual.md                        (markdown leaderboard)
"""
import json, os, glob
from collections import OrderedDict

ROOT = "/workspace/gelab-env/eval_results/individual"
BENCHES = ["screenspot", "screenspot_v2", "motif", "refexp", "vwb_ag", "vwb_eg"]

def pct(c, t):
    return None if not t else c / t


summary = OrderedDict()
for run_dir in sorted(glob.glob(ROOT + "/*/")):
    run_name = os.path.basename(run_dir.rstrip("/"))
    entry = {"benchmarks": {}}
    any_found = False
    for b in BENCHES:
        p = os.path.join(run_dir, f"{b}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        # eval_screenspot.py stores {correct, total, wrong_format, accuracy}; others same.
        total = d.get("total") or 0
        correct = d.get("correct") or 0
        acc = d.get("accuracy") if d.get("accuracy") is not None else pct(correct, total)
        entry["benchmarks"][b] = {"correct": correct, "total": total, "accuracy": acc}
        any_found = True
    if any_found:
        # Compute avg over benches that ran
        accs = [v["accuracy"] for v in entry["benchmarks"].values() if v["accuracy"] is not None]
        entry["avg"] = sum(accs) / len(accs) if accs else None
        summary[run_name] = entry

# Write structured summary
with open(os.path.join(ROOT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Write markdown
out_md = "/workspace/gelab-env/results_individual.md"
HEADER_COLS = ["Run", "SS", "SS-v2", "MoTIF", "VWB-AG", "VWB-EG", "**Avg**", "n_benches"]
HEADER_BENCHES = ["screenspot", "screenspot_v2", "motif", "vwb_ag", "vwb_eg"]
lines = []
lines.append("# Individual-benchmark re-evaluation (computer_use guided generation)\n")
lines.append("Base model eval mode: Qwen2.5-VL native `computer_use` tool-call prompt + guided decoding. "
             "max_pixels=unlimited (99,999,999). Benchmarks loaded directly via HF (rootsautomation/ScreenSpot, "
             "HongxinLi/ScreenSpot_v2, HongxinLi/MOTIF-EVAL, HongxinLi/VWB-AG, HongxinLi/VWB-EG).\n")
lines.append("")
lines.append("| " + " | ".join(HEADER_COLS) + " |")
lines.append("|" + "|".join(["---"] * len(HEADER_COLS)) + "|")

# Sort by avg desc
for run, ent in sorted(summary.items(), key=lambda kv: -(kv[1].get("avg") or 0)):
    cells = []
    for b in HEADER_BENCHES:
        v = ent["benchmarks"].get(b)
        cells.append(f"{v['accuracy']*100:.2f}" if v and v["accuracy"] is not None else "—")
    avg = ent.get("avg")
    cells.append(f"**{avg*100:.2f}**" if avg is not None else "—")
    cells.append(str(len([b for b in HEADER_BENCHES if b in ent["benchmarks"]])))
    lines.append(f"| `{run}` | " + " | ".join(cells) + " |")

with open(out_md, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"wrote {os.path.join(ROOT, 'summary.json')}")
print(f"wrote {out_md}")
print(f"aggregated {len(summary)} runs")
