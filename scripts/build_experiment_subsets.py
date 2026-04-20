"""Path B subset builder — 21,337-sample stratified subsets for each experiment.

Goal: each subset matches ContinueTrain-v2's 21k sample budget, so every run
trains with identical compute at identical recipe. Only the data source varies.

Outputs written to /workspace/gelab-env/datas_amex/ :
  pb_t1a_aug_21k.json        — aug, stratified 25/37.5/37.5 (nav/G/U)
  pb_t1b_orig_21k.json       — orig, same ratio (nav subsampled from orig's 24.8k pool)
  pb_t1c_mix_21k.json        — 50-50 aug+orig per task bucket, same ratio
  pb_t2a_aug_nav_21k.json    — aug-nav only
  pb_t2b_orig_nav_21k.json   — orig-nav only (capped at pool size)
"""
from __future__ import annotations
import json
import os
import random
from collections import Counter, defaultdict

random.seed(42)

A_PATH = "/workspace/gelab-env/datas_amex/amex-augmented-sft/train.json"
O_PATH = "/workspace/gelab-env/datas/amex_gelab_sft_full.json"
OUT_ROOT = "/workspace/gelab-env/datas_amex"
os.makedirs(OUT_ROOT, exist_ok=True)

TOTAL = 21337        # match ContinueTrain-v2 exactly
NAV_P = 0.250        # 25 % nav  (aug's natural ratio)
G_P   = 0.375        # 37.5 % grounding
U_P   = 0.375        # 37.5 % understanding

N_NAV = round(TOTAL * NAV_P)
N_G   = round(TOTAL * G_P)
N_U   = TOTAL - N_NAV - N_G
print(f"Target ratio: nav={N_NAV}, g={N_G}, u={N_U} (total={N_NAV+N_G+N_U})")


def task_of(s):
    src = s.get("source", "")
    if "nav" in src: return "nav"
    if "grounding" in src: return "g"
    if "understanding" in src: return "u"
    return None


def bucketize(data):
    b = defaultdict(list)
    for s in data:
        t = task_of(s)
        if t: b[t].append(s)
    return b


def take(pool, n, name):
    if n <= len(pool):
        return random.sample(pool, n)
    print(f"  [oversample] {name}: {n} from pool of {len(pool)} (~{n/len(pool):.2f}x)")
    return [random.choice(pool) for _ in range(n)]


def write(path, data):
    with open(path, "w") as f: json.dump(data, f)
    c = Counter(task_of(s) for s in data)
    tot = len(data)
    dist = ", ".join(f"{k}={v} ({100*v/tot:.1f}%)" for k, v in c.most_common())
    print(f"  wrote {os.path.basename(path)}: {tot:,} [{dist}]")


print("Loading datasets...")
A = json.load(open(A_PATH))
O = json.load(open(O_PATH))
print(f"  aug: {len(A):,}  orig: {len(O):,}")
A_by = bucketize(A)
O_by = bucketize(O)
print(f"  aug: nav={len(A_by['nav'])}, g={len(A_by['g'])}, u={len(A_by['u'])}")
print(f"  orig: nav={len(O_by['nav'])}, g={len(O_by['g'])}, u={len(O_by['u'])}")

# ---- T1.A aug 21k (stratified 25/37.5/37.5)
print("\n[T1.A] aug 21k")
t1a = (take(A_by["nav"], N_NAV, "t1a/nav")
       + take(A_by["g"],   N_G,   "t1a/g")
       + take(A_by["u"],   N_U,   "t1a/u"))
random.shuffle(t1a); write(f"{OUT_ROOT}/pb_t1a_aug_21k.json", t1a)

# ---- T1.B orig 21k (same ratio, orig has 24.8k nav so no oversample needed)
print("\n[T1.B] orig 21k")
t1b = (take(O_by["nav"], N_NAV, "t1b/nav")
       + take(O_by["g"],   N_G,   "t1b/g")
       + take(O_by["u"],   N_U,   "t1b/u"))
random.shuffle(t1b); write(f"{OUT_ROOT}/pb_t1b_orig_21k.json", t1b)

# ---- T1.C mix 50-50 per task bucket
print("\n[T1.C] mix 50-50 21k")
def half(n): return n // 2, n - n // 2
t1c = []
for task, n in (("nav", N_NAV), ("g", N_G), ("u", N_U)):
    fa, fo = half(n)
    t1c += take(A_by[task], fa, f"t1c/{task}/A")
    t1c += take(O_by[task], fo, f"t1c/{task}/O")
random.shuffle(t1c); write(f"{OUT_ROOT}/pb_t1c_mix_21k.json", t1c)

# ---- T2.A aug-nav 21k
print("\n[T2.A] aug-nav 21k")
t2a = take(A_by["nav"], 21000, "t2a")
random.shuffle(t2a); write(f"{OUT_ROOT}/pb_t2a_aug_nav_21k.json", t2a)

# ---- T2.B orig-nav (pool is 24.8k — take all of it, ~21k if we cap)
print("\n[T2.B] orig-nav 21k")
t2b = take(O_by["nav"], 21000, "t2b")
random.shuffle(t2b); write(f"{OUT_ROOT}/pb_t2b_orig_nav_21k.json", t2b)

print("\nDone.")
