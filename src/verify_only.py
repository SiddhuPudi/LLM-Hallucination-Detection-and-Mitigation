import json
import sys
import numpy as np
import pandas as pd
from detector import load_scored
from labels import correct_verified
from metrics import is_abstain

SPLITS = sys.argv[1:] or ["dev", "val", "test"]
T = json.load(open("results/chosen_detector.json"))["threshold"]
ABSTAIN = "Insufficient evidence"
NAMES = ["B", "B + verify", "C", "C + verify", "D keep", "D abstain"]
PAIRS = (("B + verify", "B"), ("C + verify", "C"), ("D keep", "C"),
         ("D keep", "C + verify"), ("D keep", "B + verify"))
rng = np.random.default_rng(42)


def flagged(sup_map, qid, pred):
    if is_abstain(pred):
        return False
    s = sup_map.get(int(qid), np.nan)
    return (not np.isnan(s)) and ((1 - s) >= T)


rows = []
for split in SPLITS:
    v = load_scored(split)
    sup = {src: v[v.source == src].set_index("id")["sup_mean"].to_dict() for src in ("B", "C")}
    b = pd.read_json(f"data/rag_b_k3_{split}.jsonl", lines=True).set_index("id")
    c = pd.read_json(f"data/rag_c_k3_{split}.jsonl", lines=True).set_index("id")
    d = {}
    with open(f"data/mitigate_{split}.jsonl") as f:
        for line in f:
            s = json.loads(line)
            d[int(s["id"])] = s
    for qid in sorted(set(c.index) & set(b.index) & set(d)):
        q, g = str(c.loc[qid, "question"]), str(c.loc[qid, "gold"])
        pb, pc = str(b.loc[qid, "pred"]), str(c.loc[qid, "pred"])
        preds = {
            "B": pb,
            "B + verify": ABSTAIN if flagged(sup["B"], qid, pb) else pb,
            "C": pc,
            "C + verify": ABSTAIN if flagged(sup["C"], qid, pc) else pc,
            "D keep": d[int(qid)]["final_keep"],
            "D abstain": d[int(qid)]["final_abs"],
        }
        rec = {"split": split}
        for name, p in preds.items():
            ab = bool(is_abstain(p))
            ok = bool(correct_verified(p, g, q))
            rec[name] = (ok, (not ab) and (not ok), ab)
        rows.append(rec)


def table(sel, title):
    print(f"\n{title} (n={len(sel)})")
    print(f"{'config':<12} {'correct':>8} {'wrong&ans':>10} {'abstained':>10}")
    for nm in NAMES:
        arr = np.array([r[nm] for r in sel], dtype=float)
        print(f"{nm:<12} {arr[:, 0].mean():>8.3f} {arr[:, 1].mean():>10.3f} {arr[:, 2].mean():>10.3f}")


def paired(sel, title):
    n = len(sel)
    arr = {nm: np.array([r[nm] for r in sel], dtype=float) for nm in NAMES}
    idx = rng.integers(0, n, size=(5000, n))
    print(f"\nPaired differences, {title} (bootstrap 95% interval)")
    for x, y in PAIRS:
        out = []
        for col, label in ((0, "correct"), (1, "wrong-and-answered")):
            dd = arr[x][:, col] - arr[y][:, col]
            lo, hi = np.percentile(dd[idx].mean(axis=1), [2.5, 97.5])
            out.append(f"{label} {dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {x} minus {y}: " + "; ".join(out))


for sp in SPLITS:
    table([r for r in rows if r["split"] == sp], f"Split {sp}")
if len(SPLITS) > 1:
    table(rows, "Pooled over " + ", ".join(SPLITS))
if "test" in SPLITS:
    paired([r for r in rows if r["split"] == "test"], "test only")
if len(SPLITS) > 1:
    paired(rows, "pooled")