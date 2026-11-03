import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from detector import load_scored

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
RULE = "maximize F0.5 on pooled B and C answers of the split (label = wrong under the lenient metric)"

v = load_scored(SPLIT)
rs = pd.read_csv(f"results/rescored_{SPLIT}.csv")

cfg = {"B RAG (k=3)": "B", "C RAG+grounded (k=3)": "C"}
r2 = rs[rs.config.isin(cfg)].copy()
r2["source"] = r2.config.map(cfg)
r2["y"] = 1 - r2["lenient"].astype(int)
pipe = v[v.source.isin(["B", "C"])].merge(r2[["id", "source", "y"]], on=["id", "source"])

pairs = v[v.source.isin(["right", "halluc"])].copy()
pairs["y"] = (pairs.source == "halluc").astype(int)

VARIANTS = {"mean": "sup_mean", "max": "sup_max", "nli": "sup_nli"}   # nli is for reference only
GRID = [round(float(t), 2) for t in np.arange(0.30, 0.801, 0.05)]


def score_metrics(y, score, t):
    pred = (score >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f05 = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) else 0.0
    return {"flagged": tp + fp, "tp": tp, "fp": fp, "P": p, "R": r, "F1": f1, "F05": f05}


print(f"Split: {SPLIT}")
print(f"Pipeline answers: {len(pipe)} {pipe.source.value_counts().to_dict()}, wrong: {int(pipe.y.sum())}")
print(f"HaluEval pair claims: {len(pairs)}")
print(f"Selection rule: {RULE}\n")

best = None
for name, col in VARIANTS.items():
    s_pipe, s_pair = 1 - pipe[col], 1 - pairs[col]
    print(f"Variant {name}: AUC pipeline = {roc_auc_score(pipe.y, s_pipe):.3f}, "
          f"AUC pairs = {roc_auc_score(pairs.y, s_pair):.3f}")
    print("    t  flagged  TP  FP      P      R     F1   F0.5")
    for t in GRID:
        m = score_metrics(pipe.y, s_pipe, t)
        print(f" {t:.2f}   {m['flagged']:>5} {m['tp']:>3} {m['fp']:>3}  {m['P']:.3f}  {m['R']:.3f}  "
              f"{m['F1']:.3f}  {m['F05']:.3f}")
        if name != "nli":
            key = (m["F05"], m["F1"], t)          # ties: higher F1, then higher threshold
            if best is None or key > best[0]:
                best = (key, name, col, t, m)
    print()

_, name, col, t, m = best
mp = score_metrics(pairs.y, 1 - pairs[col], t)
print("=" * 64)
print(f"CHOSEN: variant = {name}, threshold = {t}")
print(f"  pipeline answers: flagged {m['flagged']}, TP {m['tp']}, FP {m['fp']}, "
      f"P={m['P']:.3f}, R={m['R']:.3f}, F1={m['F1']:.3f}, F0.5={m['F05']:.3f}")
print(f"  HaluEval pairs  : P={mp['P']:.3f}, R={mp['R']:.3f}, F1={mp['F1']:.3f}")
if int(pipe.y.sum()) < 25:
    print("  NOTE: fewer than 25 wrong answers in this sample, so this choice is noisy.")

os.makedirs("results", exist_ok=True)
with open("results/chosen_detector.json", "w") as f:
    json.dump({"split": SPLIT, "variant": name, "support_column": col, "threshold": t,
               "rule": RULE, "n_pipeline": len(pipe), "n_wrong": int(pipe.y.sum()),
               "sources": sorted(pipe.source.unique().tolist())}, f, indent=2)
print("Saved results/chosen_detector.json")