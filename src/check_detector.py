import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from detector import load_scored
from labels import correct_verified

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
chosen = json.load(open("results/chosen_detector.json"))
T0, COL0 = chosen["threshold"], chosen["support_column"]

# Short-form answers judged INCOMPLETE by hand (these are NOT counted as correct).
# This is where your own verdicts enter: edit this list if you disagree with a row in
# Appendix A or A2 of the report.
INCOMPLETE = ("Dharmatama", "When was the movie released in 2007", "What job do both Herbert Selpin")

GRID = [round(float(t), 2) for t in np.arange(0.30, 0.801, 0.05)]

v = load_scored(SPLIT)
rs = pd.read_csv(f"results/rescored_{SPLIT}.csv")

cfg = {"B RAG (k=3)": "B", "C RAG+grounded (k=3)": "C"}
r2 = rs[rs.config.isin(cfg)].copy()
r2["source"] = r2.config.map(cfg)
r2["y_len"] = 1 - r2["lenient"].astype(int)                       # original labels
r2["y_cor"] = [1 - int(correct_verified(p, g, q)) for p, g, q in
               zip(r2["pred"].astype(str), r2["gold"].astype(str), r2["question"].astype(str))]
pipe = v[v.source.isin(["B", "C"])].merge(r2[["id", "source", "y_len", "y_cor"]], on=["id", "source"])

pairs = v[v.source.isin(["right", "halluc"])].copy()
pairs["y"] = (pairs.source == "halluc").astype(int)


def pm(y, score, t):
    pred = (score >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f05 = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) else 0.0
    return {"flagged": tp + fp, "tp": tp, "fp": fp, "P": p, "R": r, "F1": f1, "F05": f05}


print(f"Chosen detector (selected on validation): {chosen['variant']}, threshold {T0}")
print(f"Applied to split: {SPLIT}   (pipeline answers: {len(pipe)}, HaluEval pair claims: {len(pairs)})")

# ---- A. The chosen detector applied as is -------------------------------------------
for lab, desc in (("y_len", "original labels (lenient metric)"),
                  ("y_cor", "corrected labels (short-form answers count as correct)")):
    y, s = pipe[lab], 1 - pipe[COL0]
    m = pm(y, s, T0)
    print(f"\nLabels: {desc}. Wrong answers: {int(y.sum())} of {len(y)}")
    print(f"  all answers: AUC={roc_auc_score(y, s):.3f} | flagged {m['flagged']} (TP {m['tp']}, FP {m['fp']}) "
          f"P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} F0.5={m['F05']:.3f}")
    for src in ("B", "C"):
        sub = pipe[pipe.source == src]
        if len(sub):
            ms = pm(sub[lab], 1 - sub[COL0], T0)
            print(f"  Config {src}: answered {len(sub)}, wrong {int(sub[lab].sum())}, flagged {ms['flagged']} "
                  f"(TP {ms['tp']}, FP {ms['fp']}), P={ms['P']:.3f}, R={ms['R']:.3f}")

if len(pairs):
    mp = pm(pairs.y, 1 - pairs[COL0], T0)
    print(f"\nHaluEval pairs: AUC={roc_auc_score(pairs.y, 1 - pairs[COL0]):.3f} | "
          f"P={mp['P']:.3f} R={mp['R']:.3f} F1={mp['F1']:.3f}")
else:
    print("\n(no HaluEval pair claims for this split yet)")

# ---- B. Sensitivity of the threshold to the labels (validation only) ----------------
if SPLIT == "val":
    print("\n" + "=" * 70 + "\nSENSITIVITY OF THE VARIANT AND THRESHOLD TO THE LABELS")
    for lab in ("y_len", "y_cor"):
        y = pipe[lab]
        print(f"\nLabels {lab} (wrong answers: {int(y.sum())})")
        for name, col in (("mean", "sup_mean"), ("max", "sup_max"), ("nli", "sup_nli")):
            s = 1 - pipe[col]
            rows = [(pm(y, s, t), t) for t in GRID]
            bm, bt = max(rows, key=lambda x: (x[0]["F05"], x[0]["F1"], x[1]))
            at = pm(y, s, T0)
            print(f"  {name:<5} AUC={roc_auc_score(y, s):.3f} | at t={T0}: P={at['P']:.3f} R={at['R']:.3f} "
                  f"F0.5={at['F05']:.3f} | best F0.5={bm['F05']:.3f} at t={bt}")

    y, s = pipe["y_cor"], 1 - pipe[COL0]
    print(f"\nThreshold curve of the chosen variant under corrected labels")
    print("    t  flagged  TP  FP      P      R     F1   F0.5")
    rows = []
    for t in GRID:
        m = pm(y, s, t)
        rows.append((m, t))
        print(f" {t:.2f}   {m['flagged']:>5} {m['tp']:>3} {m['fp']:>3}  {m['P']:.3f}  {m['R']:.3f}  "
              f"{m['F1']:.3f}  {m['F05']:.3f}")
    bm, bt = max(rows, key=lambda x: (x[0]["F05"], x[0]["F1"], x[1]))
    at = pm(y, s, T0)
    far = round(abs(bt - T0), 2) >= 0.10
    better = (bm["F05"] - at["F05"]) >= 0.05
    print(f"\nDECISION RULE: best corrected-label threshold = {bt} (F0.5 {bm['F05']:.3f}); "
          f"F0.5 at {T0} = {at['F05']:.3f}")
    print(f"  at least 0.10 away: {far}; at least 0.05 higher F0.5: {better}")
    print("  => " + ("CHANGE the threshold" if (far and better) else f"KEEP the threshold {T0}"))