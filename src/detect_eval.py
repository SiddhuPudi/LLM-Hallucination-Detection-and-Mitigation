import sys
import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                             accuracy_score, confusion_matrix)

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   # placeholder; tune on val later

v = pd.read_json(f"data/verified_{SPLIT}.jsonl", lines=True)
rs = pd.read_csv(f"results/rescored_{SPLIT}.csv")


def report(df, name):
    for ev in ("ret", "orc"):
        score = 1 - df[f"entail_{ev}"]
        y = df["y"]
        pred = (score >= THRESH).astype(int)
        try:
            auc = f"{roc_auc_score(y, score):.3f}"
        except ValueError:
            auc = "n/a"
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        print(f"{name:<34} ev={ev:<4} n={len(df):<4} AUC={auc} acc={accuracy_score(y, pred):.3f} "
              f"P={p:.3f} R={r:.3f} F1={f:.3f}")


print(f"Detector: hallucination score = 1 - P(entailment); flagged if score >= {THRESH}\n")

# 1. HaluEval labeled pairs (right answer = 0, hallucinated answer = 1)
pairs = v[v.source.isin(["right", "halluc"])].copy()
pairs["y"] = (pairs.source == "halluc").astype(int)
piv = pairs.pivot(index="id", columns="source", values="claim")
same = piv.index[piv["right"].str.strip().str.lower() == piv["halluc"].str.strip().str.lower()]
print(f"Questions where the right and hallucinated claims are identical (label noise): {len(same)}\n")

print("1. HaluEval pairs")
report(pairs, "all pairs")
report(pairs[~pairs.id.isin(same)], "excluding identical-claim pairs")

# confusion matrix at the threshold (retrieved evidence, all pairs)
cm = confusion_matrix(pairs.y, ((1 - pairs.entail_ret) >= THRESH).astype(int))
print(f"\nConfusion matrix (retrieved evidence; rows = truth right/halluc, cols = predicted supported/flagged):\n{cm}")

# 2. Pipeline answers (label: wrong under the lenient metric)
cfg = {"B RAG (k=3)": "B", "C RAG+grounded (k=3)": "C"}
r2 = rs[rs.config.isin(cfg)].copy()
r2["source"] = r2.config.map(cfg)
r2["y"] = 1 - r2["lenient"].astype(int)
pipe = v[v.source.isin(["B", "C"])].merge(r2[["id", "source", "y"]], on=["id", "source"])

print("\n2. Pipeline answers (flag the wrong ones)")
report(pipe[pipe.source == "B"], "Config B answers")
report(pipe[pipe.source == "C"], "Config C answers")
report(pipe, "B and C together")

print("\nMean P(entailment): ")
print(pairs.groupby("source")[["entail_ret", "entail_orc"]].mean().round(3).to_string())
print(pipe.groupby("y")[["entail_ret", "entail_orc"]].mean().round(3).rename(index={0: "correct", 1: "wrong"}).to_string())