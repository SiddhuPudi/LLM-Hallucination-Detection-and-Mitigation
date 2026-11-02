import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from metrics import lenient_tokens

SPLIT = "dev"
THRESH = 0.5   # placeholder; final threshold is tuned on validation only

STOP = set("""is are was were be been being has have had do does did to in on at by for with as
that which who whom whose what when where it its this these those his her their he she they
them from into than then both also not no yes more most""".split())

v = pd.read_json(f"data/verified_{SPLIT}.jsonl", lines=True)
b = pd.read_json(f"data/rag_b_k3_{SPLIT}.jsonl", lines=True).set_index("id")
base = pd.read_json(f"data/{SPLIT}.jsonl", lines=True).set_index("id")
rs = pd.read_csv(f"results/rescored_{SPLIT}.csv")

ev_ret = {int(i): re.sub(r"\[\d+\]\s*", "", e) for i, e in b["evidence"].items()}
ev_orc = {int(i): str(k) for i, k in base["knowledge"].items()}


def lex_support(claim, question, evidence):
    q = set(lenient_tokens(question))
    novel = [t for t in lenient_tokens(claim) if t not in q and t not in STOP]
    if not novel:
        return np.nan                       # claim adds no new words: rely on NLI
    ev = set(lenient_tokens(evidence))
    return sum(t in ev for t in novel) / len(novel)


for name, evmap in (("ret", ev_ret), ("orc", ev_orc)):
    v[f"lex_{name}"] = [lex_support(c, q, evmap[int(i)]) for c, q, i in zip(v.claim, v.question, v.id)]
    lex = v[f"lex_{name}"].fillna(v[f"entail_{name}"])
    v[f"lexf_{name}"] = lex
    v[f"supmax_{name}"] = np.maximum(v[f"entail_{name}"], lex)
    v[f"supmean_{name}"] = (v[f"entail_{name}"] + lex) / 2


def evaluate(df, title):
    print(f"\n{title}  (n={len(df)})")
    for ev in ("ret", "orc"):
        variants = {
            "NLI only (1-entail)": 1 - df[f"entail_{ev}"],
            "lexical only": 1 - df[f"lexf_{ev}"],
            "max(NLI, lexical)": 1 - df[f"supmax_{ev}"],
            "mean(NLI, lexical)": 1 - df[f"supmean_{ev}"],
        }
        for vn, s in variants.items():
            pred = (s >= THRESH).astype(int)
            p, r, f, _ = precision_recall_fscore_support(df["y"], pred, average="binary", zero_division=0)
            print(f"  ev={ev:<3} {vn:<22} AUC={roc_auc_score(df['y'], s):.3f}  P={p:.3f} R={r:.3f} F1={f:.3f}")


# 1. HaluEval pairs
pairs = v[v.source.isin(["right", "halluc"])].copy()
pairs["y"] = (pairs.source == "halluc").astype(int)
evaluate(pairs, "1. HaluEval pairs (right = 0, hallucinated = 1)")

print("\nConfusion matrices, retrieved evidence (rows: right/halluc, cols: kept/flagged)")
for vn, col in (("NLI only", "entail_ret"), ("max(NLI, lexical)", "supmax_ret"), ("mean(NLI, lexical)", "supmean_ret")):
    cm = confusion_matrix(pairs.y, ((1 - pairs[col]) >= THRESH).astype(int))
    print(f"  {vn:<20} {cm.tolist()}")

# 2. Pipeline answers (label: wrong under the lenient metric)
cfg = {"B RAG (k=3)": "B", "C RAG+grounded (k=3)": "C"}
r2 = rs[rs.config.isin(cfg)].copy()
r2["source"] = r2.config.map(cfg)
r2["y"] = 1 - r2["lenient"].astype(int)
pipe = v[v.source.isin(["B", "C"])].merge(r2[["id", "source", "y"]], on=["id", "source"])
print("\n2. Pipeline answers (flag the wrong ones)")
evaluate(pipe[pipe.source == "B"], "Config B answers")
evaluate(pipe[pipe.source == "C"], "Config C answers")