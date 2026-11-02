import pandas as pd
from sklearn.metrics import roc_auc_score

SPLIT = "dev"
THRESH = 0.5

v = pd.read_json(f"data/verified_{SPLIT}.jsonl", lines=True)
b = pd.read_json(f"data/rag_b_k3_{SPLIT}.jsonl", lines=True).set_index("id")
base = pd.read_json(f"data/{SPLIT}.jsonl", lines=True).set_index("id")

pairs = v[v.source.isin(["right", "halluc"])].copy()
pairs["y"] = (pairs.source == "halluc").astype(int)
pairs["own_ret"] = pairs["id"].map(b["own_row_retrieved"].astype(bool).to_dict())

# 1. Score variants
print("1. AUC of different scores on HaluEval pairs")
for ev in ("ret", "orc"):
    e, n, c = pairs[f"entail_{ev}"], pairs[f"neutral_{ev}"], pairs[f"contra_{ev}"]
    variants = {"1 - entail": 1 - e, "contradiction": c, "contra - entail": c - e, "neutral": n}
    line = "  ".join(f"{k}={roc_auc_score(pairs.y, s):.3f}" for k, s in variants.items())
    print(f"  evidence={ev}: {line}")

# 2. Right claims flagged, split by whether retrieval found the source passage
right = pairs[pairs.source == "right"].copy()
right["flagged_ret"] = (1 - right["entail_ret"]) >= THRESH
right["flagged_orc"] = (1 - right["entail_orc"]) >= THRESH
print("\n2. Right-answer claims flagged as hallucinated (should be rare)")
print(right.groupby("own_ret")["flagged_ret"].agg(["sum", "count", "mean"]).round(3)
      .rename(index={True: "retrieval found source", False: "retrieval failed"}).to_string())
print(f"  with oracle evidence: {int(right.flagged_orc.sum())} of {len(right)} flagged")


def show(df, title, n=8):
    print("\n" + "=" * 80 + f"\n{title}")
    for _, r in df.head(n).iterrows():
        print(f"\nQ: {r.question[:110]}\n  claim: {r.claim}"
              f"\n  entail(oracle)={r.entail_orc:.2f}  entail(retrieved)={r.entail_ret:.2f}"
              f"\n  knowledge: {str(base.loc[int(r.id), 'knowledge'])[:300]}")


# 3. Examples
show(right.sort_values("entail_orc"), "3a. RIGHT claims with the LOWEST oracle entailment (false alarms)")
miss = pairs[pairs.source == "halluc"].sort_values("entail_orc", ascending=False)
show(miss, "3b. HALLUCINATED claims with the HIGHEST oracle entailment (misses)")