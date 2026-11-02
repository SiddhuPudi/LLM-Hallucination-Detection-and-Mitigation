import os
import pandas as pd
from metrics import is_abstain, lenient_match, short_form_match

FILES = {
    "A Baseline": "data/baseline_dev.jsonl",
    "B RAG (k=3)": "data/rag_b_k3_dev.jsonl",
    "C RAG+grounded (k=3)": "data/rag_c_k3_dev.jsonl",
}

rows, frames = [], {}
for name, path in FILES.items():
    d = pd.read_json(path, lines=True)
    d["pred"] = d["pred"].fillna("").astype(str)
    d["gold"] = d["gold"].astype(str)
    d["lenient"] = [lenient_match(p, g) for p, g in zip(d.pred, d.gold)]
    d["short"] = [short_form_match(p, g) for p, g in zip(d.pred, d.gold)]
    d["abst"] = d.pred.apply(is_abstain)
    short_only = (d.lenient == 0) & (d.short == 1) & (~d.abst)
    frames[name] = d
    rows.append({
        "config": name,
        "strict_contains": d["contains"].mean(),
        "lenient_contains": d["lenient"].mean(),
        "short_form_to_check": int(short_only.sum()),
        "abstained": d["abst"].mean(),
        "wrong_answered_strict": ((~d.abst) & (d.contains == 0)).mean(),
        "wrong_answered_lenient": ((~d.abst) & (d.lenient == 0)).mean(),
    })

print(pd.DataFrame(rows).round(3).to_string(index=False))

# Rows to verify by hand: short-form answers that were not auto-counted as correct
B, C = frames["B RAG (k=3)"], frames["C RAG+grounded (k=3)"]
check = C[["id", "question", "gold", "pred", "lenient", "short", "abst"]].merge(
    B[["id", "pred", "lenient", "short", "abst"]], on="id", suffixes=("_C", "_B")
)
mask = (
    ((check.lenient_C == 0) & (check.short_C == 1) & (~check.abst_C))
    | ((check.lenient_B == 0) & (check.short_B == 1) & (~check.abst_B))
)
print("\nSHORT-FORM ANSWERS TO VERIFY BY HAND:")
for _, r in check[mask].iterrows():
    print(f"\nQ: {r.question[:120]}\n  gold: {r.gold}\n  B:    {r.pred_B}\n  C:    {r.pred_C}")

os.makedirs("results", exist_ok=True)
pd.concat([f.assign(config=n) for n, f in frames.items()])[
    ["config", "id", "question", "gold", "pred", "contains", "lenient", "short", "abst"]
].to_csv("results/rescored_dev.csv", index=False)
print("\nSaved results/rescored_dev.csv")