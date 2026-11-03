import os
import sys
import pandas as pd
from metrics import is_abstain, lenient_match, short_form_match

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
FILES = {
    "A Baseline": f"data/baseline_{SPLIT}.jsonl",
    "B RAG (k=3)": f"data/rag_b_k3_{SPLIT}.jsonl",
    "C RAG+grounded (k=3)": f"data/rag_c_k3_{SPLIT}.jsonl",
}

rows, frames = [], {}
for name, path in FILES.items():
    if not os.path.exists(path):
        print(f"(skipping {name}: {path} not found)")
        continue
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
        "n": len(d),
        "strict_contains": d["contains"].mean(),
        "lenient_contains": d["lenient"].mean(),
        "short_form_to_check": int(short_only.sum()),
        "abstained": d["abst"].mean(),
        "wrong_answered_strict": ((~d.abst) & (d.contains == 0)).mean(),
        "wrong_answered_lenient": ((~d.abst) & (d.lenient == 0)).mean(),
    })

print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nSHORT-FORM ANSWERS TO VERIFY BY HAND (not counted as correct automatically):")
for name, d in frames.items():
    mask = (d.lenient == 0) & (d.short == 1) & (~d.abst)
    for _, r in d[mask].iterrows():
        print(f"\n[{name}] Q: {r.question[:110]}\n  gold: {r.gold}\n  pred: {r.pred}")

os.makedirs("results", exist_ok=True)
pd.concat([f.assign(config=n) for n, f in frames.items()])[
    ["config", "id", "question", "gold", "pred", "contains", "lenient", "short", "abst"]
].to_csv(f"results/rescored_{SPLIT}.csv", index=False)
print(f"\nSaved results/rescored_{SPLIT}.csv")