import pandas as pd
from metrics import is_abstain

FILES = {
    "A Baseline": "data/baseline_dev.jsonl",
    "B RAG (k=3)": "data/rag_b_k3_dev.jsonl",
    "C RAG+grounded (k=3)": "data/rag_c_k3_dev.jsonl",
}

rows = []
for name, path in FILES.items():
    try:
        d = pd.read_json(path, lines=True)
    except FileNotFoundError:
        continue
    ab = d["pred"].fillna("").astype(str).apply(is_abstain)
    rows.append({
        "config": name,
        "n": len(d),
        "contains": d["contains"].mean(),
        "f1": d["f1"].mean(),
        "abstained": ab.mean(),
        "wrong_and_answered": ((~ab) & (d["contains"] == 0)).mean(),
        "avg_tokens": d["tokens"].mean() if "tokens" in d else None,
        "avg_llm_latency_s": d["latency"].mean(),
    })

print(pd.DataFrame(rows).round(3).to_string(index=False))