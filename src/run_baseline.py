import json
import sys
import pandas as pd
from tqdm import tqdm
from llm import ask_llm
from metrics import exact_match, contains_match, f1_score

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
N = int(sys.argv[2]) if len(sys.argv) > 2 else None

df = pd.read_json(f"data/{SPLIT}.jsonl", lines=True)
if N:
    df = df.head(N)

PROMPT = (
    "Answer the question in as few words as possible. "
    "Give only the answer, no explanation.\n\n"
    "Question: {q}\nAnswer:"
)

rows = []
for _, r in tqdm(df.iterrows(), total=len(df)):
    out = ask_llm(PROMPT.format(q=r["question"]), with_meta=True)
    pred = out["text"]
    rows.append({
        "id": int(r["id"]),
        "question": r["question"],
        "gold": r["right_answer"],
        "pred": pred,
        "em": exact_match(pred, r["right_answer"]),
        "contains": contains_match(pred, r["right_answer"]),
        "f1": f1_score(pred, r["right_answer"]),
        "latency": out["latency"],
        "model": out["model_version"],
        "tokens": out.get("tokens", 0),
    })

res = pd.DataFrame(rows)
res.to_json(f"data/baseline_{SPLIT}.jsonl", orient="records", lines=True)

print(f"\nBaseline on {SPLIT} ({len(res)} questions)")
print(f"Exact match : {res.em.mean():.3f}")
print(f"Contains    : {res.contains.mean():.3f}")
print(f"Token F1    : {res.f1.mean():.3f}")
print(f"Avg latency : {res.latency.mean():.2f}s")
print(f"Model(s)    : {res.model.unique().tolist()}")
print(f"Avg tokens  : {res.tokens.mean():.0f}  (total {res.tokens.sum():.0f})")