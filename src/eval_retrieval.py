import os
import sys
import pandas as pd
from retrieve import Retriever
from metrics import normalize

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
CW = int(sys.argv[2]) if len(sys.argv) > 2 else 120
KS = [1, 3, 5, 10]

df = pd.read_json(f"data/{SPLIT}.jsonl", lines=True)
ret = Retriever(CW)
results = ret.search_many(df["question"].tolist(), k=max(KS))

hit = {k: [] for k in KS}
ans_hit = {k: [] for k in KS}
rr, rows = [], []

for (_, row), res in zip(df.iterrows(), results):
    qid = int(row["id"])
    gold = normalize(row["right_answer"])
    rel = [qid in c["source_ids"] for c in res]
    has_ans = [gold != "" and gold in normalize(c["text"]) for c in res]
    first = next((i for i, x in enumerate(rel) if x), None)
    rr.append(1 / (first + 1) if first is not None else 0.0)
    for k in KS:
        hit[k].append(any(rel[:k]))
        ans_hit[k].append(any(has_ans[:k]))
    rows.append({
        "id": qid,
        "question": row["question"],
        "gold": row["right_answer"],
        "first_relevant_rank": None if first is None else first + 1,
        "top1_chunk": res[0]["text"],
    })

print(f"\nRetrieval on {SPLIT} ({len(df)} questions), chunk size {CW} words")
for k in KS:
    print(f"Recall@{k:<2}        : {sum(hit[k]) / len(df):.3f}")
for k in KS:
    print(f"AnswerInChunk@{k:<2}: {sum(ans_hit[k]) / len(df):.3f}")
print(f"MRR@{max(KS)}           : {sum(rr) / len(rr):.3f}")

os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).to_csv(f"results/retrieval_{SPLIT}_{CW}.csv", index=False)
print(f"\nPer-question details saved to results/retrieval_{SPLIT}_{CW}.csv")