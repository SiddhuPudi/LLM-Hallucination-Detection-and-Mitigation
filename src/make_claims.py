import sys
import pandas as pd
from tqdm import tqdm
from llm import ask_llm
from metrics import is_abstain

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
N = int(sys.argv[2]) if len(sys.argv) > 2 else None

PROMPT = (
    "Rewrite the question and its answer as ONE short, self-contained declarative sentence "
    "that states the answer as a fact. Do not add any information that is not in the "
    "question or the answer. If the answer is Yes or No, state the fact accordingly. "
    "Output only the sentence.\n\n"
    "Question: {q}\nAnswer: {a}\nSentence:"
)

df = pd.read_json(f"data/{SPLIT}.jsonl", lines=True)
if N:
    df = df.head(N)

B = pd.read_json(f"data/rag_b_k3_{SPLIT}.jsonl", lines=True).set_index("id")["pred"].astype(str).to_dict()
C = pd.read_json(f"data/rag_c_k3_{SPLIT}.jsonl", lines=True).set_index("id")["pred"].astype(str).to_dict()

items = []
for _, r in df.iterrows():
    qid = int(r["id"])
    items.append((qid, "right", r["question"], str(r["right_answer"])))
    items.append((qid, "halluc", r["question"], str(r["hallucinated_answer"])))
    for src, preds in (("B", B), ("C", C)):
        a = preds.get(qid, "")
        if a and not is_abstain(a):
            items.append((qid, src, r["question"], a))

rows, total_tokens = [], 0
for qid, src, q, a in tqdm(items):
    out = ask_llm(PROMPT.format(q=q, a=a), with_meta=True)
    total_tokens += out.get("tokens", 0)
    rows.append({"id": qid, "source": src, "question": q, "answer": a, "claim": out["text"]})

res = pd.DataFrame(rows)
res.to_json(f"data/claims_{SPLIT}.jsonl", orient="records", lines=True)

print(f"\n{len(res)} claims written to data/claims_{SPLIT}.jsonl")
print(res["source"].value_counts().to_string())
print(f"Total tokens (including cached calls): {total_tokens}")
print("\nSamples:")
for _, r in res.groupby("source").head(2).iterrows():
    print(f"\n[{r.source}] Q: {r.question[:100]}\n  answer: {r.answer[:100]}\n  claim:  {r.claim}")