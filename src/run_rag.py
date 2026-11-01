import sys
import time
import pandas as pd
from tqdm import tqdm
from retrieve import Retriever
from llm import ask_llm
from metrics import exact_match, contains_match, f1_score, normalize, is_abstain

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
MODE = sys.argv[2].lower() if len(sys.argv) > 2 else "b"   # b = plain RAG, c = grounded + abstain
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
N = int(sys.argv[4]) if len(sys.argv) > 4 else None
CW = 120

PROMPTS = {
    "b": (
        "Use the evidence below to answer the question. "
        "Give only the answer in as few words as possible, no explanation.\n\n"
        "Evidence:\n{evidence}\n\n"
        "Question: {q}\nAnswer:"
    ),
    "c": (
        "Answer the question using ONLY the evidence below. "
        "Do not use outside knowledge. "
        "If the evidence does not contain the answer, reply exactly: Insufficient evidence.\n"
        "Otherwise give only the answer in as few words as possible, no explanation.\n\n"
        "Evidence:\n{evidence}\n\n"
        "Question: {q}\nAnswer:"
    ),
}

df = pd.read_json(f"data/{SPLIT}.jsonl", lines=True)
if N:
    df = df.head(N)

ret = Retriever(CW)
t0 = time.time()
all_hits = ret.search_many(df["question"].tolist(), k=K)
retr_latency = (time.time() - t0) / len(df)   # approximate per-question retrieval time

rows = []
for (_, r), hits in tqdm(zip(df.iterrows(), all_hits), total=len(df)):
    evidence = "\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    prompt = PROMPTS[MODE].format(evidence=evidence, q=r["question"])
    out = ask_llm(prompt, with_meta=True)
    pred = out["text"]
    gold = r["right_answer"]
    rows.append({
        "id": int(r["id"]),
        "question": r["question"],
        "gold": gold,
        "pred": pred,
        "abstained": bool(is_abstain(pred)),
        "em": exact_match(pred, gold),
        "contains": contains_match(pred, gold),
        "f1": f1_score(pred, gold),
        "evidence_has_answer": bool(normalize(gold) != "" and normalize(gold) in normalize(evidence)),
        "own_row_retrieved": bool(int(r["id"]) in {i for h in hits for i in h["source_ids"]}),
        "retrieved_ids": [h["chunk_id"] for h in hits],
        "evidence": evidence,
        "latency": out["latency"],
        "retrieval_latency": retr_latency,
        "tokens": out.get("tokens", 0),
        "model": out["model_version"],
    })

res = pd.DataFrame(rows)
out_path = f"data/rag_{MODE}_k{K}_{SPLIT}.jsonl"
res.to_json(out_path, orient="records", lines=True)

wrong_answered = ((~res.abstained) & (res.contains == 0)).mean()
print(f"\nConfig {MODE.upper()} RAG on {SPLIT} ({len(res)} questions, top-{K})")
print(f"Exact match         : {res.em.mean():.3f}")
print(f"Contains            : {res.contains.mean():.3f}")
print(f"Token F1            : {res.f1.mean():.3f}")
print(f"Abstained           : {res.abstained.mean():.3f}")
print(f"Wrong and answered  : {wrong_answered:.3f}")
print(f"Evidence had answer : {res.evidence_has_answer.mean():.3f}")
print(f"Own row retrieved   : {res.own_row_retrieved.mean():.3f}")
print(f"Avg latency         : LLM {res.latency.mean():.2f}s + retrieval {retr_latency:.3f}s")
print(f"Avg tokens          : {res.tokens.mean():.0f}  (total {res.tokens.sum():.0f})")
print(f"Saved to {out_path}")