import json
import sys
import time
from collections import Counter
import numpy as np
import pandas as pd
from retrieve import Retriever            # import first (sets thread environment variables)
from llm import ask_llm
from metrics import is_abstain
from detector import lex_support, load_scored
from labels import correct_lenient, correct_verified
from nli import nli_scores

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
N = int(sys.argv[2]) if len(sys.argv) > 2 else None
KS = [5, 10]                              # evidence depth of retry rounds 1 and 2 (frozen)
ABSTAIN = "Insufficient evidence"

chosen = json.load(open("results/chosen_detector.json"))
T, VARIANT = chosen["threshold"], chosen["variant"]
SUP_COL = {"mean": "sup_mean", "max": "sup_max", "nli": "sup_nli"}[VARIANT]

PROMPT_C = (
    "Answer the question using ONLY the evidence below. "
    "Do not use outside knowledge. "
    "If the evidence does not contain the answer, reply exactly: Insufficient evidence.\n"
    "Otherwise give only the answer in as few words as possible, no explanation.\n\n"
    "Evidence:\n{evidence}\n\n"
    "Question: {q}\nAnswer:"
)
CLAIM_PROMPT = (
    "Rewrite the question and its answer as ONE short, self-contained declarative sentence "
    "that states the answer as a fact. Do not add any information that is not in the "
    "question or the answer. If the answer is Yes or No, state the fact accordingly. "
    "Output only the sentence.\n\n"
    "Question: {q}\nAnswer: {a}\nSentence:"
)


def support(entail, lex):
    lexf = entail if np.isnan(lex) else lex
    return (entail + lexf) / 2 if VARIANT == "mean" else max(entail, lexf)


# ---- first pass: Config C answers and their detector scores (already computed) ----
c = pd.read_json(f"data/rag_c_k3_{SPLIT}.jsonl", lines=True).set_index("id")
b = pd.read_json(f"data/rag_b_k3_{SPLIT}.jsonl", lines=True).set_index("id")
v = load_scored(SPLIT)
sup1 = v[v.source == "C"].set_index("id")[SUP_COL].to_dict()

states = []
for qid, r in c.iterrows():
    a1 = str(r["pred"])
    abst1 = is_abstain(a1)
    s1 = None if abst1 else float(sup1.get(int(qid), np.nan))
    flag1 = (s1 is not None) and (not np.isnan(s1)) and ((1 - s1) >= T)
    states.append({
        "id": int(qid), "question": r["question"], "gold": str(r["gold"]),
        "a1": a1, "abst1": abst1, "sup1": s1, "flag1": bool(flag1),
        "status": "pending" if (abst1 or flag1) else "ok",
        "cands": [] if abst1 else [(a1, s1)], "rounds": [],
    })
if N:
    states = states[:N]

print(f"Split {SPLIT}: {len(states)} questions | detector {VARIANT}, threshold {T}")
print(f"Triggered: {sum(s['abst1'] for s in states)} abstained in C, "
      f"{sum(s['flag1'] for s in states)} answers flagged")

# ---- retry rounds ----
ret = Retriever(120)
tokens, calls, t0 = 0, 0, time.time()
for K in KS:
    pend = [s for s in states if s["status"] == "pending"]
    if not pend:
        break
    hits = ret.search_many([s["question"] for s in pend], k=K)
    todo = []
    for s, h in zip(pend, hits):
        evid_plain = "\n".join(x["text"] for x in h)
        evid_marked = "\n".join(f"[{i}] {x['text']}" for i, x in enumerate(h, 1))
        out = ask_llm(PROMPT_C.format(evidence=evid_marked, q=s["question"]), with_meta=True)
        tokens += out.get("tokens", 0); calls += 1
        a2 = out["text"]
        rec = {"K": K, "answer": a2, "abstained": bool(is_abstain(a2))}
        s["rounds"].append(rec)
        if rec["abstained"]:
            continue
        cl = ask_llm(CLAIM_PROMPT.format(q=s["question"], a=a2), with_meta=True)
        tokens += cl.get("tokens", 0); calls += 1
        rec["claim"] = cl["text"]
        todo.append((s, rec, evid_plain))
    if todo:
        sc = nli_scores([e for _, _, e in todo], [r_["claim"] for _, r_, _ in todo])
        for (s, rec, evid_plain), sco in zip(todo, sc):
            lex = lex_support(rec["claim"], s["question"], evid_plain)
            sup2 = float(support(sco["entail"], lex))
            rec["support"] = sup2
            rec["flagged"] = bool((1 - sup2) >= T)
            s["cands"].append((rec["answer"], sup2))
            if not rec["flagged"]:
                s["status"], s["accepted_K"], s["accepted_answer"] = "accepted", K, rec["answer"]

for s in states:
    if s["status"] == "pending":
        s["status"] = "fallback"
    if s["status"] == "ok":
        s["final_keep"] = s["final_abs"] = s["a1"]
    elif s["status"] == "accepted":
        s["final_keep"] = s["final_abs"] = s["accepted_answer"]
    else:
        s["final_abs"] = ABSTAIN
        s["final_keep"] = (max(s["cands"], key=lambda x: (x[1], x[0] == s["a1"]))[0]
                           if s["cands"] else ABSTAIN)

# ---- evaluation ----
def summarize(name, preds):
    ab = np.array([is_abstain(p) for p in preds])
    cl = np.array([correct_lenient(p, s["gold"]) for p, s in zip(preds, states)])
    cv = np.array([correct_verified(p, s["gold"], s["question"]) for p, s in zip(preds, states)])
    print(f"{name:<24} {cl.mean():>8.2f} {cv.mean():>9.2f} {ab.mean():>10.2f} "
          f"{((~ab) & (~cl)).mean():>13.2f} {((~ab) & (~cv)).mean():>13.2f}")


print("\n" + "=" * 78)
print(f"{'config':<24} {'correct':>8} {'correct':>9} {'abstained':>10} {'wrong&ans':>13} {'wrong&ans':>13}")
print(f"{'':<24} {'(lenient)':>8} {'(verified)':>9} {'':>10} {'(lenient)':>13} {'(verified)':>13}")
summarize("B plain RAG", [str(b.loc[s["id"], "pred"]) for s in states])
summarize("C grounded (start)", [s["a1"] for s in states])
summarize("D keep best-supported", [s["final_keep"] for s in states])
summarize("D abstain if flagged", [s["final_abs"] for s in states])

print("\nLoop statistics")
print(f"  status counts: {dict(Counter(s['status'] for s in states))}")
print(f"  accepted per round (K): {dict(Counter(s.get('accepted_K') for s in states if s['status'] == 'accepted'))}")
print(f"  API calls: {calls}, tokens (including cached calls): {tokens}, time: {time.time() - t0:.0f}s")


def label(p, s):
    if is_abstain(p):
        return "abstain"
    return "correct" if correct_verified(p, s["gold"], s["question"]) else "wrong"


for pol, key in (("keep best-supported", "final_keep"), ("abstain if flagged", "final_abs")):
    cnt = Counter((label(s["a1"], s), label(s[key], s)) for s in states)
    print(f"\nTransitions, C -> D ({pol}), verified labels:")
    for (a, d), n in sorted(cnt.items()):
        print(f"  C {a:<8} -> D {d:<8} {n}")

with open(f"data/mitigate_{SPLIT}.jsonl", "w") as f:
    for s in states:
        rec = dict(s)
        rec["cands"] = [list(x) for x in s["cands"]]
        f.write(json.dumps(rec, default=float) + "\n")
print(f"\nSaved data/mitigate_{SPLIT}.jsonl")