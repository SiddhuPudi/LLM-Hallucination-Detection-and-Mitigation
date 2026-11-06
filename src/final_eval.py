import json
import sys
from collections import Counter
import numpy as np
import pandas as pd
from labels import correct_lenient, correct_verified
from metrics import is_abstain, contains_match

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
N_BOOT = 5000
rng = np.random.default_rng(42)

a = pd.read_json(f"data/baseline_{SPLIT}.jsonl", lines=True).set_index("id")
b = pd.read_json(f"data/rag_b_k3_{SPLIT}.jsonl", lines=True).set_index("id")
c = pd.read_json(f"data/rag_c_k3_{SPLIT}.jsonl", lines=True).set_index("id")
d = {}
with open(f"data/mitigate_{SPLIT}.jsonl") as f:
    for line in f:
        s = json.loads(line)
        d[int(s["id"])] = s

ids = sorted(set(a.index) & set(b.index) & set(c.index) & set(d))
golds = [str(c.loc[i, "gold"]) for i in ids]
quests = [str(c.loc[i, "question"]) for i in ids]
n = len(ids)

preds = {
    "A baseline": [str(a.loc[i, "pred"]) for i in ids],
    "B plain RAG": [str(b.loc[i, "pred"]) for i in ids],
    "C grounded": [str(c.loc[i, "pred"]) for i in ids],
    "D keep": [d[i]["final_keep"] for i in ids],
    "D abstain": [d[i]["final_abs"] for i in ids],
}

F = {}
for name, ps in preds.items():
    ab = np.array([is_abstain(p) for p in ps])
    st = np.array([bool(contains_match(p, g)) and not x for p, g, x in zip(ps, golds, ab)])
    le = np.array([correct_lenient(p, g) for p, g in zip(ps, golds)])
    ve = np.array([correct_verified(p, g, q) for p, g, q in zip(ps, golds, quests)])
    F[name] = {"ab": ab, "strict": st, "len": le, "ver": ve,
               "wrong_len": (~ab) & (~le), "wrong_ver": (~ab) & (~ve)}

idx = rng.integers(0, n, size=(N_BOOT, n))


def ci(arr):
    return np.percentile(arr.astype(float)[idx].mean(axis=1), [2.5, 97.5])


def ci_diff(x, y):
    return np.percentile((x.astype(float) - y.astype(float))[idx].mean(axis=1), [2.5, 97.5])


print(f"Final evaluation on {SPLIT}: {n} questions (bootstrap 95% intervals, {N_BOOT} resamples)\n")
print(f"{'config':<12} {'strict':>6} {'lenient':>7} {'verified':>8} {'95% CI':>14}  {'abstain':>7} "
      f"{'wrong(len)':>10} {'wrong(ver)':>10} {'95% CI':>14}")
for name, f in F.items():
    lo, hi = ci(f["ver"])
    wlo, whi = ci(f["wrong_ver"])
    print(f"{name:<12} {f['strict'].mean():>6.2f} {f['len'].mean():>7.2f} {f['ver'].mean():>8.2f} "
          f"[{lo:.2f},{hi:.2f}]  {f['ab'].mean():>7.2f} {f['wrong_len'].mean():>10.2f} "
          f"{f['wrong_ver'].mean():>10.2f} [{wlo:.2f},{whi:.2f}]")

print("\nPaired differences (verified labels; bootstrap 95% interval)")
for x, y in (("D keep", "C grounded"), ("D keep", "B plain RAG"), ("C grounded", "B plain RAG"),
             ("B plain RAG", "A baseline"), ("D keep", "A baseline")):
    dc = F[x]["ver"].astype(float) - F[y]["ver"].astype(float)
    dw = F[x]["wrong_ver"].astype(float) - F[y]["wrong_ver"].astype(float)
    lo, hi = ci_diff(F[x]["ver"], F[y]["ver"])
    wlo, whi = ci_diff(F[x]["wrong_ver"], F[y]["wrong_ver"])
    print(f"  {x} minus {y}: correct {dc.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]; "
          f"wrong-and-answered {dw.mean():+.3f} [{wlo:+.3f}, {whi:+.3f}]")

D, C, B = F["D keep"], F["C grounded"], F["B plain RAG"]
primary = D["ver"].sum() > C["ver"].sum() and D["wrong_ver"].sum() <= C["wrong_ver"].sum()
secondary = (D["wrong_ver"].sum() < B["wrong_ver"].sum()) and (D["ver"].mean() >= B["ver"].mean() - 0.03)
print(f"\nPRIMARY criterion (D keep: more verified-correct than C, no more wrong-and-answered): "
      f"{'MET' if primary else 'NOT MET'}")
print(f"SECONDARY criterion (D keep: fewer wrong than B, verified-correct within 3 points of B): "
      f"{'MET' if secondary else 'NOT MET'}")


def labels(name):
    f = F[name]
    return ["abstain" if f["ab"][k] else ("correct" if f["ver"][k] else "wrong") for k in range(n)]


for base in ("C grounded", "B plain RAG"):
    cnt = Counter(zip(labels(base), labels("D keep")))
    print(f"\nTransitions {base} -> D keep (verified labels):")
    for (x, y), m in sorted(cnt.items()):
        print(f"  {base[:1]} {x:<8} -> D {y:<8} {m}")

print("\nCost")
for name, df in (("A baseline", a), ("B plain RAG", b), ("C grounded", c)):
    print(f"  {name}: avg tokens {df.loc[ids, 'tokens'].mean():.0f}, avg API latency {df.loc[ids, 'latency'].mean():.2f}s")
print("  D: Config C cost plus the loop tokens printed by mitigate.py")