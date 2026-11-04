import json
import sys
from collections import Counter
from labels import correct_verified
from metrics import is_abstain

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
ABSTAIN = "Insufficient evidence"
states = [json.loads(line) for line in open(f"data/mitigate_{SPLIT}.jsonl")]


def lab(pred, s):
    if is_abstain(pred):
        return "abstain"
    return "correct" if correct_verified(pred, s["gold"], s["question"]) else "wrong"


def trigger(s):
    return "C abstained" if s["abst1"] else ("C flagged" if s["flag1"] else "not triggered")


print(f"Split {SPLIT}: {len(states)} questions")

print("\n1. Outcome by trigger type and loop status (C label -> D keep label)")
cnt = Counter((trigger(s), s["status"], lab(s["a1"], s), lab(s["final_keep"], s)) for s in states)
for (trg, st, a, d), n in sorted(cnt.items()):
    print(f"  {trg:<14} {st:<9} C {a:<8} -> D {d:<8} {n}")

print("\n2. Error rate of answers by verification status (keep policy)")
unv = [s for s in states if s["status"] == "fallback" and not is_abstain(s["final_keep"])]
ver = [s for s in states if s["status"] in ("ok", "accepted") and not is_abstain(s["final_keep"])]
for name, grp in (("unverified (still flagged)", unv), ("verified (not flagged or accepted)", ver)):
    w = sum(lab(s["final_keep"], s) == "wrong" for s in grp)
    print(f"  {name:<36} answers {len(grp):>3}, wrong {w}" + (f" ({w / len(grp):.1%})" if grp else ""))

print("\n3. Accepted retries by round")
acc = [s for s in states if s["status"] == "accepted"]
for K in sorted({s["accepted_K"] for s in acc}):
    grp = [s for s in acc if s["accepted_K"] == K]
    ok = sum(lab(s["final_keep"], s) == "correct" for s in grp)
    print(f"  K={K}: accepted {len(grp)}, of which correct {ok}")


def one_round_final(s):
    """Same loop with only the first retry round (K=5), computed from the saved log."""
    if s["status"] == "ok":
        return s["a1"]
    r5 = next((r for r in s["rounds"] if r["K"] == 5), None)
    if r5 and not r5["abstained"] and not r5.get("flagged", True):
        return r5["answer"]
    cands = []
    if not s["abst1"]:
        cands.append((s["a1"], s["sup1"]))
    if r5 and not r5["abstained"] and "support" in r5:
        cands.append((r5["answer"], r5["support"]))
    return max(cands, key=lambda x: (x[1], x[0] == s["a1"]))[0] if cands else ABSTAIN


print("\n4. One retry round only vs two rounds (keep policy, verified labels)")
for name, fn in (("two rounds (K=5, 10)", lambda s: s["final_keep"]), ("one round (K=5)", one_round_final)):
    c = Counter(lab(fn(s), s) for s in states)
    print(f"  {name:<22} correct {c['correct']}, wrong {c['wrong']}, abstain {c['abstain']}")

print("\n5. Accepted retries (up to 6)")
for s in acc[:6]:
    r = next(r for r in s["rounds"] if r["K"] == s["accepted_K"])
    print(f"\nQ: {s['question'][:110]}\n  gold: {s['gold']}\n  C answer: {s['a1']}"
          f"\n  retry K={r['K']}: {r['answer']} (support {r['support']:.2f}) -> {lab(s['final_keep'], s)}")

print("\n6. Wrong answers remaining after D (keep policy)")
for s in states:
    if lab(s["final_keep"], s) == "wrong":
        print(f"\nQ: {s['question'][:110]}\n  gold: {s['gold']}\n  final: {s['final_keep']}"
              f"\n  status: {s['status']}, flagged in C: {s['flag1']}")