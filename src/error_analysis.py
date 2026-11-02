import os
import pandas as pd

A = pd.read_json("data/baseline_dev.jsonl", lines=True)
B = pd.read_json("data/rag_b_k3_dev.jsonl", lines=True)
C = pd.read_json("data/rag_c_k3_dev.jsonl", lines=True)

KEEP = ["pred", "contains", "abstained", "evidence_has_answer", "own_row_retrieved", "f1"]


def prep(df, s):
    d = df[["id"] + KEEP].copy()
    for c in ["abstained", "evidence_has_answer", "own_row_retrieved"]:
        d[c] = d[c].astype(bool)
    d.columns = ["id"] + [f"{c}_{s}" for c in KEEP]
    return d


m = A[["id", "question", "gold"]].merge(
    A[["id", "pred", "contains"]].rename(columns={"pred": "pred_A", "contains": "contains_A"}),
    on="id",
)
m = m.merge(prep(B, "B"), on="id").merge(prep(C, "C"), on="id")
m = m.merge(C[["id", "evidence"]], on="id")


def category(r, s):
    if r[f"contains_{s}"] == 1:
        return "correct"
    if r[f"abstained_{s}"]:
        return "abstain_over_cautious" if r[f"evidence_has_answer_{s}"] else "abstain_appropriate"
    if r[f"evidence_has_answer_{s}"]:
        return "wrong_despite_evidence"
    return "wrong_evidence_lacks_answer"


for s in ["B", "C"]:
    m[f"cat_{s}"] = m.apply(lambda r: category(r, s), axis=1)

ORDER = ["correct", "abstain_appropriate", "abstain_over_cautious",
         "wrong_despite_evidence", "wrong_evidence_lacks_answer"]

print("=" * 80)
print(f"OUTCOMES on dev ({len(m)} questions)")
counts = pd.DataFrame({s: m[f"cat_{s}"].value_counts().reindex(ORDER, fill_value=0) for s in ["B", "C"]})
print(counts.to_string())

# ---- movement between configurations ----
wrong_ans_B = (m.contains_B == 0) & (~m.abstained_B)
wrong_ans_C = (m.contains_C == 0) & (~m.abstained_C)
moves = {
    "Baseline right -> B wrong": ((m.contains_A == 1) & (m.contains_B == 0)).sum(),
    "Baseline wrong -> B right": ((m.contains_A == 0) & (m.contains_B == 1)).sum(),
    "B right -> C abstained (cost of grounded prompt)": ((m.contains_B == 1) & m.abstained_C).sum(),
    "B wrong answer -> C abstained (fabrication avoided)": (wrong_ans_B & m.abstained_C).sum(),
    "B wrong answer -> C right": (wrong_ans_B & (m.contains_C == 1)).sum(),
    "B right -> C wrong answer": ((m.contains_B == 1) & wrong_ans_C).sum(),
}
print("\n" + "=" * 80)
print("MOVEMENT BETWEEN CONFIGS (number of questions)")
for k, v in moves.items():
    print(f"  {k:<55} {v}")

# ---- outcome by whether the retrieved evidence contained the gold answer ----
flag = m["evidence_has_answer_B"].map({True: "evidence has answer", False: "evidence lacks answer"})
for s in ["B", "C"]:
    print("\n" + "=" * 80)
    print(f"Config {s}: outcome by whether the evidence contained the gold answer")
    print(pd.crosstab(flag, m[f"cat_{s}"]).to_string())

# ---- possible phrasing differences among wrong answers ----
for s in ["B", "C"]:
    n = ((m[f"f1_{s}"] >= 0.4) & m[f"cat_{s}"].str.startswith("wrong")).sum()
    print(f"\nConfig {s}: wrong answers with F1 >= 0.4 (possible phrasing differences): {n}")


def show(title, mask, n=4):
    sub = m[mask].head(n)
    print("\n" + "=" * 80)
    print(f"{title}  ({int(mask.sum())} total, showing {len(sub)})")
    for _, r in sub.iterrows():
        print(f"\nQ: {r.question}")
        print(f"  gold:     {r.gold}")
        print(f"  baseline: {r.pred_A}")
        print(f"  B:        {r.pred_B}")
        print(f"  C:        {r.pred_C}")
        print(f"  evidence: {r.evidence[:350]}")


show("C: WRONG DESPITE EVIDENCE", m.cat_C == "wrong_despite_evidence")
show("C: ABSTAINED EVEN THOUGH EVIDENCE HAD THE ANSWER", m.cat_C == "abstain_over_cautious")
show("C: WRONG AND EVIDENCE LACKED THE ANSWER", m.cat_C == "wrong_evidence_lacks_answer", n=3)
show("BASELINE RIGHT BUT B WRONG", (m.contains_A == 1) & (m.contains_B == 0), n=3)

os.makedirs("results", exist_ok=True)
m.to_csv("results/error_analysis_dev.csv", index=False)
print("\nFull table saved to results/error_analysis_dev.csv")