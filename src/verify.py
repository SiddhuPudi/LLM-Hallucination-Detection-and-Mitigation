import re
import sys
import time
import pandas as pd
from nli import nli_scores

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
N = int(sys.argv[2]) if len(sys.argv) > 2 else None   # limit for a smoke test

claims = pd.read_json(f"data/claims_{SPLIT}.jsonl", lines=True)
if N:
    claims = claims.head(N)

base = pd.read_json(f"data/{SPLIT}.jsonl", lines=True).set_index("id")
ev_ret = pd.read_json(f"data/rag_b_k3_{SPLIT}.jsonl", lines=True).set_index("id")["evidence"].to_dict()

claims["ev_retrieved"] = claims["id"].map(lambda i: re.sub(r"\[\d+\]\s*", "", ev_ret[int(i)]))
claims["ev_oracle"] = claims["id"].map(lambda i: str(base.loc[int(i), "knowledge"]))

for name, col in (("ret", "ev_retrieved"), ("orc", "ev_oracle")):
    t0 = time.time()
    sc = nli_scores(claims[col].tolist(), claims["claim"].tolist())
    claims[f"entail_{name}"] = [s["entail"] for s in sc]
    claims[f"neutral_{name}"] = [s["neutral"] for s in sc]
    claims[f"contra_{name}"] = [s["contra"] for s in sc]
    print(f"{name}: {len(claims)} claims in {time.time() - t0:.0f}s")

keep = [c for c in claims.columns if c not in ("ev_retrieved", "ev_oracle")]
claims[keep].to_json(f"data/verified_{SPLIT}.jsonl", orient="records", lines=True)

print("\nMean P(entailment) by claim source:")
print(claims.groupby("source")[["entail_ret", "entail_orc"]].mean().round(3).to_string())
print(f"\nSaved data/verified_{SPLIT}.jsonl")