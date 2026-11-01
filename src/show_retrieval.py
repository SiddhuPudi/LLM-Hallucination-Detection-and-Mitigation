import pandas as pd
from retrieve import Retriever

df = pd.read_json("data/dev.jsonl", lines=True).head(8)
ret = Retriever(120)

for _, row in df.iterrows():
    print("=" * 80)
    print("Q:", row["question"])
    print("GOLD ANSWER:", row["right_answer"])
    for rank, c in enumerate(ret.search(row["question"], k=3), 1):
        mine = "  <-- its own row" if int(row["id"]) in c["source_ids"] else ""
        print(f"\n  [{rank}] score={c['score']:.3f}{mine}\n  {c['text'][:300]}")