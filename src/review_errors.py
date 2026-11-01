import pandas as pd

d = pd.read_json("data/baseline_dev.jsonl", lines=True)
bad = d[d.contains == 0]
print(f"{len(bad)} of {len(d)} dev answers do not contain the gold answer\n")

for _, r in bad.head(20).iterrows():
    print(f"Q:    {r.question}\n  gold: {r.gold}\n  pred: {r.pred}\n")