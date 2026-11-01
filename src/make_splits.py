import os
import urllib.request
import pandas as pd

URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
RAW = "data/qa_data.json"
REQUIRED = ["knowledge", "question", "right_answer", "hallucinated_answer"]

os.makedirs("data", exist_ok=True)

# 1. Download once
if not os.path.exists(RAW):
    print("Downloading HaluEval QA...")
    urllib.request.urlretrieve(URL, RAW)

# 2. Load (the file is JSON lines: one JSON object per line)
df = pd.read_json(RAW, lines=True)
print("Raw shape:", df.shape)
assert all(c in df.columns for c in REQUIRED), f"Unexpected columns: {df.columns.tolist()}"

# 3. Clean
df = df.dropna(subset=REQUIRED)
for c in REQUIRED:
    df[c] = df[c].astype(str).str.strip()
df = df[(df[REQUIRED] != "").all(axis=1)]
df = df.drop_duplicates(subset="question")
print("After cleaning:", df.shape)

# 4. Shuffle with a fixed seed, then give every row a permanent id
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "id", df.index)

# 5. Split
dev = df.iloc[:100]
val = df.iloc[100:400]
test = df.iloc[400:900]

# 6. Sanity checks: no overlap between splits
assert not (set(dev.id) & set(val.id))
assert not (set(dev.id) & set(test.id))
assert not (set(val.id) & set(test.id))

# 7. Save
df.to_json("data/all.jsonl", orient="records", lines=True)
dev.to_json("data/dev.jsonl", orient="records", lines=True)
val.to_json("data/val.jsonl", orient="records", lines=True)
test.to_json("data/test.jsonl", orient="records", lines=True)

print(f"all={len(df)}  dev={len(dev)}  val={len(val)}  test={len(test)}")
print("\nSample row:")
print(dev.iloc[0].to_dict())