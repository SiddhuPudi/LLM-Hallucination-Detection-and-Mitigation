import sys
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CHUNK_WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OVERLAP = CHUNK_WORDS // 5
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_words(text, size, overlap):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


# 1. Build chunks from every row's knowledge.
#    Identical chunk texts are merged; we remember which rows they came from.
df = pd.read_json("data/all.jsonl", lines=True)
text_to_ids = {}
for _, r in df.iterrows():
    for c in chunk_words(r["knowledge"], CHUNK_WORDS, OVERLAP):
        text_to_ids.setdefault(c, set()).add(int(r["id"]))

chunks = [
    {"chunk_id": i, "text": t, "source_ids": sorted(ids)}
    for i, (t, ids) in enumerate(text_to_ids.items())
]

with open(f"data/chunks_{CHUNK_WORDS}.jsonl", "w") as f:
    for c in chunks:
        f.write(json.dumps(c) + "\n")

avg_words = np.mean([len(c["text"].split()) for c in chunks])
print(f"rows={len(df)}  chunks={len(chunks)}  avg words/chunk={avg_words:.0f}")

# 2. Embed (cosine similarity = dot product on normalized vectors)
model = SentenceTransformer(EMB_MODEL)
emb = model.encode(
    [c["text"] for c in chunks],
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
).astype("float32")

np.save(f"data/emb_{CHUNK_WORDS}.npy", emb)
print("saved embeddings:", emb.shape)