import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import numpy as np
from sentence_transformers import SentenceTransformer

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Exact numpy search is the default. Set USE_FAISS=1 to try FAISS instead.
USE_FAISS = os.getenv("USE_FAISS", "0") == "1"
faiss = None
if USE_FAISS:
    import faiss
    faiss.omp_set_num_threads(1)


class Retriever:
    def __init__(self, chunk_words=120):
        with open(f"data/chunks_{chunk_words}.jsonl") as f:
            self.chunks = [json.loads(line) for line in f]
        self.emb = np.load(f"data/emb_{chunk_words}.npy")
        self.model = SentenceTransformer(EMB_MODEL)

        self.index = None
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb)
        print("search backend:", "FAISS" if self.index is not None else "numpy (exact cosine)")

    def search_many(self, queries, k=3):
        q = self.model.encode(
            queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        if self.index is not None:
            scores, idx = self.index.search(q, k)
        else:
            sims = q @ self.emb.T
            idx = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, idx, axis=1)

        return [
            [{**self.chunks[i], "score": float(s)} for s, i in zip(srow, irow)]
            for srow, irow in zip(scores, idx)
        ]

    def search(self, query, k=3):
        return self.search_many([query], k)[0]