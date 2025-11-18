
#!/usr/bin/env python3
"""
retriever.py
Simple wrapper to load FAISS index + meta and perform retrieval.

Usage (example):
    from retriever import Retriever
    r = Retriever(index_path="artifacts/faiss.index", meta_path="artifacts/chunks_meta.json")
    results = r.retrieve("what is gradient descent?", top_k=6)
"""

import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path="artifacts/faiss.index", meta_path="artifacts/chunks_meta.json",
                 embed_model="all-MiniLM-L6-v2"):
        self.index = faiss.read_index(index_path)
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.embedder = SentenceTransformer(embed_model)

    def embed_query(self, query):
        q = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        return q

    def retrieve(self, query, top_k=6):
        q = self.embed_query(query)
        distances, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.meta[idx]
            results.append({
                "chunk_id": meta["chunk_id"],
                "source": meta["source"],
                "score": float(score),
                "text": meta["text"],
                "word_count": meta.get("word_count", None)
            })
        return results

if __name__ == "__main__":
    # quick CLI
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="artifacts/faiss.index")
    p.add_argument("--meta", default="artifacts/chunks_meta.json")
    p.add_argument("--k", type=int, default=6)
    args = p.parse_args()
    r = Retriever(index_path=args.index, meta_path=args.meta)
    while True:
        q = input("\nQuery> ").strip()
        if not q:
            continue
        out = r.retrieve(q, top_k=args.k)
        for i, rres in enumerate(out):
            print(f"\n[{i}] chunk:{rres['chunk_id']} score:{rres['score']:.4f} source:{rres['source']}\n{rres['text'][:500]}...\n")
