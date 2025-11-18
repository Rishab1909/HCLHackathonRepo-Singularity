#!/usr/bin/env python3
"""
build_index.py
Load artifacts/chunks_meta.json -> compute embeddings (sentence-transformers),
normalize embeddings (L2), build a FAISS IndexFlatIP and write artifacts/faiss.index

Usage:
    python src/build_index.py --chunks artifacts/chunks_meta.json --index artifacts/faiss.index
"""

import argparse
import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

def load_chunks(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def build_embeddings(texts, model_name=DEFAULT_MODEL, batch_size=BATCH_SIZE):
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embs.astype("float32")

def normalize_embeddings(embs):
    # L2 normalize rows
    faiss.normalize_L2(embs)
    return embs

def build_faiss_index(embs, index_path):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors = cosine
    index.add(embs)
    faiss.write_index(index, index_path)
    return index

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="artifacts/chunks_meta.json")
    p.add_argument("--index", default="artifacts/faiss.index")
    p.add_argument("--embedding_model", default=DEFAULT_MODEL)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.index), exist_ok=True)
    chunks = load_chunks(args.chunks)
    texts = [c["text"] for c in chunks]
    print(f"Computing embeddings for {len(texts)} chunks with {args.embedding_model}")
    embs = build_embeddings(texts, model_name=args.embedding_model)
    embs = normalize_embeddings(embs)
    print("Building FAISS index (IndexFlatIP)...")
    build_faiss_index(embs, args.index)
    # Save a small meta mapping too (if not already saved)
    print("Done. FAISS index saved to", args.index)

if __name__ == "__main__":
    main()