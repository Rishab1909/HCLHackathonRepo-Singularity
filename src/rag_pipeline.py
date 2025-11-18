
#!/usr/bin/env python3
"""
rag_pipeline.py
End-to-end CLI: loads retriever + generator and answers queries interactively.
Saves logs to artifacts/query_logs.jsonl for auditing.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from retriever import Retriever
from generator import LLMGenerator

LOG_PATH = "artifacts/query_logs.jsonl"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="artifacts/faiss.index")
    p.add_argument("--meta", default="artifacts/chunks_meta.json")
    p.add_argument("--gguf", default="models/llama-3.2-3b-instruct.gguf")
    p.add_argument("--hf", default="models/llama-3.2-3b-instruct")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--log", default=LOG_PATH)
    args = p.parse_args()

    retriever = Retriever(index_path=args.index, meta_path=args.meta)
    generator = LLMGenerator(gguf_path=args.gguf, hf_folder=args.hf)

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    print("Mini RAG Assistant ready. Type 'exit' to quit.")
    while True:
        q = input("\nQuery> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        retrieved = retriever.retrieve(q, top_k=args.k)
        out = generator.answer_query(q, retrieved)
        print("\n=== ANSWER ===\n")
        print(out["answer"])
        print("\n=== SOURCES ===\n")
        for r in retrieved:
            print(f"chunk:{r['chunk_id']} score:{r['score']:.4f} source:{r['source']}")

        # log
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": q,
            "answer": out["answer"],
            "retrieved": [{"chunk_id": r["chunk_id"], "score": r["score"], "source": r["source"]} for r in retrieved],
            "prompt": out.get("prompt")
        }
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
