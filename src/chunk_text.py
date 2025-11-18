"""
chunk_text.py
Read all txt files in data/, clean lightly, split into chunks with overlap,
and write artifacts/chunks_meta.json (list of chunk dicts).

Usage:
    python src/chunk_text.py --data_dir data --out artifacts/chunks_meta.json \
                             --chunk_words 450 --overlap_words 120
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import re

DEFAULT_CHUNK_WORDS = 450
DEFAULT_OVERLAP_WORDS = 120

def load_all_texts(data_dir):
    texts = []
    for p in sorted(Path(data_dir).glob("*.txt")):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        texts.append({"source": p.name, "text": raw})
    return texts

def simple_clean(text):
    # Remove repeated header/footer lines (simple heuristic), normalize whitespace
    # Remove lines with only digits (possible page numbers)
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        ln_stripped = ln.strip()
        # skip bare page numbers
        if re.fullmatch(r"\d{1,4}", ln_stripped):
            continue
        # skip lines that are long repeated hyphens or digits
        if re.fullmatch(r"[-=]{3,}", ln_stripped):
            continue
        cleaned.append(ln)
    joined = "\n".join(cleaned)
    # collapse multiple newlines
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()

def paragraph_split(text):
    # split by blank lines
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras

def chunk_paragraphs(paragraphs, chunk_words=DEFAULT_CHUNK_WORDS, overlap_words=DEFAULT_OVERLAP_WORDS):
    chunks = []
    cur = []
    cur_word_count = 0
    cur_sources = set()
    char_start = 0
    # We'll track approximate char offsets (best-effort)
    char_cursor = 0
    for para in paragraphs:
        words = para.split()
        wlen = len(words)
        if cur_word_count + wlen > chunk_words and cur:
            text = " ".join(cur)
            chunks.append({"text": text, "word_count": cur_word_count, "char_start": max(0, char_start), "char_end": char_cursor})
            # create overlap
            if overlap_words > 0:
                # keep last overlap_words from cur
                all_words = " ".join(cur).split()
                keep = all_words[-overlap_words:] if overlap_words < len(all_words) else all_words
                cur = keep + words
                cur_word_count = len(cur)
                # char_start approximate (not exact)
                char_start = char_cursor - len(" ".join(keep)) - 1
            else:
                cur = words
                cur_word_count = len(words)
                char_start = char_cursor
        else:
            cur += words
            cur_word_count += wlen
        char_cursor += len(para) + 2  # approximate position with paragraph length
    if cur:
        chunks.append({"text": " ".join(cur), "word_count": cur_word_count, "char_start": char_start, "char_end": char_cursor})
    return chunks

def build_chunks_for_all(text_entries, chunk_words, overlap_words):
    all_chunks = []
    chunk_id = 0
    for entry in text_entries:
        cleaned = simple_clean(entry["text"])
        paragraphs = paragraph_split(cleaned)
        chunks = chunk_paragraphs(paragraphs, chunk_words=chunk_words, overlap_words=overlap_words)
        for c in chunks:
            meta = {
                "chunk_id": chunk_id,
                "source": entry["source"],
                "text": c["text"],
                "word_count": c["word_count"],
                "char_start": c["char_start"],
                "char_end": c["char_end"]
            }
            all_chunks.append(meta)
            chunk_id += 1
    return all_chunks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="artifacts/chunks_meta.json")
    p.add_argument("--chunk_words", type=int, default=DEFAULT_CHUNK_WORDS)
    p.add_argument("--overlap_words", type=int, default=DEFAULT_OVERLAP_WORDS)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    texts = load_all_texts(args.data_dir)
    print(f"Found {len(texts)} txt files")
    all_chunks = build_chunks_for_all(texts, args.chunk_words, args.overlap_words)
    print(f"Created {len(all_chunks)} chunks")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()