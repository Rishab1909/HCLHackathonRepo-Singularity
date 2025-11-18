"""
chunk_text.py
Read all txt and pdf files in data/, clean lightly, split into chunks with overlap,
and write artifacts/chunks_meta.json (list of chunk dicts).

Usage:
    python src/chunk_text.py --data_dir data --out artifacts/chunks_meta.json \
                             --chunk_words 450 --overlap_words 120

Dependencies:
    pip install PyPDF2
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import re
from typing import List, Dict
from PyPDF2 import PdfReader   

DEFAULT_CHUNK_WORDS = 450
DEFAULT_OVERLAP_WORDS = 50


# ---------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------
def extract_text_from_pdf(path: Path) -> str:
    """
    Extract text from a PDF file using PyPDF2 only.
    Returns text from all pages separated by double newlines.
    """
    text_pages = []
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            t = page.extract_text() or ""
            text_pages.append(t)
    except Exception as e:
        print(f"[WARN] Failed to read PDF {path.name}: {e}")
        return ""

    return "\n\n".join(text_pages).strip()


# ---------------------------------------------------------
# Load txt + pdf files
# ---------------------------------------------------------
def load_all_texts(data_dir: str) -> List[Dict[str, str]]:
    texts = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load all .txt
    for p in sorted(data_path.glob("*.txt")):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        texts.append({"source": p.name, "text": raw})

    # Load all .pdf
    for p in sorted(data_path.glob("*.pdf")):
        raw = extract_text_from_pdf(p)
        texts.append({"source": p.name, "text": raw})

    return texts


# ---------------------------------------------------------
# Cleaning + splitting + chunking
# ---------------------------------------------------------
def simple_clean(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        ln_stripped = ln.strip()

        if re.fullmatch(r"\d{1,4}", ln_stripped):
            continue

        if re.fullmatch(r"[-=]{3,}", ln_stripped):
            continue

        if len(ln_stripped) == 0:
            continue

        cleaned.append(ln)

    joined = "\n".join(cleaned)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def paragraph_split(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_paragraphs(paragraphs, chunk_words=DEFAULT_CHUNK_WORDS, overlap_words=DEFAULT_OVERLAP_WORDS):
    """
    Create chunks from paragraph list. This version splits paragraphs that are
    longer than chunk_words into multiple chunks (sliding-window style),
    respecting overlap_words.
    """
    chunks = []
    char_cursor = 0
    char_start = 0

    cur_words = []
    cur_word_count = 0

    for para in paragraphs:
        words = para.split()
        wlen = len(words)

        # If paragraph itself is longer than chunk size, process it in slices
        if wlen >= chunk_words:
            # First: flush any existing current chunk
            if cur_words:
                text = " ".join(cur_words)
                chunks.append({
                    "text": text,
                    "word_count": cur_word_count,
                    "char_start": max(0, char_start),
                    "char_end": char_cursor
                })
                # prepare overlap baseline for next slicing
                if overlap_words > 0:
                    kept = text.split()[-overlap_words:]
                else:
                    kept = []
                cur_words = list(kept)
                cur_word_count = len(cur_words)
                char_start = max(0, char_cursor - len(" ".join(kept)) - 1)

            # Now slice this long paragraph into chunks directly
            i = 0
            while i < wlen:
                slice_words = words[i:i + (chunk_words - cur_word_count)]
                # Append slice to current words (which may contain overlap kept above)
                cur_words += slice_words
                cur_word_count = len(cur_words)

                # If we've reached or exceeded chunk size, flush
                if cur_word_count >= chunk_words:
                    text = " ".join(cur_words)
                    chunks.append({
                        "text": text,
                        "word_count": cur_word_count,
                        "char_start": max(0, char_start),
                        "char_end": char_cursor
                    })

                    # build overlap for next chunk
                    if overlap_words > 0:
                        all_words = text.split()
                        keep = all_words[-overlap_words:] if overlap_words < len(all_words) else all_words
                        cur_words = list(keep)
                        cur_word_count = len(cur_words)
                        kept_text = " ".join(keep)
                        char_start = max(0, char_cursor - len(kept_text) - 1)
                    else:
                        cur_words = []
                        cur_word_count = 0
                        char_start = char_cursor

                # advance i by the number of words consumed in this iteration
                i += len(slice_words)

            # advance char cursor after processing the paragraph
            char_cursor += len(para) + 2
            continue  # move to next paragraph

        # Normal case: paragraph shorter than chunk size
        if cur_word_count + wlen > chunk_words and cur_words:
            # flush current chunk
            text = " ".join(cur_words)
            chunks.append({
                "text": text,
                "word_count": cur_word_count,
                "char_start": max(0, char_start),
                "char_end": char_cursor
            })

            # create overlap for next chunk
            if overlap_words > 0:
                all_words = text.split()
                keep = all_words[-overlap_words:] if overlap_words < len(all_words) else all_words
                cur_words = keep + words
                cur_word_count = len(cur_words)
                kept_text = " ".join(keep)
                char_start = max(0, char_cursor - len(kept_text) - 1)
            else:
                cur_words = words
                cur_word_count = wlen
                char_start = char_cursor
        else:
            cur_words += words
            cur_word_count += wlen

        char_cursor += len(para) + 2

    # flush final chunk
    if cur_words:
        chunks.append({
            "text": " ".join(cur_words),
            "word_count": cur_word_count,
            "char_start": char_start,
            "char_end": char_cursor
        })

    return chunks


def build_chunks_for_all(text_entries: List[Dict[str, str]],
                         chunk_words: int,
                         overlap_words: int):

    all_chunks = []
    chunk_id = 0

    for entry in text_entries:
        cleaned = simple_clean(entry["text"])
        paragraphs = paragraph_split(cleaned)
        chunks = chunk_paragraphs(paragraphs,
                                  chunk_words=chunk_words,
                                  overlap_words=overlap_words)

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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="artifacts/chunks_meta.json")
    p.add_argument("--chunk_words", type=int, default=DEFAULT_CHUNK_WORDS)
    p.add_argument("--overlap_words", type=int, default=DEFAULT_OVERLAP_WORDS)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    texts = load_all_texts(args.data_dir)
    print(f"Found {len(texts)} source files (txt/pdf)")

    all_chunks = build_chunks_for_all(texts, args.chunk_words, args.overlap_words)
    print(f"Created {len(all_chunks)} chunks")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()