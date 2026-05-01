"""
LangChain RAG over SRT files — find timestamps matching a user query.

Usage:
    pip install langchain langchain-openai langchain-community faiss-cpu sentence-transformers

    # Interactive mode (prompts for queries until you quit)
    python srt_rag.py --srt video.srt

    # Single query
    python srt_rag.py --srt video.srt --query "when does the speaker mention budgets"

    # Use local embeddings (no API key needed)
    python srt_rag.py --srt video.srt --embedder local

    # Return more results
    python srt_rag.py --srt video.srt --query "introduction" --top-k 5
"""

import argparse
import os
import re
import sys
from pathlib import Path


# ── SRT parsing ────────────────────────────────────────────────────────────────

def parse_srt(path: Path) -> list[dict]:
    """
    Parse an SRT file into a list of blocks:
      {index, start, end, start_seconds, end_seconds, text}
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    # Split on blank lines that separate blocks
    raw_blocks = re.split(r"\n\s*\n", text.strip())

    blocks = []
    time_re = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )

    for raw in raw_blocks:
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        # First line: sequence number
        # Second line: timestamp range
        # Rest: subtitle text
        m = time_re.search(lines[1])
        if not m:
            continue

        def to_seconds(h, mn, s, ms):
            return int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000

        start_s = to_seconds(*m.group(1, 2, 3, 4))
        end_s   = to_seconds(*m.group(5, 6, 7, 8))
        subtitle_text = " ".join(lines[2:])

        blocks.append({
            "index":         lines[0],
            "start":         m.group(0).split("-->")[0].strip(),
            "end":           m.group(0).split("-->")[1].strip(),
            "start_seconds": start_s,
            "end_seconds":   end_s,
            "text":          subtitle_text,
        })

    return blocks


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# ── Embedder selection ─────────────────────────────────────────────────────────

def get_embedder(mode: str):
    if mode == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()

    # local: use sentence-transformers via langchain-community
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def auto_embedder():
    """Use OpenAI if key is set, otherwise fall back to local."""
    if os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY found — using OpenAI embeddings.")
        return get_embedder("openai")
    print("No OPENAI_API_KEY — using local sentence-transformers (all-MiniLM-L6-v2).")
    return get_embedder("local")


# ── Index building ─────────────────────────────────────────────────────────────

def build_index(blocks: list[dict], embedder):
    from langchain.schema import Document
    from langchain_community.vectorstores import FAISS

    docs = [
        Document(
            page_content=b["text"],
            metadata={
                "index":         b["index"],
                "start":         b["start"],
                "end":           b["end"],
                "start_seconds": b["start_seconds"],
            },
        )
        for b in blocks
        if b["text"].strip()
    ]

    if not docs:
        sys.exit("SRT file contains no subtitle text.")

    print(f"Indexing {len(docs)} subtitle block(s)...")
    store = FAISS.from_documents(docs, embedder)
    print("Index ready.\n")
    return store


# ── Query ──────────────────────────────────────────────────────────────────────

def query_index(store, query: str, top_k: int) -> list[tuple]:
    """Returns list of (score, Document) sorted by relevance."""
    results = store.similarity_search_with_score(query, k=top_k)
    # Lower L2 distance = more similar; sort ascending
    results.sort(key=lambda x: x[1])
    return results


def display_results(results: list[tuple], query: str):
    print(f"\nQuery: \"{query}\"")
    print("─" * 60)
    if not results:
        print("No results found.")
        return
    for doc, score in results:
        m = doc.metadata
        print(f"[{m['start']}]  (block #{m['index']}, score={score:.3f})")
        print(f"  {doc.page_content}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG search over an SRT file — returns timestamps for matching content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--srt", required=True, help="Path to the .srt file")
    parser.add_argument("--query", "-q", default=None,
                        help="Query string. Omit to enter interactive mode.")
    parser.add_argument("--top-k", "-k", type=int, default=3,
                        help="Number of results to return (default: 3)")
    parser.add_argument(
        "--embedder", choices=["auto", "openai", "local"], default="auto",
        help="Embedding backend: auto (default), openai, or local sentence-transformers",
    )

    args = parser.parse_args()

    srt_path = Path(args.srt)
    if not srt_path.exists():
        sys.exit(f"File not found: {srt_path}")

    # Parse
    blocks = parse_srt(srt_path)
    if not blocks:
        sys.exit("No subtitle blocks found in the SRT file.")
    print(f"Parsed {len(blocks)} block(s) from {srt_path.name}")

    # Embed
    if args.embedder == "auto":
        embedder = auto_embedder()
    else:
        embedder = get_embedder(args.embedder)

    store = build_index(blocks, embedder)

    # Query
    if args.query:
        results = query_index(store, args.query, args.top_k)
        display_results(results, args.query)
    else:
        print("Interactive mode — type a query and press Enter. Ctrl+C or empty input to quit.\n")
        while True:
            try:
                q = input("Query> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not q:
                break
            results = query_index(store, q, args.top_k)
            display_results(results, q)

    print("Done.")


if __name__ == "__main__":
    main()
