"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 512 --chunk-overlap 64

Chunking strategy: heading-aware (split on ## / ###) because Helix docs are structured
Markdown — keeps sections coherent. Long sections are sub-chunked with sentence splitting
to keep sizes bounded. Overlap preserves context at sentence boundaries.

Stable chunk IDs: sha256(file_path::chunk_index)[:16] — deterministic, so re-ingest
is idempotent (ChromaDB upsert deduplicates by ID).
"""
import argparse
import asyncio
import hashlib
import re
from pathlib import Path

import chromadb
import yaml

from app.settings import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter block. Returns (metadata_dict, body_text)."""
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {}, text
    try:
        metadata: dict = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        metadata = {}
    body = text[match.end():]
    return metadata, body


def extract_metadata(file_path: Path, text: str) -> tuple[dict, str]:
    """
    Extract metadata from a markdown file's frontmatter.

    Returns (metadata_dict, body_without_frontmatter).
    Falls back to filename-derived metadata if no frontmatter found.
    """
    meta, body = extract_frontmatter(text)
    meta.setdefault("source", file_path.name)
    meta.setdefault("title", file_path.stem.replace("-", " ").title())
    return meta, body


def _chunk_sentences(text: str, max_chars: int, overlap_sentences: int = 1) -> list[str]:
    """Split text on sentence boundaries, keeping chunks <= max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:]
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_markdown(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split markdown text into chunks using heading-aware strategy.

    Splits on ## and ### headings first (natural section boundaries).
    Sub-chunks any section longer than chunk_size using sentence splitting.
    Filters empty strings.
    """
    sections = re.split(r"\n(?=#{2,3} )", text)
    chunks: list[str] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            sub = _chunk_sentences(section, max_chars=chunk_size, overlap_sentences=1)
            chunks.extend(sub)

    return [c for c in chunks if c.strip()]


def make_chunk_id(file_path: str, chunk_index: int) -> str:
    """Generate a stable, deterministic chunk ID."""
    raw = f"{file_path}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

from google import genai

_genai_client = genai.Client(api_key=settings.google_api_key)

EMBEDDING_MODEL = "gemini-embedding-001"


def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    """Synchronous embedding call (run via asyncio.to_thread at call sites)."""
    result = _genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    return [e.values for e in result.embeddings]


async def embed_texts(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """Embed a list of texts in batches, returning embeddings in the same order."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        
        # Retry loop for Free Tier rate limits (429)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                batch_embeddings = await asyncio.to_thread(_embed_batch, batch, "retrieval_document")
                all_embeddings.extend(batch_embeddings)
                await asyncio.sleep(2.0)  # Pacing: avoid hitting limits on next batch
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"    Rate limit hit. Waiting 20 seconds (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(20.0)
                else:
                    raise
    return all_embeddings


def get_or_create_collection() -> chromadb.Collection:
    """Return (or create) the helix_docs ChromaDB collection."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Main ingest logic
# ---------------------------------------------------------------------------

async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """
    Walk docs_path, chunk and embed every .md file, upsert into vector store.

    Uses deterministic chunk IDs so re-ingest is safe (no duplicates).
    """
    md_files = list(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    collection = get_or_create_collection()
    total_chunks = 0

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        meta, body = extract_metadata(file_path, text)
        chunks = chunk_markdown(body, chunk_size, chunk_overlap)

        if not chunks:
            print(f"  {file_path.name}: 0 chunks — skipping")
            continue

        ids = [make_chunk_id(str(file_path), i) for i in range(len(chunks))]
        metadatas = [
            {
                "source": meta.get("source", file_path.name),
                "title": meta.get("title", file_path.stem),
                "product_area": meta.get("product_area", ""),
            }
            for _ in chunks
        ]

        print(f"  {file_path.name}: {len(chunks)} chunks — embedding …")
        embeddings = await embed_texts(chunks)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f"  {file_path.name}: upserted {len(chunks)} chunks ✓")

    print(f"\nIngest complete. Total chunks upserted: {total_chunks}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
