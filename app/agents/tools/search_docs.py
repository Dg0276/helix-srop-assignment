"""
search_docs tool — used by KnowledgeAgent.

Queries ChromaDB for relevant documentation chunks.
Returns a list of dicts (JSON-serializable for ADK tool protocol):
    [{"chunk_id": ..., "score": ..., "content": ..., "metadata": {...}}, ...]

The agent's instruction must tell it to cite chunk_ids in its answers.

E4: When rerank_enabled=True, results are reranked using an LLM-as-judge
    before being returned.
"""
import asyncio
from dataclasses import dataclass

import chromadb

from app.settings import settings

SCORE_THRESHOLD = 0.45  # discard chunks below this cosine similarity


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"},
    )


from google import genai

_genai_client = genai.Client(api_key=settings.google_api_key)

EMBEDDING_MODEL = "gemini-embedding-001"


def _embed_query_sync(query: str) -> list[float]:
    """Synchronous query embedding (called via asyncio.to_thread)."""
    result = _genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
    )
    return result.embeddings[0].values


async def search_docs(
    query: str,
    k: int = 5,
    product_area: str | None = None,
) -> list[dict]:
    """
    Search Helix documentation for the top-k most relevant chunks.

    Args:
        query: natural language query from the user
        k: number of chunks to return (default 5)
        product_area: optional metadata filter (e.g. "security", "ci-cd")

    Returns:
        List of dicts ordered by descending similarity score. Each dict has:
        chunk_id, score (0–1), content, metadata.
        Always cite chunk_id in your answer (e.g. "According to [chunk_abc123]...").
    """
    query_vec = await asyncio.to_thread(_embed_query_sync, query)

    where: dict | None = {"product_area": product_area} if product_area else None

    collection = _get_collection()
    # Fetch more results than needed for reranking (2x k)
    fetch_k = min(k * 2, collection.count() or 1) if settings.rerank_enabled else min(k, collection.count() or 1)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=fetch_k,
        where=where,
        include=["documents", "distances", "metadatas"],
    )

    chunks: list[DocChunk] = []
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for chunk_id, distance, doc, meta in zip(ids, distances, documents, metadatas):
        score = round(1.0 - float(distance), 4)
        if score >= SCORE_THRESHOLD:
            chunks.append(
                DocChunk(
                    chunk_id=chunk_id,
                    score=score,
                    content=doc,
                    metadata=meta or {},
                )
            )

    chunks.sort(key=lambda c: c.score, reverse=True)

    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "score": c.score,
            "content": c.content,
            "metadata": c.metadata,
        }
        for c in chunks
    ]

    # E4: LLM-as-judge reranking
    if settings.rerank_enabled and len(chunk_dicts) > 1:
        try:
            from app.rag.reranker import rerank
            chunk_dicts = await rerank(query, chunk_dicts, top_k=k)
        except Exception:
            # Graceful fallback to vector-only ranking
            chunk_dicts = chunk_dicts[:k]
    else:
        chunk_dicts = chunk_dicts[:k]

    return chunk_dicts
