"""
E4 — LLM-as-judge reranker.

After initial vector search retrieves top-k chunks, this module sends each
chunk + the query to Gemini and asks it to score relevance (0.0–1.0).
The results are then re-sorted by the LLM relevance score.

This improves answer quality by pushing truly relevant chunks above
superficially similar but less useful ones.
"""
import asyncio
import json

import structlog

from app.settings import settings

log = structlog.get_logger()


async def rerank(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank search results using Gemini as a relevance judge.

    Args:
        query: the user's search query
        chunks: list of dicts with chunk_id, score, content, metadata
        top_k: max number of results to return after reranking

    Returns:
        Reranked list of chunk dicts with an additional 'rerank_score' field.
        Original 'score' is preserved for comparison.
    """
    if not chunks or not settings.rerank_enabled:
        return chunks[:top_k]

    try:
        scored = await asyncio.to_thread(_rerank_sync, query, chunks)
        scored.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        return scored[:top_k]
    except Exception as exc:
        log.warning("rerank_failed_falling_back", error=str(exc))
        # Graceful degradation: return original ranking
        return chunks[:top_k]


def _rerank_sync(query: str, chunks: list[dict]) -> list[dict]:
    """Synchronous reranking via Gemini (called via asyncio.to_thread)."""
    from google import genai

    client = genai.Client(api_key=settings.google_api_key)

    # Build the prompt with all chunks for batch scoring
    chunk_descriptions = []
    for i, chunk in enumerate(chunks):
        chunk_descriptions.append(
            f"[Chunk {i}] (id: {chunk['chunk_id']})\n{chunk['content'][:300]}"
        )

    prompt = f"""You are a relevance judge. Given a user query and a set of document chunks,
score each chunk's relevance to the query on a scale of 0.0 to 1.0.

Query: "{query}"

Chunks:
{chr(10).join(chunk_descriptions)}

Respond with ONLY a JSON array of objects, one per chunk, in the same order:
[{{"index": 0, "score": 0.85}}, {{"index": 1, "score": 0.3}}, ...]

Rules:
- 1.0 = perfectly answers the query
- 0.0 = completely irrelevant
- Consider semantic relevance, not just keyword overlap
- Be strict: most chunks should score below 0.7 unless truly relevant
"""

    response = client.models.generate_content(
        model=settings.adk_model,
        contents=prompt,
    )

    response_text = response.text.strip()
    # Extract JSON from response (handle markdown code blocks)
    if "```" in response_text:
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    scores = json.loads(response_text)

    result = []
    for chunk in chunks:
        idx = chunks.index(chunk)
        rerank_score = 0.0
        for s in scores:
            if s.get("index") == idx:
                rerank_score = float(s.get("score", 0.0))
                break
        result.append({
            **chunk,
            "original_score": chunk["score"],
            "rerank_score": rerank_score,
            "score": rerank_score,  # override score with reranked value
        })

    return result
