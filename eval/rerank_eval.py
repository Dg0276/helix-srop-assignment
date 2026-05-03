"""
E4 — Reranking evaluation script.

Runs 5 sample queries against the vector store with and without reranking,
showing before/after comparison.

Usage:
    python eval/rerank_eval.py
"""
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


EVAL_QUERIES = [
    "How do I rotate a deploy key?",
    "What are the billing plans available?",
    "How to configure CI/CD runners?",
    "What is secret scanning?",
    "How does API authentication work in Helix?",
]


async def run_comparison() -> None:
    """Run queries with and without reranking and compare."""
    from app.settings import settings

    # Import after path setup
    from app.agents.tools.search_docs import search_docs
    from app.rag.reranker import rerank

    print("E4 — Reranking Before/After Comparison")
    print("=" * 80)

    for query in EVAL_QUERIES:
        print(f"\nQuery: \"{query}\"")
        print("-" * 60)

        # Without reranking
        settings.rerank_enabled = False
        original_results = await search_docs(query, k=5)

        # With reranking
        settings.rerank_enabled = True
        reranked_results = await search_docs(query, k=5)

        print(f"\n  {'Rank':<6} {'Before (vector score)':<40} {'After (reranked score)':<40}")
        print(f"  {'─'*6} {'─'*40} {'─'*40}")

        max_len = max(len(original_results), len(reranked_results))
        for i in range(min(max_len, 5)):
            before = ""
            after = ""
            if i < len(original_results):
                r = original_results[i]
                before = f"{r['chunk_id'][:20]:<20} ({r['score']:.4f})"
            if i < len(reranked_results):
                r = reranked_results[i]
                orig_score = r.get("original_score", r["score"])
                new_score = r.get("rerank_score", r["score"])
                after = f"{r['chunk_id'][:20]:<20} ({new_score:.4f})"
            print(f"  {i+1:<6} {before:<40} {after:<40}")

    print("\n" + "=" * 80)
    print("Done. Reranking uses Gemini as a relevance judge to improve result ordering.")


def main() -> None:
    asyncio.run(run_comparison())


if __name__ == "__main__":
    main()
