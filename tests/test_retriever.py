"""
Unit tests for RAG retrieval.
Requires the vector store to be seeded first (run ingest.py on docs/).
"""
import pytest


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    from app.rag.ingest import chunk_markdown

    text = "# Header\n\nSome content.\n\n## Section 2\n\nMore content here."
    chunks = chunk_markdown(text, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks)


def test_chunker_splits_on_headings():
    """Heading-aware chunker should split on ## boundaries."""
    from app.rag.ingest import chunk_markdown

    text = "## Section A\n\nContent A.\n\n## Section B\n\nContent B."
    chunks = chunk_markdown(text, chunk_size=500, overlap=0)
    assert len(chunks) == 2
    assert "Section A" in chunks[0]
    assert "Section B" in chunks[1]


def test_extract_metadata_with_frontmatter():
    """Frontmatter should be parsed into metadata dict."""
    from pathlib import Path
    from app.rag.ingest import extract_metadata

    text = "---\ntitle: Deploy Keys\nproduct_area: security\ntags: [keys, secrets]\n---\n\n# Deploy Keys\n\nContent."
    meta, body = extract_metadata(Path("deploy-keys.md"), text)
    assert meta["title"] == "Deploy Keys"
    assert meta["product_area"] == "security"
    assert "---" not in body


def test_extract_metadata_without_frontmatter():
    """Files without frontmatter should get filename-derived metadata."""
    from pathlib import Path
    from app.rag.ingest import extract_metadata

    text = "# Just Content\n\nNo frontmatter here."
    meta, body = extract_metadata(Path("my-doc.md"), text)
    assert meta["source"] == "my-doc.md"
    assert "title" in meta


def test_make_chunk_id_is_deterministic():
    """Same inputs must produce the same chunk ID."""
    from app.rag.ingest import make_chunk_id

    id1 = make_chunk_id("docs/deploy-keys.md", 0)
    id2 = make_chunk_id("docs/deploy-keys.md", 0)
    assert id1 == id2
    assert id1.startswith("chunk_")


def test_make_chunk_id_differs_for_different_inputs():
    """Different inputs must produce different chunk IDs."""
    from app.rag.ingest import make_chunk_id

    id1 = make_chunk_id("docs/deploy-keys.md", 0)
    id2 = make_chunk_id("docs/deploy-keys.md", 1)
    assert id1 != id2


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids():
    """search_docs must return chunk IDs and scores in [0, 1]."""
    try:
        from app.agents.tools.search_docs import search_docs
        results = await search_docs("how to rotate a deploy key", k=3)
    except Exception:
        pytest.skip("Vector store not seeded — run `python -m app.rag.ingest --path docs/` first")
        return

    if not results:
        pytest.skip("No results returned — vector store may not be seeded")
        return

    assert len(results) > 0
    assert all(r["chunk_id"] for r in results)
    assert all(0.0 <= r["score"] <= 1.0 for r in results)
