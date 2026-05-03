"""
KnowledgeAgent — answers product/documentation questions via RAG.

Uses the search_docs tool to retrieve relevant chunks from the vector store,
then answers citing chunk IDs. Never answers from its own training data.
"""
from google.adk.agents import LlmAgent

from app.agents.tools.search_docs import search_docs
from app.settings import settings

KNOWLEDGE_INSTRUCTION = """\
You are the Helix Knowledge Agent — a documentation specialist.

When the user asks a product or feature question:
1. Call the search_docs tool with an appropriate query.
2. Read the returned chunks carefully.
3. Answer ONLY using information from those chunks.
4. Always cite the chunk_id in your answer, e.g. "According to [chunk_abc123], …"
5. If no chunks are relevant or returned, say: "I don't have documentation on that topic."

Do NOT guess or use training data. Only use retrieved context.
"""

knowledge_agent = LlmAgent(
    name="knowledge_agent",
    model=settings.adk_model,
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[search_docs],
)
