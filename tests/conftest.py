"""
Test fixtures.

Key fixtures:
- `client`: async test client with in-memory SQLite DB
- `mock_adk`: patches the ADK pipeline so tests don't hit the real LLM
- `seeded_db`: DB with a test user and session pre-created
"""
import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import AgentTrace, Base, Session, Ticket
from app.db.session import get_db
from app.main import app
from app.srop.pipeline import PipelineResult
from app.srop.state import SessionState


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db):
    """Async test client with DB overridden to in-memory SQLite."""

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch):
    """
    Patch the ADK pipeline so tests don't call the real LLM.

    Routes based on message content:
    - "rotate" or "deploy key" → knowledge route with chunk IDs
    - "plan" or "tier" → account route reading plan_tier from state
    - "build" → account route
    - "escalate" or "ticket" or "bug" → escalation route (E2)
    - "poem" or "write me" → root (guardrails refusal, E5)
    - fallback → root (smalltalk)
    """

    async def mock_run(session_id: str, message: str, db: AsyncSession) -> PipelineResult:
        trace_id = f"test-trace-{uuid.uuid4().hex[:8]}"

        # Check session exists (mirrors real pipeline behavior)
        result = await db.execute(
            select(Session).where(Session.session_id == session_id)
        )
        session_row = result.scalar_one_or_none()
        if session_row is None:
            from app.api.errors import SessionNotFoundError
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        # Load state for context-aware responses
        state = SessionState.from_db_dict(session_row.state)

        # Determine routing based on message content
        msg_lower = message.lower()
        if "rotate" in msg_lower or "deploy key" in msg_lower:
            routed_to = "knowledge"
            content = "To rotate a deploy key, go to Settings > Deploy Keys. [chunk_abc123]"
            chunk_ids = ["chunk_abc123", "chunk_def456"]
        elif "plan" in msg_lower or "tier" in msg_lower:
            # Read state from DB to prove persistence works
            tier = state.plan_tier
            routed_to = "account"
            content = f"Your current plan tier is {tier}."
            chunk_ids = []
        elif "build" in msg_lower:
            routed_to = "account"
            content = "Here are your recent builds: b_001 (passed), b_002 (failed)."
            chunk_ids = []
        elif "escalate" in msg_lower or "ticket" in msg_lower or "bug" in msg_lower:
            # E2: Escalation agent
            ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
            routed_to = "escalation"
            content = f"I've created support ticket {ticket_id} for your issue. Our team will follow up."
            chunk_ids = []
            # Persist ticket to DB
            db.add(Ticket(
                ticket_id=ticket_id,
                user_id=state.user_id,
                summary=message,
                priority="medium",
                status="open",
            ))
            # Update state with ticket ID
            state.last_ticket_id = ticket_id
            session_row.state = state.to_db_dict()
        elif "poem" in msg_lower or "write me" in msg_lower:
            # E5: Guardrails refusal
            routed_to = "root"
            content = "I can only help with Helix product and account questions."
            chunk_ids = []
        else:
            routed_to = "root"
            content = "Hello! How can I help you with Helix today?"
            chunk_ids = []

        # Write trace row so GET /v1/traces/{trace_id} works
        db.add(AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=routed_to,
            tool_calls=[{"tool_name": "mock_tool", "args": {}, "result": "mock"}],
            retrieved_chunk_ids=chunk_ids,
            latency_ms=42,
        ))
        await db.commit()

        return PipelineResult(content=content, routed_to=routed_to, trace_id=trace_id)

    monkeypatch.setattr("app.srop.pipeline.run", mock_run)
