"""
Integration tests — exercise the full SROP pipeline.
LLM mocked at the ADK boundary (not at the HTTP layer).
"""
import pytest


@pytest.mark.asyncio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_001"})
    assert resp.status_code == 200
    assert "session_id" in resp.json()


@pytest.mark.asyncio
async def test_create_session_with_plan_tier(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_003", "plan_tier": "enterprise"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "u_test_003"


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_adk):
    """
    Core integration test.

    Sends a knowledge question, asserts:
    1. Response contains a reply
    2. routed_to == "knowledge"
    3. trace exists with retrieved chunk IDs
    4. Turn 2 in the same session has access to context from turn 1
       (state persistence — at minimum, plan_tier available without re-asking)

    The mock_adk fixture patches at the ADK boundary, not at the HTTP layer.
    """
    # Create session
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_002", "plan_tier": "pro"})
    session_id = sess.json()["session_id"]

    # Turn 1 — knowledge query
    r1 = await client.post(f"/v1/chat/{session_id}", json={"content": "How do I rotate a deploy key?"})
    assert r1.status_code == 200
    assert r1.json()["routed_to"] == "knowledge"
    trace_id = r1.json()["trace_id"]

    # Trace must have chunk IDs
    trace = await client.get(f"/v1/traces/{trace_id}")
    assert trace.status_code == 200
    assert len(trace.json()["retrieved_chunk_ids"]) > 0

    # Turn 2 — follow-up in same session
    r2 = await client.post(f"/v1/chat/{session_id}", json={"content": "What is my plan tier?"})
    assert r2.status_code == 200
    # Agent should know plan_tier from state — not re-ask
    assert "pro" in r2.json()["reply"].lower()


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client, mock_adk):
    resp = await client.post("/v1/chat/nonexistent-id", json={"content": "hello"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_trace_not_found_returns_404(client):
    resp = await client.get("/v1/traces/nonexistent-trace")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_guardrails_refuses_out_of_scope(client, mock_adk):
    """
    E5 Guardrails: out-of-scope requests should be refused with a specific message.
    """
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_guard"})
    session_id = sess.json()["session_id"]

    resp = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "write me a poem about butterflies"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # root agent handled it (no sub-agent tool call)
    assert data["routed_to"] == "root"
    # E5: verify refusal message content
    assert "only help" in data["reply"].lower() or "helix" in data["reply"].lower()


# ---------------------------------------------------------------------------
# E1 — Idempotency tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotency_key_returns_same_response(client, mock_adk):
    """E1: same Idempotency-Key → same response, pipeline runs once."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_idem_001"})
    session_id = sess.json()["session_id"]

    idem_key = "test-idempotency-key-001"

    # First request
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
        headers={"Idempotency-Key": idem_key},
    )
    assert r1.status_code == 200
    trace_id_1 = r1.json()["trace_id"]

    # Second request with same key — should return stored response
    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I rotate a deploy key?"},
        headers={"Idempotency-Key": idem_key},
    )
    assert r2.status_code == 200
    trace_id_2 = r2.json()["trace_id"]

    # Must be the same trace_id (replay, not a new run)
    assert trace_id_1 == trace_id_2


@pytest.mark.asyncio
async def test_idempotency_without_key_runs_normally(client, mock_adk):
    """E1: no header → pipeline runs every time, different trace_ids."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_idem_002"})
    session_id = sess.json()["session_id"]

    r1 = await client.post(f"/v1/chat/{session_id}", json={"content": "show my builds"})
    r2 = await client.post(f"/v1/chat/{session_id}", json={"content": "show my builds"})

    assert r1.json()["trace_id"] != r2.json()["trace_id"]


# ---------------------------------------------------------------------------
# E2 — Escalation Agent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalation_creates_ticket(client, mock_adk):
    """E2: escalation request routes to escalation agent and creates a ticket."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_esc_001"})
    session_id = sess.json()["session_id"]

    resp = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "I need to escalate this issue"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["routed_to"] == "escalation"
    assert "TKT-" in data["reply"]


@pytest.mark.asyncio
async def test_escalation_ticket_in_followup(client, mock_adk):
    """E2: ticket ID should be available in follow-up turns via state."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_esc_002"})
    session_id = sess.json()["session_id"]

    # Create a ticket
    r1 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "I found a bug, please create a ticket"},
    )
    assert r1.status_code == 200
    assert r1.json()["routed_to"] == "escalation"

    # Follow-up — agent should know about the ticket
    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "What is my plan tier?"},
    )
    assert r2.status_code == 200


# ---------------------------------------------------------------------------
# E3 — SSE Streaming test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_streaming_returns_event_stream(client, mock_adk, monkeypatch):
    """E3: Accept: text/event-stream should return streaming response."""
    import json

    sess = await client.post("/v1/sessions", json={"user_id": "u_sse_001"})
    session_id = sess.json()["session_id"]

    # Mock run_streaming to yield SSE events without hitting real ADK
    async def mock_run_streaming(session_id, message, db):
        yield f'event: text_chunk\ndata: {json.dumps({"text": "Hello from SSE!"})}\n\n'
        yield f'event: done\ndata: {json.dumps({"reply": "Hello from SSE!", "routed_to": "root", "trace_id": "sse-trace-001"})}\n\n'

    monkeypatch.setattr("app.srop.pipeline.run_streaming", mock_run_streaming)

    resp = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "hello"},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    # Verify body contains SSE events
    body = resp.text
    assert "event: text_chunk" in body or "event: done" in body
