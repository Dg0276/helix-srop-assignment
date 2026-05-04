"""
SROP entrypoint — called by the message route.

This is the core of the assignment. It ties together:
  - Loading session state from DB
  - Running the ADK orchestrator with that state as context (Pattern 3)
  - Extracting routing decision and tool calls from ADK events
  - Recording the trace
  - Persisting updated session state to DB

Extensions integrated:
  - E2: Escalation agent ticket persistence
  - E3: Streaming SSE support via run_streaming()
  - E5: PII redaction in logs

The route calls: result = await pipeline.run(session_id, user_message, db)
It receives: PipelineResult(content, routed_to, trace_id)
"""
import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from google.genai import types

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import build_root_agent
from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, Message, Session, Ticket
from app.obs.guardrails import redact_pii
from app.settings import settings
from app.srop.state import SessionState

log = structlog.get_logger()


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


async def _run_agent_and_collect(
    state: SessionState,
    user_message: str,
) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    """
    Build the root agent, run it, and collect events.

    Returns (final_text, routed_to, tool_calls, retrieved_chunk_ids).
    """
    from google.adk.runners import InMemoryRunner
    from google.adk.sessions import InMemorySessionService

    agent = build_root_agent(state)
    runner = InMemoryRunner(agent=agent, app_name="helix_srop")
    session_service = runner.session_service

    adk_session = await session_service.create_session(
        app_name="helix_srop",
        user_id=state.user_id,
    )

    routed_to = "root"
    tool_calls: list[dict[str, Any]] = []
    retrieved_chunk_ids: list[str] = []
    final_text = ""
    current_tool_call: dict[str, Any] | None = None

    response = runner.run_async(
        user_id=state.user_id,
        session_id=adk_session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        ),
    )

    async for event in response:
        # Capture tool calls for tracing
        if hasattr(event, "tool_name") and event.tool_name:
            current_tool_call = {
                "tool_name": event.tool_name,
                "args": getattr(event, "tool_args", {}) or getattr(event, "args", {}),
                "result": None,
            }
            tool_calls.append(current_tool_call)

        # Capture tool results
        if hasattr(event, "tool_result") and event.tool_result is not None:
            if current_tool_call is not None:
                current_tool_call["result"] = event.tool_result

        # Extract chunk IDs from search_docs results
        if hasattr(event, "tool_name") and event.tool_name == "search_docs":
            result_data = getattr(event, "tool_result", None)
            if result_data and isinstance(result_data, list):
                for chunk in result_data:
                    if isinstance(chunk, dict) and "chunk_id" in chunk:
                        retrieved_chunk_ids.append(chunk["chunk_id"])

        # Final response — get text and routing info
        if hasattr(event, "is_final_response") and event.is_final_response():
            author = getattr(event, "author", None)
            if author and author != "srop_root":
                routed_to = author.replace("_agent", "")
            if hasattr(event, "content") and event.content:
                parts = getattr(event.content, "parts", [])
                if parts:
                    final_text = getattr(parts[0], "text", "") or ""

    return final_text, routed_to, tool_calls, retrieved_chunk_ids


async def _run_agent_streaming(
    state: SessionState,
    user_message: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    E3 — Streaming variant. Yields SSE-formatted event dicts as they arrive.

    Events: {"event": "tool_call"|"text_chunk"|"done", "data": ...}
    """
    from google.adk.runners import InMemoryRunner
    from google.adk.sessions import InMemorySessionService

    agent = build_root_agent(state)
    runner = InMemoryRunner(agent=agent, app_name="helix_srop")
    session_service = runner.session_service

    adk_session = await session_service.create_session(
        app_name="helix_srop",
        user_id=state.user_id,
    )

    routed_to = "root"
    tool_calls: list[dict[str, Any]] = []
    retrieved_chunk_ids: list[str] = []
    final_text = ""

    response = runner.run_async(
        user_id=state.user_id,
        session_id=adk_session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        ),
    )

    async for event in response:
        # Capture and stream tool calls
        if hasattr(event, "tool_name") and event.tool_name:
            tc = {
                "tool_name": event.tool_name,
                "args": getattr(event, "tool_args", {}) or getattr(event, "args", {}),
                "result": None,
            }
            tool_calls.append(tc)
            yield {"event": "tool_call", "data": {"tool_name": tc["tool_name"], "args": tc["args"]}}

        # Stream tool results
        if hasattr(event, "tool_result") and event.tool_result is not None:
            if tool_calls:
                tool_calls[-1]["result"] = event.tool_result

        # Extract chunk IDs
        if hasattr(event, "tool_name") and event.tool_name == "search_docs":
            result_data = getattr(event, "tool_result", None)
            if result_data and isinstance(result_data, list):
                for chunk in result_data:
                    if isinstance(chunk, dict) and "chunk_id" in chunk:
                        retrieved_chunk_ids.append(chunk["chunk_id"])

        # Final response
        if hasattr(event, "is_final_response") and event.is_final_response():
            author = getattr(event, "author", None)
            if author and author != "srop_root":
                routed_to = author.replace("_agent", "")
            if hasattr(event, "content") and event.content:
                parts = getattr(event.content, "parts", [])
                if parts:
                    text = getattr(parts[0], "text", "") or ""
                    if text:
                        final_text = text
                        yield {"event": "text_chunk", "data": {"text": text}}

    if not final_text:
        final_text = "I'm sorry, I wasn't able to generate a response. Please try again."
        yield {"event": "text_chunk", "data": {"text": final_text}}

    yield {
        "event": "done",
        "data": {
            "routed_to": routed_to,
            "tool_calls": tool_calls,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "final_text": final_text,
        },
    }


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    """Run one turn of the SROP pipeline."""
    trace_id = str(uuid.uuid4())
    start = time.monotonic()

    # Bind structlog context for this request
    structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)

    # ── 1. Load session from DB ──────────────────────────────────────────
    result = await db.execute(
        select(Session).where(Session.session_id == session_id)
    )
    session_row = result.scalar_one_or_none()
    if session_row is None:
        raise SessionNotFoundError(f"Session {session_id} does not exist")

    state = SessionState.from_db_dict(session_row.state)
    structlog.contextvars.bind_contextvars(user_id=state.user_id)

    # E5: log with PII redacted
    log.info("pipeline_started", user_message_len=len(user_message), turn=state.turn_count + 1,
             message_preview=redact_pii(user_message[:100]))

    # ── 2. Run ADK agent with timeout ────────────────────────────────────
    try:
        final_text, routed_to, tool_calls, retrieved_chunk_ids = await asyncio.wait_for(
            _run_agent_and_collect(state, user_message),
            timeout=settings.llm_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        )

    if not final_text:
        final_text = "I'm sorry, I wasn't able to generate a response. Please try again."

    latency_ms = int((time.monotonic() - start) * 1000)

    log.info(
        "pipeline_completed",
        routed_to=routed_to,
        latency_ms=latency_ms,
        tool_call_count=len(tool_calls),
        chunk_count=len(retrieved_chunk_ids),
    )

    # ── 3. Persist messages, trace, and updated state ────────────────────
    try:
        # User message
        db.add(Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=user_message,
            trace_id=trace_id,
        ))

        # Assistant message
        db.add(Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=final_text,
            trace_id=trace_id,
        ))

        # Agent trace
        db.add(AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=routed_to,
            tool_calls=tool_calls,
            retrieved_chunk_ids=retrieved_chunk_ids,
            latency_ms=latency_ms,
        ))

        # E2: persist any tickets created by the escalation agent
        from app.agents.tools.escalation_tools import _PENDING_TICKETS
        for ticket_data in _PENDING_TICKETS:
            db.add(Ticket(
                ticket_id=ticket_data["ticket_id"],
                user_id=ticket_data["user_id"],
                summary=ticket_data["summary"],
                priority=ticket_data["priority"],
                status=ticket_data["status"],
            ))
            state.last_ticket_id = ticket_data["ticket_id"]
        _PENDING_TICKETS.clear()

        # Update session state
        state.turn_count += 1
        state.last_agent = routed_to if routed_to in ("knowledge", "account", "escalation") else state.last_agent
        session_row.state = state.to_db_dict()

        await db.commit()
    except Exception as exc:
        # DB failure after LLM responded — log but return the reply (don't 500 the user)
        log.error("db_write_failed_after_llm", error=str(exc), exc_info=True)
        try:
            await db.rollback()
        except Exception:
            pass

    return PipelineResult(content=final_text, routed_to=routed_to, trace_id=trace_id)


async def run_streaming(
    session_id: str,
    user_message: str,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    """
    E3 — Streaming variant of run(). Yields SSE-formatted lines.

    Format:
        event: tool_call
        data: {"tool_name": "search_docs", "args": {...}}

        event: text_chunk
        data: {"text": "..."}

        event: done
        data: {"routed_to": "knowledge", "trace_id": "..."}
    """
    import json

    trace_id = str(uuid.uuid4())
    start = time.monotonic()

    structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)

    # Load session
    result = await db.execute(
        select(Session).where(Session.session_id == session_id)
    )
    session_row = result.scalar_one_or_none()
    if session_row is None:
        yield f"event: error\ndata: {json.dumps({'error': 'SESSION_NOT_FOUND'})}\n\n"
        return

    state = SessionState.from_db_dict(session_row.state)
    log.info("pipeline_streaming_started", turn=state.turn_count + 1,
             message_preview=redact_pii(user_message[:100]))

    final_text = ""
    routed_to = "root"
    tool_calls: list[dict[str, Any]] = []
    retrieved_chunk_ids: list[str] = []

    try:
        async for event in _run_agent_streaming(state, user_message):
            if event["event"] == "done":
                done_data = event["data"]
                final_text = done_data["final_text"]
                routed_to = done_data["routed_to"]
                tool_calls = done_data["tool_calls"]
                retrieved_chunk_ids = done_data["retrieved_chunk_ids"]
            else:
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
    except asyncio.TimeoutError:
        yield f"event: error\ndata: {json.dumps({'error': 'UPSTREAM_TIMEOUT'})}\n\n"
        return

    if not final_text:
        final_text = "I'm sorry, I wasn't able to generate a response. Please try again."

    latency_ms = int((time.monotonic() - start) * 1000)

    # Persist (same as non-streaming)
    try:
        db.add(Message(message_id=str(uuid.uuid4()), session_id=session_id,
                       role="user", content=user_message, trace_id=trace_id))
        db.add(Message(message_id=str(uuid.uuid4()), session_id=session_id,
                       role="assistant", content=final_text, trace_id=trace_id))
        db.add(AgentTrace(trace_id=trace_id, session_id=session_id, routed_to=routed_to,
                          tool_calls=tool_calls, retrieved_chunk_ids=retrieved_chunk_ids,
                          latency_ms=latency_ms))

        from app.agents.tools.escalation_tools import _PENDING_TICKETS
        for ticket_data in _PENDING_TICKETS:
            db.add(Ticket(
                ticket_id=ticket_data["ticket_id"], user_id=ticket_data["user_id"],
                summary=ticket_data["summary"], priority=ticket_data["priority"],
                status=ticket_data["status"],
            ))
            state.last_ticket_id = ticket_data["ticket_id"]
        _PENDING_TICKETS.clear()

        state.turn_count += 1
        state.last_agent = routed_to if routed_to in ("knowledge", "account", "escalation") else state.last_agent
        session_row.state = state.to_db_dict()
        await db.commit()
    except Exception as exc:
        log.error("db_write_failed_after_llm_streaming", error=str(exc), exc_info=True)
        try:
            await db.rollback()
        except Exception:
            pass

    # Final SSE event
    yield f"event: done\ndata: {json.dumps({'reply': final_text, 'routed_to': routed_to, 'trace_id': trace_id})}\n\n"
