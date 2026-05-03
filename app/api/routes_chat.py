"""
POST /v1/chat/{session_id} — send a user message, get assistant reply.

Extensions:
  E1: Idempotency-Key header support (replay returns original response)
  E3: Accept: text/event-stream → SSE streaming response
"""
import json
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IdempotencyRecord
from app.db.session import get_db
from app.srop import pipeline

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    reply: str
    routed_to: str   # which sub-agent handled this turn
    trace_id: str


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> ChatResponse | StreamingResponse:
    """
    Run one turn of the SROP pipeline.

    Headers:
    - Idempotency-Key (optional): if provided, replays return original response
    - Accept: text/event-stream → streams SSE events

    Error cases:
    - Session not found → 404
    - LLM timeout → 504
    """
    # ── E1: Idempotency check ────────────────────────────────────────────
    if idempotency_key:
        existing = await db.execute(
            select(IdempotencyRecord).where(
                IdempotencyRecord.key == idempotency_key,
                IdempotencyRecord.session_id == session_id,
            )
        )
        record = existing.scalar_one_or_none()
        if record is not None:
            # Replay: return the stored response without running the pipeline
            return ChatResponse(**record.response_json)

    # ── E3: SSE streaming ────────────────────────────────────────────────
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        return StreamingResponse(
            pipeline.run_streaming(session_id, body.content, db),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Standard JSON response ───────────────────────────────────────────
    result = await pipeline.run(session_id, body.content, db)
    response = ChatResponse(reply=result.content, routed_to=result.routed_to, trace_id=result.trace_id)

    # ── E1: Store response for idempotency replay ────────────────────────
    if idempotency_key:
        try:
            db.add(IdempotencyRecord(
                key=idempotency_key,
                session_id=session_id,
                response_json=response.model_dump(),
            ))
            await db.commit()
        except Exception:
            # Don't fail the request if idempotency write fails
            try:
                await db.rollback()
            except Exception:
                pass

    return response
