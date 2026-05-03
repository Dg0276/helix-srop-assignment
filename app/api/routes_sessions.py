"""
POST /v1/sessions — create a session.
"""
import uuid
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session, User
from app.db.session import get_db
from app.srop.state import SessionState

router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    plan_tier: Literal["free", "pro", "enterprise"] = "free"


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """
    Create a new session. Upsert the user if not seen before.
    Initialize SessionState and persist to DB.
    """
    session_id = str(uuid.uuid4())

    # Upsert user
    existing_user = await db.get(User, body.user_id)
    if existing_user is None:
        db.add(User(user_id=body.user_id, plan_tier=body.plan_tier))
    else:
        existing_user.plan_tier = body.plan_tier

    # Build initial state
    state = SessionState(user_id=body.user_id, plan_tier=body.plan_tier)

    # Create session row
    db.add(Session(
        session_id=session_id,
        user_id=body.user_id,
        state=state.to_db_dict(),
    ))

    await db.commit()

    return CreateSessionResponse(session_id=session_id, user_id=body.user_id)
