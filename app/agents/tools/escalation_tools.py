"""
E2 — Escalation tools used by the EscalationAgent.

create_ticket writes a row to the tickets table and returns the ticket ID.
The ticket ID is then stored in SessionState for follow-up turns.
"""
import uuid
from datetime import datetime, timezone


# In-memory store for ADK tool context (the pipeline will persist to DB separately).
# This allows the tool to work without needing a DB session injected.
_PENDING_TICKETS: list[dict] = []


async def create_ticket(
    user_id: str,
    summary: str,
    priority: str = "medium",
) -> dict:
    """
    Create a support ticket for a user.

    Args:
        user_id: the user requesting escalation
        summary: description of the issue to escalate
        priority: ticket priority — one of: low, medium, high, critical

    Returns:
        Dict with ticket_id, user_id, summary, priority, status, and created_at.
    """
    valid_priorities = {"low", "medium", "high", "critical"}
    if priority.lower() not in valid_priorities:
        priority = "medium"

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc).isoformat()

    ticket = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "summary": summary,
        "priority": priority.lower(),
        "status": "open",
        "created_at": now,
    }

    _PENDING_TICKETS.append(ticket)

    return ticket
