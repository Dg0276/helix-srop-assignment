"""
E2 — EscalationAgent: handles requests to create support tickets.

When a user wants to escalate an issue, report a bug, or request human support,
the root orchestrator routes to this agent. It calls create_ticket and returns
the ticket ID to the user.
"""
from google.adk.agents import LlmAgent

from app.agents.tools.escalation_tools import create_ticket
from app.settings import settings

ESCALATION_INSTRUCTION = """\
You are the Helix Escalation Agent — a support ticket specialist.

When the user wants to:
- Escalate an issue to a human
- Report a bug or problem
- Create a support ticket
- Get help from the support team

Do the following:
1. Call the create_ticket tool with the user's user_id, a clear summary of their issue, and an appropriate priority level.
2. Tell the user their ticket has been created and give them the ticket ID.
3. Reassure them that the support team will follow up.

Priority guidelines:
- critical: system outage, data loss, security breach
- high: feature broken, blocking work
- medium: general issues, questions needing human help
- low: feature requests, minor annoyances
"""

escalation_agent = LlmAgent(
    name="escalation_agent",
    model=settings.adk_model,
    instruction=ESCALATION_INSTRUCTION,
    tools=[create_ticket],
)
