"""
AccountAgent — handles account lookups, build history, usage queries.

Exposes get_recent_builds and get_account_status as ADK tools.
Uses mock data for the take-home; the wiring and routing is what's evaluated.
"""
from google.adk.agents import LlmAgent

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.settings import settings

ACCOUNT_INSTRUCTION = """\
You are the Helix Account Agent — an account and build specialist.

When the user asks about their account, builds, usage, or plan:
1. Call the appropriate tool (get_recent_builds or get_account_status).
2. Summarize the results clearly for the user.
3. If asked about failed builds specifically, filter and highlight those.

Always base your answer on the tool results, not assumptions.
"""

account_agent = LlmAgent(
    name="account_agent",
    model=settings.adk_model,
    instruction=ACCOUNT_INSTRUCTION,
    tools=[get_recent_builds, get_account_status],
)
