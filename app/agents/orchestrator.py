"""
SROP Root Orchestrator — Google ADK agent.

Routes every user turn to KnowledgeAgent, AccountAgent, or EscalationAgent
via ADK's AgentTool. This means the LLM decides which tool to call — you
do not parse its output.

Intent → sub-agent:
  knowledge:  "how do I X", "what is X", docs questions
  account:    "show my builds", "my account status", usage questions
  escalation: "create a ticket", "escalate", "report a bug", "I need help from support"
  smalltalk:  greetings, thanks — root agent handles inline (no tool call)

Pattern 3 is used: SessionState is loaded from DB and injected into the
instruction string on every turn. This means the root agent is built fresh
per turn via build_root_agent(state).

See docs/google-adk-guide.md for AgentTool pattern and event extraction.
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents.account import account_agent
from app.agents.escalation import escalation_agent
from app.agents.knowledge import knowledge_agent
from app.settings import settings
from app.srop.state import SessionState

ROOT_INSTRUCTION = """\
You are the Helix Support Concierge — a routing agent.
Call the correct specialist tool based on the user's intent.

Intent → tool:
- HOW to do something, WHAT something is, docs/feature questions → knowledge_agent
- Their account, builds, status, usage → account_agent
- Create a ticket, escalate, report a bug, need human support → escalation_agent
- Greetings or off-topic → respond directly, no tool call

If the user's request is not about Helix products, their account, or support
(e.g. creative writing, poems, unrelated topics), respond:
"I can only help with Helix product and account questions."
Do NOT call any tools for out-of-scope requests.

Always call a tool when intent matches. Never answer knowledge or account questions yourself.
User context will be provided below — use it when relevant.
"""

_knowledge_tool = AgentTool(agent=knowledge_agent)
_account_tool = AgentTool(agent=account_agent)
_escalation_tool = AgentTool(agent=escalation_agent)


def build_root_agent(state: SessionState) -> LlmAgent:
    """
    Build the root orchestrator with session state injected into its instruction.

    Pattern 3: the agent sees user context (plan_tier, last_agent, etc.) via its
    system prompt rather than via ADK session state or full message history.
    """
    ticket_info = f"- last_ticket_id: {state.last_ticket_id}" if state.last_ticket_id else ""

    instruction_with_context = f"""{ROOT_INSTRUCTION}

Current user context:
- user_id: {state.user_id}
- plan_tier: {state.plan_tier}
- last_agent: {state.last_agent or "none"}
- turn_count: {state.turn_count}
{ticket_info}
"""
    return LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        instruction=instruction_with_context,
        tools=[_knowledge_tool, _account_tool, _escalation_tool],
    )
