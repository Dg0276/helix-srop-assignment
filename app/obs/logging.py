"""
Structured logging setup.

All log lines must include session_id, trace_id, user_id when available.
Use structlog's context vars for request-scoped fields.

E5: PII redaction processor strips sensitive data before logging.
"""
import logging
import sys
from typing import Any

import structlog

from app.obs.guardrails import redact_pii


def _pii_redaction_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """E5: Redact PII from all string values in log events."""
    for key, value in event_dict.items():
        if isinstance(value, str) and key not in ("event", "level", "timestamp"):
            event_dict[key] = redact_pii(value)
    return event_dict


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _pii_redaction_processor,  # E5: redact before rendering
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    )


# Usage in request handlers:
#   import structlog
#   log = structlog.get_logger()
#   structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)
#   log.info("pipeline_started", user_message_len=len(message))
