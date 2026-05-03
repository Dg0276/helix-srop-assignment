"""
E5 — PII redaction and guardrails utilities.

Redacts emails, phone numbers, SSNs, and credit card numbers from text
before it is logged or stored in traces. This ensures sensitive data
never reaches application logs or debug endpoints.
"""
import re


# ---------------------------------------------------------------------------
# PII patterns (compiled once for performance)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(
    r"(?:\+?1[-.\s]?)?"  # optional country code
    r"(?:\(?\d{3}\)?[-.\s]?)"  # area code
    r"\d{3}[-.\s]?\d{4}"  # number
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")


def redact_pii(text: str) -> str:
    """
    Redact known PII patterns from text.

    Replaces:
    - Email addresses → [EMAIL_REDACTED]
    - Phone numbers → [PHONE_REDACTED]
    - SSNs → [SSN_REDACTED]
    - Credit card numbers → [CC_REDACTED]

    Returns the redacted text. Original text is never stored.
    """
    text = _SSN_RE.sub("[SSN_REDACTED]", text)
    text = _CREDIT_CARD_RE.sub("[CC_REDACTED]", text)
    text = _EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = _PHONE_RE.sub("[PHONE_REDACTED]", text)
    return text
