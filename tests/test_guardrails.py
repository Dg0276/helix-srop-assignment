"""
E5 — Guardrails unit tests: PII redaction and refusal verification.
"""
import pytest

from app.obs.guardrails import redact_pii


class TestPiiRedaction:
    """E5: PII redaction must catch common patterns."""

    def test_redacts_email(self):
        text = "Contact me at john.doe@example.com for details"
        result = redact_pii(text)
        assert "john.doe@example.com" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_redacts_phone_number(self):
        text = "Call me at 555-123-4567 or (555) 987-6543"
        result = redact_pii(text)
        assert "555-123-4567" not in result
        assert "[PHONE_REDACTED]" in result

    def test_redacts_ssn(self):
        text = "My SSN is 123-45-6789"
        result = redact_pii(text)
        assert "123-45-6789" not in result
        assert "[SSN_REDACTED]" in result

    def test_redacts_credit_card(self):
        text = "Card number: 4111 1111 1111 1111"
        result = redact_pii(text)
        assert "4111 1111 1111 1111" not in result
        assert "[CC_REDACTED]" in result

    def test_preserves_non_pii_text(self):
        text = "How do I rotate a deploy key?"
        result = redact_pii(text)
        assert result == text

    def test_redacts_multiple_pii_types(self):
        text = "Email: user@test.com, Phone: 555-111-2222, SSN: 999-88-7777"
        result = redact_pii(text)
        assert "user@test.com" not in result
        assert "555-111-2222" not in result
        assert "999-88-7777" not in result
        assert "[EMAIL_REDACTED]" in result
        assert "[PHONE_REDACTED]" in result
        assert "[SSN_REDACTED]" in result

    def test_empty_string(self):
        assert redact_pii("") == ""
