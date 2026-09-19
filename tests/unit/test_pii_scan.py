"""Privacy: PII scanner - zero raw PII in decision/trace artifacts."""
from __future__ import annotations

from types import SimpleNamespace

from src.privacy.pii_scan import scan_spans, scan_value


def test_scan_value_flags_email_phone_ssn() -> None:
    """Negative control: the scanner DOES flag raw PII when present."""
    findings = scan_value("contact jane.doe@example.com or 555-123-4567, SSN 123-45-6789")
    kinds = {k for k, _ in findings}
    assert "email" in kinds and "phone" in kinds and "ssn" in kinds


def test_scan_spans_flags_bad_span_and_clean_spans_pass() -> None:
    """Spans carrying only pseudonymous ids scan clean; raw PII is flagged."""
    clean = SimpleNamespace(
        name="agent.support_agent",
        attributes={"c360.customer_id": "CUST_00088", "c360.top_k": 5},
    )
    bad = SimpleNamespace(
        name="bad_span",
        attributes={"c360.contact": "jane.doe@example.com", "c360.phone": "555-123-4567"},
    )
    assert scan_spans([clean]) == {}
    violations = scan_spans([bad])
    assert "bad_span" in violations
    kinds = {k.split(":")[-1] for k, _ in violations["bad_span"]}
    assert "email" in kinds and "phone" in kinds