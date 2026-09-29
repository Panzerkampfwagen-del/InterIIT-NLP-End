"""PII scanner over exported spans (Phase 11, #27).

Raw PII (emails, phone numbers, SSN-like ids) must never appear in span
attributes or names. The scanner walks exported spans and flags violations -
used in the DoD test to prove spans are PII-clean.
"""
from __future__ import annotations

import re
from typing import Any

PII_PATTERNS = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    # phone: 10+ chars of digits/punctuation; the date lookahead prevents
    # ISO dates (2026-05-01, 8 digits) from false-positiveing as phones
    ("phone", re.compile(r"(?!\d{4}-\d{2}-\d{2})(?:\+?\d[\d\s().-]{8,}\d)")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
)


def scan_value(value: Any) -> list[tuple[str, str]]:
    """Scan one attribute value for raw PII patterns."""
    findings: list[tuple[str, str]] = []
    if not isinstance(value, str):
        return findings
    for name, pattern in PII_PATTERNS:
        for match in pattern.findall(value):
            findings.append((name, match))
    return findings


def scan_spans(spans: list[Any]) -> dict[str, list[tuple[str, str]]]:
    """Scan exported spans; return {span_name: [pii findings]} (empty = clean)."""
    out: dict[str, list[tuple[str, str]]] = {}
    for span in spans:
        name = span.name
        findings: list[tuple[str, str]] = []
        findings.extend(_scan_attrs(span.attributes or {}))
        findings.extend(scan_value(name))
        if findings:
            out.setdefault(name, []).extend(findings)
    return out


def _scan_attrs(attrs: dict[str, Any]) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for key, value in attrs.items():
        if isinstance(value, str):
            findings.extend((f"{key}:{name}", match) for name, match in scan_value(value))
        elif isinstance(value, (list, tuple)):
            for v in value:
                if isinstance(v, str):
                    findings.extend((f"{key}:{name}", match) for name, match in scan_value(v))
    return findings


def redact_text(text: str) -> str:
    """Replace raw PII with typed placeholders (used before LLM/tool calls)."""
    for name, pattern in PII_PATTERNS:
        text = pattern.sub(f"[{name.upper()}_REDACTED]", text)
    return text


def redact_value(value: Any) -> Any:
    """Recursively redact PII from strings in dicts/lists/tuples.

    The single sanitization choke point: every string that leaves the process
    (LLM prompts, tool payloads, exported attributes) flows through here.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(v) for v in value]
    return value
