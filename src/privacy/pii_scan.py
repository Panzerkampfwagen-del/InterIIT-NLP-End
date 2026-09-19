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
    ("phone", re.compile(r"(?:\+?\d[\d\s().-]{8,}\d)")),
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
