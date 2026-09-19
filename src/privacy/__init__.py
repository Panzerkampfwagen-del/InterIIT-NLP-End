"""Privacy utilities: PII scanning over trace/decision artifacts.

Tracing itself is deferred to the final submission; the PII scanner is kept
as the privacy production-bar component: raw PII must never appear in
observability artifacts or decision logs.
"""
from src.privacy.pii_scan import PII_PATTERNS, scan_spans, scan_value

__all__ = ["PII_PATTERNS", "scan_spans", "scan_value"]