"""KYC agent (Phase 5) - fully deterministic rule engine.

KYC/compliance facts are high-stakes and low-velocity: they are deterministic
rules with **confidence 1.0** and **zero LLM calls** (per the Phase 5 prompt's
KYC-agent design; RQ-L1 corroborates: facts are not inferred).
"""
from __future__ import annotations

from typing import Any

from src.agents.base import SignalAgent
from src.ingestion.schemas.events import EventEnvelope

WATCHLIST_CATEGORIES = frozenset({"watchlist_hit", "pep_hit", "sanctions_hit"})


class KycAgent(SignalAgent):
    name = "kyc_agent"
    domain = "compliance"

    def detect(self, event: EventEnvelope, state) -> list[dict[str, Any]]:
        if event.source != "loan_kyc":
            return []
        flags: list[dict[str, Any]] = []
        payload = event.payload if isinstance(event.payload, dict) else {}
        subtype = payload.get("event_subtype") or event.event_type

        # deterministic facts, confidence 1.0 each
        if event.event_type in {"marital_status_change", "dependents_change"}:
            flags.append({
                "type": "household_change",
                "confidence": 1.0,
                "evidence": [event.event_id],
                "detail": f"{event.event_type}: {payload.get('old_value')} -> {payload.get('new_value')}",
            })
        if event.event_type == "address_change" or subtype == "address_change":
            flags.append({
                "type": "address_change_recent",
                "confidence": 1.0,
                "evidence": [event.event_id],
                "detail": f"address changed: {payload.get('old_value')} -> {payload.get('new_value')}",
            })
        if event.event_type == "kyc_update":
            flags.append({
                "type": "kyc_update",
                "confidence": 1.0,
                "evidence": [event.event_id],
                "detail": f"kyc update: {subtype}",
            })
        if subtype in WATCHLIST_CATEGORIES:
            flags.append({
                "type": "watchlist_hit",
                "confidence": 1.0,
                "evidence": [event.event_id],
                "detail": f"watchlist category: {subtype}",
            })
        if event.event_type in {"loan_application", "loan_disbursed"}:
            flags.append({
                "type": "loan_activity",
                "confidence": 1.0,
                "evidence": [event.event_id],
                "detail": event.event_type,
            })
        return flags

    def narrate(self, event: EventEnvelope, flags: list[dict[str, Any]]) -> str:
        """KYC narration is deterministic - NEVER calls the LLM (design contract)."""
        return self._deterministic_summary(event, flags)
