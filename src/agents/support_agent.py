"""Support agent (Phase 5) - lexicon sentiment + repeated contact.

Sentiment classification is a lightweight deterministic lexicon (negative-key
matching on ticket/transcript text); topic extraction on negative or
repeated-contact tickets is the only LLM step. Detection itself is
statistical-first (RQ-L1).
"""
from __future__ import annotations

from typing import Any

from src.agents.base import SignalAgent
from src.ingestion.schemas.events import EventEnvelope

# Deterministic negative-sentiment lexicon (domain words, not general sentiment).
NEGATIVE_LEXICON = frozenset(
    {
        "hospital", "hospitalized", "emergency", "refund", "complaint", "complain",
        "cancel", "cancelled", "canceled", "fee", "dispute", "angry", "terrible",
        "unacceptable", "frustrated", "frustrating", "broken", "failed", "denied",
        "overcharged", "fraud", "stolen", "scam", "income dropped", "lost my job",
        "layoff", "hardship", "payment plan",
    }
)
REPEATED_CONTACT_THRESHOLD = 2


class SupportAgent(SignalAgent):
    name = "support_agent"
    domain = "support"

    def __init__(self, board, llm=None, indexer=None) -> None:
        super().__init__(board, llm)
        self.indexer = indexer  # optional RetrievalIndexer (Phase 6 wiring)

    def analyze_and_record(self, event, state, event_time=None):
        finding = super().analyze_and_record(event, state, event_time)
        # Phase 6 wiring: index the ticket/transcript text incrementally (RQ-R1)
        if self.indexer is not None and event.source == "support_logs":
            self.indexer.index_event(event)
        return finding

    def detect(self, event: EventEnvelope, state) -> list[dict[str, Any]]:
        if event.source != "support_logs":
            return []
        flags: list[dict[str, Any]] = []
        text = (event.payload.get("raw_text") or "").lower() if isinstance(event.payload, dict) else ""
        category = event.payload.get("category") if isinstance(event.payload, dict) else None

        # negative sentiment via lexicon match
        matched = sorted(w for w in NEGATIVE_LEXICON if w in text)
        if matched or event.event_type == "ticket_created" and (category in {"complaint", "dispute", "fee_issue"}):
            confidence = min(0.9, 0.55 + 0.08 * len(matched))
            flags.append({
                "type": "negative_sentiment",
                "confidence": confidence,
                "evidence": [event.event_id],
                "detail": f"lexicon matches: {matched[:5]}" if matched else f"category={category}",
            })

        # repeated contact: >= 2 tickets in 30d from recent refs
        recent = getattr(state, "recent_events", []) or []
        tickets = [
            r for r in recent
            if r.get("source_system") == "support_logs" and r.get("event_type") == "ticket_created"
            and _within(r.get("event_time"), event.event_time, days=30)
        ]
        if event.event_type == "ticket_created" and len(tickets) + 1 >= REPEATED_CONTACT_THRESHOLD:
            flags.append({
                "type": "repeated_contact",
                "confidence": 0.65,
                "evidence": [r["event_id"] for r in tickets] + [event.event_id],
                "detail": f"{len(tickets) + 1} tickets in 30d",
            })
        return flags


def _within(ref_time, event_time, days: float) -> bool:
    from datetime import datetime

    if ref_time is None:
        return False
    if isinstance(ref_time, str):
        ref = datetime.fromisoformat(ref_time.replace("Z", "+00:00"))
    else:
        ref = ref_time
    et = event_time if event_time.tzinfo else event_time.replace(tzinfo=ref.tzinfo)
    return abs((et - ref).total_seconds()) <= days * 86400
