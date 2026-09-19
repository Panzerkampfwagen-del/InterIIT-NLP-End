"""Transaction agent (Phase 5) - statistical detection (RQ-L1).

Detects transaction-pattern breakdowns against historical baselines:
- decline bursts (decline_count_24h threshold);
- accelerated withdrawals (large_withdrawal_count_7d);
- count anomalies (z-score of current txn_count vs baseline mean/std).
Detection is deterministic; the LLM only narrates flagged findings.
"""
from __future__ import annotations

from typing import Any

from src.agents.base import SignalAgent
from src.ingestion.schemas.events import EventEnvelope
from src.streaming.baselines import zscore

DECLINE_BURST_THRESHOLD = 3  # declines in 24h (docs example)
WITHDRAWAL_BURST_THRESHOLD = 2  # large withdrawals in 7d
ZSCORE_FLAG = 2.0


class TransactionAgent(SignalAgent):
    name = "transaction_agent"
    domain = "transaction"

    def detect(self, event: EventEnvelope, state) -> list[dict[str, Any]]:
        if event.source not in {"card_payments", "core_banking_ledger", "instant_payments", "ach_wire"}:
            return []
        flags: list[dict[str, Any]] = []
        baseline = (state.historical_baseline or {}) if state is not None else {}
        recent = getattr(state, "recent_events", []) or []

        # decline burst: count declines in the last 24h from recent event refs
        declines = [
            r for r in recent
            if r.get("source_system") == "card_payments" and r.get("event_type") == "decline"
            and r.get("event_time") and _within(r["event_time"], event.event_time, hours=24)
        ]
        if len(declines) >= DECLINE_BURST_THRESHOLD:
            flags.append({
                "type": "decline_burst",
                "confidence": min(0.95, 0.6 + 0.1 * (len(declines) - DECLINE_BURST_THRESHOLD)),
                "evidence": [r["event_id"] for r in declines] + [event.event_id],
                "detail": f"{len(declines) + 1} declines in 24h",
            })

        # accelerated withdrawals vs baseline z-score
        bl = baseline.get("large_withdrawal_count_7d")
        if bl and event.source == "core_banking_ledger" and event.event_type == "withdrawal":
            current = _count_recent(recent, "core_banking_ledger", "withdrawal", event.event_time, days=7) + 1
            z = zscore(float(current), bl)
            if z is not None and z >= ZSCORE_FLAG:
                flags.append({
                    "type": "accelerated_withdrawals",
                    "confidence": min(0.9, 0.5 + 0.1 * z),
                    "evidence": [event.event_id],
                    "detail": f"withdrawal count z={z:.2f} vs baseline",
                })

        # large/unusual amount on the triggering event itself
        amount = event.payload.get("amount") if isinstance(event.payload, dict) else None
        if amount is not None and float(amount) >= 5000:
            flags.append({
                "type": "large_amount",
                "confidence": 0.7,
                "evidence": [event.event_id],
                "detail": f"amount {amount} crosses large-transaction threshold",
            })
        return flags


def _within(ref_time: str, event_time, hours: float) -> bool:
    from datetime import datetime

    if isinstance(ref_time, str):
        ref = datetime.fromisoformat(ref_time.replace("Z", "+00:00"))
    else:
        ref = ref_time
    et = event_time if event_time.tzinfo else event_time.replace(tzinfo=ref.tzinfo)
    return abs((et - ref).total_seconds()) <= hours * 3600


def _count_recent(recent: list[dict], source: str, event_type: str, event_time, days: float) -> int:

    et = event_time if event_time.tzinfo else event_time
    count = 0
    for r in recent:
        if r.get("source_system") != source or r.get("event_type") != event_type:
            continue
        rt = r.get("event_time")
        if isinstance(rt, str):
            from datetime import datetime
            rt = datetime.fromisoformat(rt.replace("Z", "+00:00"))
        if rt and abs((et - rt).total_seconds()) <= days * 86400:
            count += 1
    return count
