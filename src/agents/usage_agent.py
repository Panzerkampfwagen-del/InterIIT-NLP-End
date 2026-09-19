"""Usage agent (Phase 5) - statistical engagement detection (RQ-L1).

Detects engagement-pattern shifts vs baseline: login-frequency drop
(negative z-score on login_count_7d) and search spikes (intent signals).
Detection is deterministic; the LLM only narrates flagged findings.
"""
from __future__ import annotations

from typing import Any

from src.agents.base import SignalAgent
from src.ingestion.schemas.events import EventEnvelope
from src.streaming.baselines import zscore

ENGAGEMENT_DROP_Z = -2.0  # negative z: current logins well below baseline
SEARCH_SPIKE_THRESHOLD = 3


class UsageAgent(SignalAgent):
    name = "usage_agent"
    domain = "usage"

    def detect(self, event: EventEnvelope, state) -> list[dict[str, Any]]:
        if event.source != "web_app_events":
            return []
        flags: list[dict[str, Any]] = []
        baseline = (state.historical_baseline or {}) if state is not None else {}
        recent = getattr(state, "recent_events", []) or []

        # engagement drop: current 7d login count well below baseline mean
        bl = baseline.get("login_count_7d")
        if bl and event.event_type == "login":
            current = 1 + sum(
                1 for r in recent
                if r.get("source_system") == "web_app_events" and r.get("event_type") == "login"
                and _within(r.get("event_time"), event.event_time, days=7)
            )
            z = zscore(float(current), bl)
            if z is not None and z <= ENGAGEMENT_DROP_Z:
                flags.append({
                    "type": "engagement_drop",
                    "confidence": min(0.85, 0.5 + 0.1 * abs(z)),
                    "evidence": [event.event_id],
                    "detail": f"login count z={z:.2f} vs baseline",
                })

        # search spike: intent signal (e.g. hardship-plan searches)
        if event.event_type == "search_query":
            searches = [
                r for r in recent
                if r.get("source_system") == "web_app_events" and r.get("event_type") == "search_query"
                and _within(r.get("event_time"), event.event_time, days=7)
            ]
            if len(searches) + 1 >= SEARCH_SPIKE_THRESHOLD:
                flags.append({
                    "type": "search_spike",
                    "confidence": 0.6,
                    "evidence": [r["event_id"] for r in searches] + [event.event_id],
                    "detail": f"{len(searches) + 1} searches in 7d",
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
