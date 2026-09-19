"""Confidence decay (Phase 3, RQ-M2) - lazy at read time, no background job.

Per docs/reference/03 Section 8: confidence on inferred fields decays on a
per-field half-life unless reinforced by a new corroborating signal; decay is
computed lazily at read time as ``confidence * 0.5^(age_days / half_life_days)``
and a decayed-below-threshold field is dropped from current state but remains
in episodic history.
"""
from __future__ import annotations

from datetime import UTC, datetime

from src.state.models import StateSnapshot

DECAYED_FIELD_PATHS = ("current_state.life_phase", "active_risks", "active_opportunities")


def decayed_confidence(confidence: float, age_days: float, half_life_days: float) -> float:
    """Exponential half-life decay of a confidence value."""
    if half_life_days <= 0:
        return confidence
    return confidence * (0.5 ** (age_days / half_life_days))


def _age_days(since: datetime | str, now: datetime) -> float:
    if isinstance(since, str):  # JSONB round-trips timestamps as ISO strings
        since = datetime.fromisoformat(since.replace("Z", "+00:00"))
    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return max(0.0, (now - since).total_seconds() / 86400.0)


def _half_life(snapshot: StateSnapshot, path: str, default: float) -> float:
    meta = snapshot.decay_metadata.get(path)
    if meta and "half_life_days" in meta:
        return float(meta["half_life_days"])
    return default


def apply_decay(
    snapshot: StateSnapshot,
    now: datetime,
    default_half_life_days: float = 30.0,
    min_confidence: float = 0.10,
) -> StateSnapshot:
    """Return a copy of the snapshot with lazy decay applied to inferred fields.

    Fields decayed below ``min_confidence`` are dropped from the snapshot
    (they remain in the episodic history - never deleted).
    """
    data = snapshot.model_copy(deep=True)

    # life_phase
    cs = data.current_state
    if cs.get("life_phase") and cs.get("life_phase_confidence") is not None:
        valid_from = cs.get("life_phase_valid_from") or data.last_updated_event_time
        if valid_from is not None:
            hl = _half_life(snapshot, "life_phase", default_half_life_days)
            c = decayed_confidence(float(cs["life_phase_confidence"]), _age_days(valid_from, now), hl)
            if c < min_confidence:
                cs["life_phase"] = None
                cs["life_phase_confidence"] = 0.0
            else:
                cs["life_phase_confidence"] = round(c, 4)

    # active risks / opportunities
    def _decay_list(items: list, path: str) -> list:
        kept = []
        hl = _half_life(snapshot, path, default_half_life_days)
        for item in items:
            opened = item.opened_at
            c = decayed_confidence(item.confidence, _age_days(opened, now), hl)
            if c >= min_confidence:
                item.confidence = round(c, 4)
                kept.append(item)
        return kept

    data.active_risks = _decay_list(data.active_risks, "active_risks")
    data.active_opportunities = _decay_list(data.active_opportunities, "active_opportunities")
    return data
