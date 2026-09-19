"""Deterministic eligibility engine (Phase 8).

Business/compliance rules computed deterministically from the state board:
- ``compliance_hold``: active fraud hold (watchlist hit / fraud inference);
- ``personalized_offer``: value tier + no compliance hold + offer policies;
- ``retention_outreach``: no open duplicate intervention recently.

These rules are POLICY ENFORCEMENT (#21) - evaluated in code, never in a prompt.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.state.models import StateSnapshot

FRAUD_STATES = frozenset({"potential_fraud_or_takeover", "elder_vulnerability_or_scam_risk"})
OFFER_ELIGIBLE_STATES = frozenset({"new_child_life_event", "wealth_growth_or_windfall", "retirement_transition", "marriage_or_relationship_change"})
RECENT_INTERVENTION_WINDOW_DAYS = 14.0


def compute_eligibility(snapshot: StateSnapshot, now: datetime | None = None) -> dict[str, Any]:
    """Compute the ``eligibility`` block deterministically from state."""
    now = now or datetime.now(UTC)
    state = snapshot.current_state.get("life_phase")
    confidence = float(snapshot.current_state.get("life_phase_confidence") or 0.0)

    compliance_flags = {fl.get("type") for fl in (snapshot.findings.get("compliance").flags if snapshot.findings.get("compliance") else [])}
    active_fraud_hold = (
        state in FRAUD_STATES and confidence >= 0.6
    ) or "watchlist_hit" in compliance_flags

    # offer eligibility: eligible life-event state, no fraud hold, value tier not low
    tier = (snapshot.eligibility or {}).get("customer_value_tier", snapshot.historical_baseline.get("customer_value_tier", "mid"))
    offer_ok = (
        not active_fraud_hold
        and state in OFFER_ELIGIBLE_STATES
        and confidence >= 0.5
        and tier != "low"
    )

    # retention outreach: churn risk present, no very recent identical intervention
    recent_intervention = any(
        (now - i.closed_at).days <= RECENT_INTERVENTION_WINDOW_DAYS
        for i in snapshot.past_interventions
        if i.closed_at is not None
    )
    retention_ok = (state == "churn_risk" and confidence >= 0.5) and not recent_intervention

    return {
        "compliance_hold": active_fraud_hold,
        "personalized_offer": offer_ok,
        "retention_outreach": retention_ok,
        "customer_value_tier": tier,
        "recent_intervention": recent_intervention,
        "computed_at": now.isoformat(),
    }
