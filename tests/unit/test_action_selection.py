"""Phase 8: action selection - zero silence, bounded enum, eligibility gating."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.action.selector import ALLOWED_ACTIONS, Action, ActionSelector, confidence_band
from src.state.models import Finding, Provenance, StateSnapshot

NOW = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)


def _snapshot(state="no_signal", conf=0.3, findings=None, eligibility=None, interventions=None) -> StateSnapshot:
    return StateSnapshot(
        customer_id="C1",
        current_state={"life_phase": state, "life_phase_confidence": conf},
        findings=findings or {},
        eligibility=eligibility or {"customer_value_tier": "mid"},
        past_interventions=interventions or [],
    )


def test_bounded_enum_values() -> None:
    """The action enum is exactly the dataset's fixed values."""
    assert ALLOWED_ACTIONS == {
        "no_action", "proactive_retention_outreach", "relationship_manager_escalation",
        "personalized_offer", "support_intervention", "compliance_fraud_hold",
    }


def test_zero_silence_every_cycle_produces_decision() -> None:
    """Zero-silence: every decision cycle emits an action or explicit NO_ACTION."""
    selector = ActionSelector()
    states = [
        ("no_signal", 0.3), ("stable", 0.4), ("medical_hardship", 0.7),
        ("churn_risk", 0.8), ("potential_fraud_or_takeover", 0.9),
        ("new_child_life_event", 0.75), ("financial_distress_general", 0.65),
        ("unknown_future_state", 0.2),
    ]
    for state, conf in states:
        d = selector.select(_snapshot(state=state, conf=conf), now=NOW)
        assert d.action in ALLOWED_ACTIONS, f"unbounded action for {state}: {d.action}"
        assert d.reasoning_summary, "decision must carry reasoning (never silence)"
        assert 0.0 <= d.action_confidence <= 1.0


def test_compliance_hold_blocks_offer_and_forces_hold() -> None:
    """Policy enforcement (#21): no offers under an active compliance hold."""
    selector = ActionSelector()
    findings = {
        "compliance": Finding(
            summary="watchlist hit", confidence=1.0, as_of=NOW,
            provenance=Provenance(agent="kyc_agent"),
            flags=[{"type": "watchlist_hit"}],
        )
    }
    snap = _snapshot(state="new_child_life_event", conf=0.9, findings=findings, eligibility={"customer_value_tier": "high"})
    d = selector.select(snap, now=NOW)
    assert d.action == Action.COMPLIANCE_FRAUD_HOLD.value
    assert d.eligibility["compliance_hold"] is True
    assert d.eligibility["personalized_offer"] is False  # offer gated OFF


def test_high_value_churn_gets_rm_escalation() -> None:
    selector = ActionSelector()
    d = selector.select(_snapshot(state="churn_risk", conf=0.8, eligibility={"customer_value_tier": "high"}), now=NOW)
    assert d.action == Action.RELATIONSHIP_MANAGER_ESCALATION.value


def test_low_value_churn_gets_retention_outreach() -> None:
    selector = ActionSelector()
    d = selector.select(_snapshot(state="churn_risk", conf=0.6, eligibility={"customer_value_tier": "mid"}), now=NOW)
    assert d.action == Action.PROACTIVE_RETENTION_OUTREACH.value


def test_recent_identical_intervention_holds() -> None:
    """Churn risk but a recent identical intervention -> explicit NO_ACTION."""
    selector = ActionSelector()
    from src.state.models import Intervention

    interventions = [
        Intervention(decision_id="dec_1", action="proactive_retention_outreach", outcome="no_response", closed_at=NOW - timedelta(days=3)),
    ]
    d = selector.select(_snapshot(state="churn_risk", conf=0.7, eligibility={"customer_value_tier": "mid"}, interventions=interventions), now=NOW)
    assert d.action == Action.NO_ACTION.value
    assert "intervention" in d.reasoning_summary


def test_medical_hardship_support_intervention() -> None:
    selector = ActionSelector()
    findings = {
        "support": Finding(
            summary="hospital payment plan request", confidence=0.8, as_of=NOW,
            provenance=Provenance(agent="support_agent"), flags=[{"type": "negative_sentiment"}],
        )
    }
    d = selector.select(_snapshot(state="medical_hardship", conf=0.8, findings=findings), now=NOW)
    assert d.action == Action.SUPPORT_INTERVENTION.value


def test_life_event_offer_when_eligible() -> None:
    selector = ActionSelector()
    d = selector.select(_snapshot(state="new_child_life_event", conf=0.8, eligibility={"customer_value_tier": "mid"}), now=NOW)
    assert d.action == Action.PERSONALIZED_OFFER.value


def test_life_event_no_offer_for_low_tier() -> None:
    selector = ActionSelector()
    d = selector.select(_snapshot(state="new_child_life_event", conf=0.8, eligibility={"customer_value_tier": "low"}), now=NOW)
    assert d.action == Action.NO_ACTION.value  # tier gate


def test_confidence_band_mapping() -> None:
    assert confidence_band(0.9) == "high"
    assert confidence_band(0.7) == "high"
    assert confidence_band(0.5) == "medium"
    assert confidence_band(0.45) == "medium"
    assert confidence_band(0.2) == "low"


def test_decision_carries_evidence_from_findings() -> None:
    selector = ActionSelector()
    findings = {
        "transaction": Finding(
            summary="decline burst", confidence=0.9, as_of=NOW,
            provenance=Provenance(agent="transaction_agent"),
            flags=[{"type": "decline_burst"}], evidence_refs=["EVT_1", "EVT_2"],
        )
    }
    d = selector.select(_snapshot(state="potential_fraud_or_takeover", conf=0.85, findings=findings), now=NOW)
    assert d.action == Action.COMPLIANCE_FRAUD_HOLD.value
    assert len(d.evidence) >= 1
    assert d.evidence[0]["agent"] == "transaction_agent"
    assert d.evidence[0]["refs"] == ["EVT_1", "EVT_2"]
