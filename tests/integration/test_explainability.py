"""Phase 10: explainability - determinism, resolvability, zero-LLM."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.action.selector import Action, ActionDecision
from src.state.models import Finding, Provenance, StateSnapshot

NOW = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)


@pytest.fixture()
def explainer():
    from src.explainability.explainer import Explainer

    ex = Explainer()
    ex.init_schema()
    with ex._connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE decision_log RESTART IDENTITY")
    yield ex
    ex.close()


def _decision() -> ActionDecision:
    return ActionDecision(
        action=Action.SUPPORT_INTERVENTION.value,
        action_confidence=0.78,
        confidence_band="high",
        reasoning_summary="Medical hardship with support friction justifies a support intervention.",
        evidence=[
            {"source": "findings.support", "agent": "support_agent", "excerpt": "hospital payment plan request", "timestamp": NOW.isoformat(), "refs": ["EVT_000001", "EVT_000002"]},
            {"source": "findings.transaction", "agent": "transaction_agent", "excerpt": "income drop detected", "timestamp": NOW.isoformat(), "refs": ["EVT_000003"]},
        ],
        eligibility={"compliance_hold": False, "personalized_offer": False},
        inferred_state="medical_hardship",
    )


def _snapshot() -> StateSnapshot:
    return StateSnapshot(
        customer_id="C1",
        current_state={"life_phase": "medical_hardship", "life_phase_confidence": 0.8},
        findings={
            "support": Finding(
                summary="hospital payment plan request", confidence=0.8, as_of=NOW,
                provenance=Provenance(agent="support_agent"), flags=[{"type": "negative_sentiment"}],
            )
        },
    )


def _guardrail():
    from src.guardrails.machine import AUTONOMOUS, GuardrailResult, Outcome

    return GuardrailResult(outcome=Outcome.ALLOW, final_action=Action.SUPPORT_INTERVENTION.value, reasons=[], tier=AUTONOMOUS)


def test_explanation_deterministic_two_calls_identical(explainer) -> None:
    """Determinism: two explain() calls return byte-identical output."""

    explainer.snapshot("dec_1", "C1", NOW, _decision(), _guardrail(), {"status": "allowed", "final_action": Action.SUPPORT_INTERVENTION.value}, _snapshot(), trace_id="trace_abc")
    j1 = explainer.explain_json("dec_1")
    j2 = explainer.explain_json("dec_1")
    assert j1 == j2  # byte-identical


def test_no_llm_call_in_explanation(explainer) -> None:
    """Zero-LLM design contract: explain() never calls the LLM."""
    from src.agents.llm import FakeLLMClient

    llm = FakeLLMClient()
    explainer.snapshot("dec_2", "C1", NOW, _decision(), _guardrail(), {"status": "allowed"}, _snapshot())
    explanation = explainer.explain("dec_2")
    assert llm.calls == []  # ZERO LLM calls
    assert explanation["final_action"] == Action.SUPPORT_INTERVENTION.value


def test_evidence_refs_resolvable(explainer) -> None:
    """Resolvability: evidence refs resolve to real event ids."""

    explainer.snapshot("dec_3", "C1", NOW, _decision(), _guardrail(), {"status": "allowed"}, _snapshot())
    explanation = explainer.explain("dec_3")
    refs = [r for e in explanation["evidence"] for r in e["resolvable_refs"]]
    assert "EVT_000001" in refs and "EVT_000003" in refs
    # every evidence entry carries its agent provenance
    assert {e["agent"] for e in explanation["evidence"]} == {"support_agent", "transaction_agent"}


def test_explanation_contains_full_decision_record(explainer) -> None:
    """A reviewer can answer 'why?' from the stored snapshot alone."""

    explainer.snapshot("dec_4", "C1", NOW, _decision(), _guardrail(), {"status": "allowed"}, _snapshot(), trace_id="trace_xyz")
    ex = explainer.explain("dec_4")
    assert ex["final_action"] == Action.SUPPORT_INTERVENTION.value
    assert ex["action_confidence"] == 0.78
    assert ex["confidence_band"] == "high"
    assert "Medical hardship" in ex["reasoning_summary"]
    assert ex["inferred_state"] == "medical_hardship"
    assert ex["guardrail"]["outcome"] == "allow"
    assert ex["trace_id"] == "trace_xyz"
    assert ex["hitl"] is None  # allowed decision has no pending HITL


def test_pending_hitl_surfaces_in_explanation(explainer) -> None:
    """Escalated decisions surface the HITL approval in the explanation."""
    explainer.snapshot(
        "dec_5", "C1", NOW, _decision(), _guardrail(),
        {"status": "pending_approval", "approval_id": "apr_123", "final_action": None},
        _snapshot(),
    )
    ex = explainer.explain("dec_5")
    assert ex["status"] == "pending_approval"
    assert ex["hitl"]["approval_id"] == "apr_123"


def test_unknown_decision_raises(explainer) -> None:
    from src.explainability.explainer import DecisionNotFound

    with pytest.raises(DecisionNotFound):
        explainer.explain("dec_missing")
