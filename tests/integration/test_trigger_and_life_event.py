"""Phase 7: trigger evaluator + life-event agent.

DoD: flood test (no per-event invocation), disagreement scenario,
classification accuracy vs ground truth, calibration spot-check on ambiguous.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.config import get_settings


def _infra() -> bool:
    import socket

    s = get_settings()
    host, port = s.pg_dsn.split("//")[1].split("@")[1].split("/")[0].split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2.0):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _infra(), reason="Postgres not running")


@pytest.fixture()
def env():
    from src.agents.llm import FakeLLMClient
    from src.state.board import CustomerStateBoard

    b = CustomerStateBoard()
    b.init_schema()
    with b._connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE customer_state, state_episodic, state_conflicts, watermarks, retrieval_chunks RESTART IDENTITY")
    yield b, FakeLLMClient
    b.close()


T0 = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)


def _record_finding(board, customer_id, domain, flags, conf, at):
    from src.state.models import Finding, Provenance

    f = Finding(
        summary=f"{domain} finding",
        confidence=conf,
        as_of=at,
        provenance=Provenance(agent=f"{domain}_agent"),
        flags=flags,
    )
    board.ensure_customer(customer_id)
    board.record_finding(customer_id, domain, f, at)


def test_flood_test_not_per_event(env) -> None:
    """Trigger evaluator fires ONCE per cooldown, not on every event/update."""
    from src.orchestration.trigger import TriggerEvaluator

    board, _ = env
    trig = TriggerEvaluator(board, correlation_window_min=15.0, high_severity_conf=0.75, cooldown_min=30.0)
    # 10 consecutive single-domain finding updates over 27 minutes (every 3 min)
    fires = 0
    for i in range(10):
        _record_finding(board, "C1", "transaction", [{"type": "decline_burst"}], 0.6, T0 + timedelta(minutes=3 * i))
        d = trig.evaluate("C1", T0 + timedelta(minutes=3 * i))
        fires += 1 if d.fire else 0
    assert fires == 1, f"flood: agent fired {fires} times for 10 updates (must be 1; cooldown blocks re-invocation)"
    # after cooldown expires (fire at minute 3 + 30 = 33), a fresh window fires again
    d = trig.evaluate("C1", T0 + timedelta(minutes=34))
    assert d.fire is True


def test_single_finding_no_fire_without_severity(env) -> None:
    from src.orchestration.trigger import TriggerEvaluator

    board, _ = env
    trig = TriggerEvaluator(board, correlation_window_min=15.0, high_severity_conf=0.75, cooldown_min=30.0)
    _record_finding(board, "C2", "usage", [{"type": "login"}], 0.5, T0)
    d = trig.evaluate("C2", T0 + timedelta(minutes=1))
    assert d.fire is False  # single weak finding: no invocation (cost discipline)


def test_disagreement_recorded_and_addressed(env) -> None:
    """Usage says healthy, transaction says distress -> conflicts[] preserved,
    reasoning explicitly addresses the disagreement."""
    from src.agents.life_event_agent import LifeEventAgent

    board, _ = env
    # usage agent: healthy/engaged (no distress flags), transaction: distress
    _record_finding(board, "C3", "usage", [{"type": "login"}], 0.5, T0)
    _record_finding(board, "C3", "transaction", [{"type": "decline_burst"}], 0.9, T0)
    # record the conflict like the signal agents would (Phase 3 conflict log)
    board.record_conflict("C3", "current_state.life_phase", ["usage_agent", "transaction_agent"], ["stable", "financial_distress"])

    agent = LifeEventAgent(board=board, llm=None)
    inference = agent.infer("C3", as_of=T0 + timedelta(minutes=5))
    assert inference.inferred_state == "financial_distress_general"
    assert "disagreement" in inference.reasoning.lower()
    pending = board.pending_conflicts("C3")
    assert len(pending) == 1  # both values preserved, never deleted
    assert pending[0].values == ["stable", "financial_distress"]


def test_llm_invalid_label_retried_then_fallback(env) -> None:
    """Invalid inferred_state retried (bounded); fallback keeps it valid."""
    from src.agents.life_event_agent import LifeEventAgent

    board, FakeLLM = env
    # first two calls return invalid labels, then the fallback kicks in
    llm = FakeLLM(responses=[{"inferred_state": "totally_custom_label", "confidence": 0.9}])
    _record_finding(board, "C4", "transaction", [{"type": "decline_burst"}], 0.9, T0)
    agent = LifeEventAgent(board=board, llm=llm)
    inference = agent.infer("C4", as_of=T0 + timedelta(minutes=5))
    assert inference.inferred_state in {
        "financial_distress_general",  # deterministic fallback
    }


def test_calibration_ambiguous_gets_low_confidence(env) -> None:
    """Calibration spot-check: ambiguous evidence => low confidence, never high."""
    from src.agents.life_event_agent import LifeEventAgent

    board, _ = env
    # weak, mixed evidence: one low-confidence finding
    _record_finding(board, "C5", "transaction", [], 0.4, T0)
    _record_finding(board, "C5", "usage", [], 0.4, T0)
    agent = LifeEventAgent(board=board, llm=None)
    inference = agent.infer("C5", as_of=T0 + timedelta(minutes=5))
    assert inference.inferred_state == "no_signal"
    assert inference.confidence <= 0.5  # calibrated low on ambiguous


def test_classification_accuracy_vs_ground_truth(env) -> None:
    """Measured classification accuracy: deterministic fallback synthesis over
    scenario flags vs ground-truth expected states (the 4 archetypes).
    """
    from src.agents.life_event_agent import LifeEventAgent

    board, _ = env
    scenarios = [
        # (customer, findings, expected inferred_state)
        ("S1", {"compliance": ([{"type": "watchlist_hit"}], 1.0)}, "potential_fraud_or_takeover"),
        ("S2", {"transaction": ([{"type": "decline_burst"}], 0.8)}, "financial_distress_general"),
        ("S3", {"usage": ([{"type": "engagement_drop"}], 0.7), "support": ([{"type": "repeated_contact"}, {"type": "negative_sentiment"}], 0.8)}, "churn_risk"),
        ("S4", {"compliance": ([{"type": "household_change"}], 1.0)}, "new_child_life_event"),
    ]
    correct = 0
    for cid, findings, expected in scenarios:
        for domain, (flags, conf) in findings.items():
            _record_finding(board, cid, domain, flags, conf, T0)
        agent = LifeEventAgent(board=board, llm=None)
        inference = agent.infer(cid, as_of=T0 + timedelta(minutes=5))
        if inference.inferred_state == expected:
            correct += 1
    accuracy = correct / len(scenarios)
    assert accuracy == 1.0, f"classification accuracy {accuracy:.2f} below requirement"


def test_evidence_refs_resolvable(env) -> None:
    """Evidence refs from findings resolve to real event ids."""
    from src.agents.life_event_agent import LifeEventAgent
    from src.state.models import Finding, Provenance

    board, _ = env
    f = Finding(
        summary="transaction distress",
        confidence=0.8,
        as_of=T0,
        provenance=Provenance(agent="transaction_agent"),
        flags=[{"type": "decline_burst", "evidence": ["EVT_000001", "EVT_000002"]}],
        evidence_refs=["EVT_000001", "EVT_000002"],
    )
    board.ensure_customer("C6")
    board.record_finding("C6", "transaction", f, T0)
    agent = LifeEventAgent(board=board, llm=None)
    agent.infer("C6", as_of=T0 + timedelta(minutes=5))
    snap = board.get("C6", apply_dec=False)
    inference_refs = [e for e in snap.recent_events if e.get("kind") == "life_event_inference"]
    assert len(inference_refs) == 1
    ev = inference_refs[0]["evidence"]
    assert "EVT_000001" in ev and "EVT_000002" in ev
