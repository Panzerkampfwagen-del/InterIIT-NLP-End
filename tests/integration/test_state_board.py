"""Phase 3: state board tests - CAS, decay, conflict, reconstruction.

Definition of Done (Phase 3): CAS + decay + conflict tests pass and a past
decision's state is reproducible from the episodic history.
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
def board():
    from src.state.board import CustomerStateBoard

    b = CustomerStateBoard()
    b.init_schema()
    with b._connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE customer_state, state_episodic, state_conflicts RESTART IDENTITY")
    yield b
    b.close()


T0 = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)


def _prov(agent="life_event_agent", decision_id="dec_1"):
    from src.state.models import Provenance

    return Provenance(agent=agent, decision_id=decision_id)


def test_cas_version_conflict_detected(board) -> None:
    """Optimistic concurrency: stale version raises VersionConflict."""
    from src.state.board import VersionConflict

    board.ensure_customer("C1")
    v0 = board.version("C1")
    board.update_field("C1", "current_state.life_phase", "stable", 0.9, _prov(), T0, expected_version=v0)
    with pytest.raises(VersionConflict):
        board.update_field(
            "C1", "current_state.life_phase", "distress", 0.8, _prov(decision_id="dec_2"),
            T0, expected_version=v0,  # stale - version already bumped
        )
    # retry with fresh version succeeds
    v1 = board.version("C1")
    board.update_field("C1", "current_state.life_phase", "distress", 0.8, _prov(decision_id="dec_2"), T0, expected_version=v1)
    snap = board.get("C1", apply_dec=False)
    assert snap.current_state["life_phase"] == "distress"


def test_update_bumps_version_each_write(board) -> None:
    board.ensure_customer("C2")
    v0 = board.version("C2")
    board.update_field("C2", "current_state.life_phase", "stable", 0.9, _prov(), T0)
    v1 = board.version("C2")
    board.update_field(
        "C2", "findings.transaction",
        {"summary": "ok", "confidence": 0.6, "as_of": T0.isoformat(), "provenance": {"agent": "transaction_agent"}},
        0.6, _prov(), T0,
    )
    v2 = board.version("C2")
    assert v1 == v0 + 1 and v2 == v1 + 1


def test_episodic_supersede_preserves_history(board) -> None:
    """Superseding writes set valid_to/superseded_by; prior rows never deleted."""
    board.ensure_customer("C3")
    board.update_field("C3", "current_state.life_phase", "stable", 0.9, _prov(), T0)
    board.update_field("C3", "current_state.life_phase", "financial_distress", 0.74, _prov(decision_id="dec_2"), T0 + timedelta(days=10))
    with board._connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT episode_id, value, valid_to, superseded_by FROM state_episodic WHERE customer_id = 'C3' AND field_path = 'current_state.life_phase' ORDER BY episode_id"
            )
            rows = cur.fetchall()
    assert len(rows) == 2
    first, second = rows
    assert first[3] == second[0]  # prior row superseded_by -> new episode id
    assert first[2] is not None  # prior row has valid_to set (not deleted)
    assert second[2] is None  # current row still valid


def test_lazy_decay_drops_aged_fields(board) -> None:
    """Decay computed at read time: aged life_phase falls below threshold and drops."""
    board.ensure_customer("C4")
    old = T0 - timedelta(days=200)
    board.update_field("C4", "current_state.life_phase", "relocating", 0.6, _prov(), old)
    snap = board.get("C4", apply_dec=True, now=T0)  # 200 days later
    assert snap.current_state.get("life_phase") is None  # decayed out
    # ... but the episodic history still holds it
    with board._connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM state_episodic WHERE customer_id = 'C4'")
            assert cur.fetchone()[0] == 1


def test_decay_respects_per_field_half_life(board) -> None:
    from src.state.decay import decayed_confidence

    # 30-day half-life, 30 days age -> exactly half
    assert abs(decayed_confidence(0.8, 30, 30) - 0.4) < 1e-9
    # 14-day half-life (churn risk per docs), 28 days age -> quarter
    assert abs(decayed_confidence(0.8, 28, 14) - 0.2) < 1e-9


def test_conflict_log_preserves_both_values(board) -> None:
    """Usage agent says stable, transaction agent says distress -> both kept."""
    board.ensure_customer("C5")
    board.update_field("C5", "current_state.life_phase", "stable", 0.6, _prov(agent="usage_agent"), T0)
    board.update_field(
        "C5", "current_state.life_phase", "financial_distress", 0.9,
        _prov(agent="transaction_agent"), T0, conflict_with="usage_agent",
    )
    pending = board.pending_conflicts("C5")
    assert len(pending) == 1
    c = pending[0]
    assert set(c.agents) == {"usage_agent", "transaction_agent"}
    assert c.values[0] == "stable" and c.values[1] == "financial_distress"
    # resolution keeps the losing value visible in the log (never deleted)
    board.resolve_conflict(1, resolved_by="critique_agent", resolution="financial_distress")
    assert board.pending_conflicts("C5") == []
    with board._connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT resolved_by, resolution FROM state_conflicts WHERE conflict_id = 1")
            resolved_by, resolution = cur.fetchone()
    assert resolved_by == "critique_agent"


def test_reconstruct_past_state_reproducible(board) -> None:
    """Given an as_of instant, the exact past state is reconstructable (#30)."""
    board.ensure_customer("C6")
    board.update_field("C6", "current_state.life_phase", "stable", 0.9, _prov(), T0)
    board.update_field(
        "C6", "findings.transaction",
        {"summary": "baseline", "confidence": 0.5, "as_of": T0.isoformat(), "provenance": {"agent": "transaction_agent"}},
        0.5, _prov(), T0,
    )
    later = T0 + timedelta(days=15)
    board.update_field("C6", "current_state.life_phase", "financial_distress", 0.74, _prov(decision_id="dec_881"), later)

    # reconstruct at a point AFTER first writes but BEFORE the later write
    mid = T0 + timedelta(days=5)
    past = board.reconstruct("C6", mid)
    assert past.current_state["life_phase"] == "stable"
    assert past.findings["transaction"].summary == "baseline"
    # reconstruct at now -> latest
    now_state = board.reconstruct("C6", T0 + timedelta(days=20))
    assert now_state.current_state["life_phase"] == "financial_distress"


def test_ring_buffer_bounded_and_deduped(board) -> None:
    board.ensure_customer("C7")
    for i in range(60):
        board.append_event_ref("C7", {"event_id": f"EVT_{i:06d}", "event_time": T0.isoformat()})
    snap = board.get("C7", apply_dec=False)
    assert len(snap.recent_events) == 50
    assert snap.recent_events[-1]["event_id"] == "EVT_000059"
    # duplicate append does not duplicate the entry
    board.append_event_ref("C7", {"event_id": "EVT_000059", "event_time": T0.isoformat()})
    snap = board.get("C7", apply_dec=False)
    assert len(snap.recent_events) == 50
    assert len({e["event_id"] for e in snap.recent_events}) == 50
