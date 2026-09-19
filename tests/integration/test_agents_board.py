"""Phase 5 integration: agents write findings to the shared state board;
baselines computed from the history seed feed statistical detection."""
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
            cur.execute("TRUNCATE customer_state, state_episodic, state_conflicts, watermarks RESTART IDENTITY")
    yield b
    b.close()


T0 = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)


def test_finding_written_with_provenance(board) -> None:
    from src.agents.kyc_agent import KycAgent

    agent = KycAgent(board=board, llm=None)
    from tests.unit.test_signal_agents import _ev, _state

    event = _ev(1, T0, "loan_kyc", "dependents_change", {"event_subtype": "dependents_change", "old_value": 1, "new_value": 2})
    finding = agent.analyze_and_record(event, _state(), event_time=T0)
    assert finding.domain if hasattr(finding, "domain") else True
    snap = board.get("C1", apply_dec=False)
    f = snap.findings.get("compliance")
    assert f is not None
    assert f.provenance.agent == "kyc_agent"
    assert f.provenance.event_id == "EVT_000001"
    assert f.confidence == 1.0
    assert any(fl["type"] == "household_change" for fl in f.flags)


def test_baselines_from_history_seed_drive_detection(board) -> None:
    """Baselines from history -> z-score detection flags the pattern shift."""
    from src.agents.transaction_agent import TransactionAgent
    from src.streaming.baselines import compute_baselines
    from src.streaming.features import DEFAULT_SPECS
    from tests.unit.test_signal_agents import _ev, _state

    # 30 days of stable deposits, a small withdrawal pair (baseline variance),
    # then a withdrawal burst in the live window
    history = [
        _ev(i, T0 - timedelta(days=i), "core_banking_ledger", "deposit", {"amount": 100, "balance_after": 1})
        for i in range(1, 31)
    ]
    history += [
        _ev(100, T0 - timedelta(days=20), "core_banking_ledger", "withdrawal", {"amount": 400, "balance_after": 1}),
        _ev(101, T0 - timedelta(days=19), "core_banking_ledger", "withdrawal", {"amount": 350, "balance_after": 1}),
    ]
    specs = tuple(s for s in DEFAULT_SPECS if s.name == "large_withdrawal_count_7d")
    baseline = compute_baselines(history, specs, history_end=T0)
    assert "large_withdrawal_count_7d" in baseline

    refs = [
        {"event_id": f"EVT_{i:06d}", "source_system": "core_banking_ledger", "event_type": "withdrawal", "event_time": (T0 - timedelta(days=i - 10)).isoformat()}
        for i in range(11, 13)
    ]
    state = _state(recent_refs=refs, baseline=baseline)
    agent = TransactionAgent(board=board, llm=None)
    event = _ev(20, T0, "core_banking_ledger", "withdrawal", {"amount": 3500})
    finding = agent.analyze_and_record(event, state, event_time=T0)
    assert any(fl["type"] == "accelerated_withdrawals" for fl in finding.flags)
    snap = board.get("C1", apply_dec=False)
    assert snap.findings["transaction"].provenance.agent == "transaction_agent"


def test_llm_failure_fallback_in_pipeline(board) -> None:
    """LLM unavailable => deterministic finding still written to the board."""
    from src.agents.llm import FakeLLMClient
    from src.agents.transaction_agent import TransactionAgent
    from tests.unit.test_signal_agents import _ev, _state

    agent = TransactionAgent(board=board, llm=FakeLLMClient(fail=True))
    event = _ev(1, T0, "ach_wire", "outbound_transfer", {"amount": 12000})
    finding = agent.analyze_and_record(event, _state(), event_time=T0)
    assert any(fl["type"] == "large_amount" for fl in finding.flags)
    snap = board.get("C1", apply_dec=False)
    assert "large_amount" in snap.findings["transaction"].summary
