"""Phase 4: event-time consumer - dedup, lateness, recompute, watermark."""
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
def consumer():
    from src.state.board import CustomerStateBoard
    from src.streaming.consumer import EventTimeConsumer
    from src.streaming.watermarks import WatermarkTracker

    b = CustomerStateBoard()
    b.init_schema()
    t = WatermarkTracker()
    t.init_schema()
    with b._connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE customer_state, state_episodic, state_conflicts, watermarks RESTART IDENTITY")
    c = EventTimeConsumer(b, t, allowed_lateness_s=3600.0)
    yield c
    b.close()
    t.close()


T0 = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)


def _ev(i, et, source, event_type, payload=None, cid="C1"):
    from src.ingestion.schemas.events import EventEnvelope

    return EventEnvelope.model_validate(
        {
            "event_id": f"EVT_{i:06d}",
            "event_time": et,
            "ingestion_time": et,
            "customer_id": cid,
            "account_id": "ACC_CHK_001",
            "source_system": source,
            "event_type": event_type,
            "schema_version": "1.0",
            "payload": payload or {},
        }
    )


def test_late_event_within_lateness_recomputes(consumer) -> None:
    """Late event within allowed lateness: accepted + recomputed + supersede."""
    events = [
        _ev(1, T0, "core_banking_ledger", "deposit", {"amount": 100, "balance_after": 1}),
        _ev(2, T0 + timedelta(days=5), "core_banking_ledger", "deposit", {"amount": 200, "balance_after": 2}),
        # late: 2 days behind max, within 1h lateness? NO -> too late. Use minutes:
        _ev(3, T0 + timedelta(days=5) - timedelta(minutes=30), "core_banking_ledger", "deposit", {"amount": 300, "balance_after": 3}),
    ]
    results = consumer.process_batch(events)
    outcomes = [r.outcome.value for r in results]
    assert outcomes[0] == "emitted"
    assert outcomes[1] == "emitted"
    assert outcomes[2] == "late_recomputed"
    assert results[2].features is not None
    # baseline written to the state board with recomputed features
    snap = consumer.board.get("C1", apply_dec=False)
    assert snap.historical_baseline.get("txn_count_30d") == 3.0
    assert snap.historical_baseline.get("txn_amount_sum_30d") == 600.0
    # episodic store holds the superseded prior baseline write
    with consumer.board._connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM state_episodic WHERE customer_id='C1' AND field_path='historical_baseline'")
            assert cur.fetchone()[0] == 1  # recomputed baseline written as a new episodic row


def test_too_late_event_dropped(consumer) -> None:
    """Events older than allowed lateness are dropped (counted, not processed)."""
    events = [
        _ev(1, T0 + timedelta(days=10), "core_banking_ledger", "deposit", {"amount": 100}),
        _ev(2, T0 - timedelta(days=1), "core_banking_ledger", "deposit", {"amount": 200}),  # 11 days late > 1h
    ]
    results = consumer.process_batch(events)
    assert results[0].outcome.value == "emitted"
    assert results[1].outcome.value == "too_late_dropped"


def test_duplicate_never_double_counts(consumer) -> None:

    events = [
        _ev(1, T0, "core_banking_ledger", "deposit", {"amount": 100}),
        _ev(1, T0, "core_banking_ledger", "deposit", {"amount": 100}),
    ]
    results = consumer.process_batch(events)
    assert [r.outcome.value for r in results] == ["emitted", "duplicate"]


def test_watermark_lag_observable(consumer) -> None:
    """Watermark lag reflects bounded out-of-orderness (Phase 11 timing)."""
    consumer.process_batch([
        _ev(1, T0, "core_banking_ledger", "deposit", {"amount": 100}),
        _ev(2, T0 + timedelta(hours=2), "card_payments", "purchase", {"amount": 10}),
    ])
    lag = consumer.tracker.lag(now=T0 + timedelta(hours=2))
    assert lag > 0  # observable and positive


def test_idle_source_does_not_stall_watermark(consumer) -> None:
    """A source silent for > idle timeout is excluded (Flink idle semantics)."""

    consumer.process_batch(
        [_ev(1, T0, "core_banking_ledger", "deposit", {"amount": 100})],
        processing_time=T0,
    )
    # card_payments never produced; simulate: ledger source silent for 10s (idle timeout 5s)
    idle = consumer.tracker.mark_idle_sources(now=T0 + timedelta(seconds=10))
    assert "core_banking_ledger" in idle
    # with all sources idle, watermark is -inf equivalent (nothing considered late)
    wm = consumer.tracker.watermark(now=T0 + timedelta(seconds=10))
    assert wm == float("-inf")
