"""Phase 1: replay simulator impairment assertions.

The simulator is SIMULATION ONLY; these tests verify the impairment knobs do
what they claim (late, duplicate, out-of-order) and that a clean replay
preserves event-time order and publishes every event exactly once.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.ingestion.dataset_loader import Scenario
from src.ingestion.replay_simulator import ReplayConfig, ReplaySimulator
from src.ingestion.schemas.events import EventEnvelope


def _scenario(n_events: int = 20) -> Scenario:
    events: list[EventEnvelope] = []
    start = datetime(2026, 5, 1, tzinfo=UTC)
    for i in range(n_events):
        events.append(
            EventEnvelope.model_validate(
                {
                    "event_id": f"GEVT_{i:06d}",
                    "event_time": start + timedelta(hours=i),
                    "ingestion_time": start + timedelta(hours=i),
                    "customer_id": "CUST_90001",
                    "account_id": "ACC_CHK_001",
                    "source_system": "core_banking_ledger",
                    "event_type": "deposit",
                    "schema_version": "1.0",
                    "payload": {"amount": 100 + i, "balance_after": 1000 + i, "transaction_type": "salary_credit"},
                }
            )
        )
    return Scenario(
        scenario_id="test_scenario",
        entities={"customer_id": "CUST_90001", "profile": {}, "accounts": []},
        history_seed=(),
        live_stream=tuple(events),
        replay_config={},
    )


def _producer():
    from src.ingestion.producers.memory_producer import InMemoryEventProducer

    return InMemoryEventProducer()


def test_clean_replay_publishes_all_in_order() -> None:
    producer = _producer()
    report = ReplaySimulator(producer, _scenario(), ReplayConfig()).run()
    assert report.published == 20
    assert report.duplicates_injected == 0
    assert report.late_injected == 0
    published = producer.events
    assert [e.event_id for e in published] == [f"GEVT_{i:06d}" for i in range(20)]
    # event_time order preserved
    times = [e.event_time for e in published]
    assert times == sorted(times)


def test_late_injection_marks_ingestion_time_after_event_time() -> None:
    producer = _producer()
    config = ReplayConfig(late_rate=1.0, late_by_seconds=600.0, seed=7)
    report = ReplaySimulator(producer, _scenario(10), config).run()
    assert report.late_injected == 10
    for e in producer.events:
        assert e.ingestion_time is not None and e.ingestion_time > e.event_time
        assert e.is_late()


def test_duplicate_injection_publishes_twice() -> None:
    producer = _producer()
    config = ReplayConfig(duplicate_rate=1.0, seed=7)
    report = ReplaySimulator(producer, _scenario(10), config).run()
    assert report.published == 20
    assert report.duplicates_injected == 10
    assert len(producer.published) == 20
    # same event ids appear (in pairs)
    ids = [e.event_id for e in producer.events]
    assert len(ids) == 20 and len(set(ids)) == 10


def test_out_of_order_injection_breaks_event_time_order() -> None:
    producer = _producer()
    config = ReplayConfig(out_of_order_rate=1.0, seed=7)
    report = ReplaySimulator(producer, _scenario(10), config).run()
    assert report.out_of_order_pairs > 0
    times = [e.event_time for e in producer.events]
    assert times != sorted(times)  # order actually broken


def test_determinism_given_seed() -> None:
    r1 = ReplaySimulator(_producer(), _scenario(30), ReplayConfig(late_rate=0.3, duplicate_rate=0.2, out_of_order_rate=0.2, seed=123)).run()
    r2 = ReplaySimulator(_producer(), _scenario(30), ReplayConfig(late_rate=0.3, duplicate_rate=0.2, out_of_order_rate=0.2, seed=123)).run()
    assert (r1.published, r1.duplicates_injected, r1.late_injected, r1.out_of_order_pairs) == (
        r2.published, r2.duplicates_injected, r2.late_injected, r2.out_of_order_pairs,
    )
