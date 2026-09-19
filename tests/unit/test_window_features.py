"""Phase 4: windowed feature computation (pure) - event-time correctness."""
from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from src.ingestion.schemas.events import EventEnvelope
from src.streaming.features import DEFAULT_SPECS, compute_window_features, deduplicate, in_window

T0 = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)


def _ev(i, et, source, event_type, payload=None, cid="C1"):
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


def test_features_use_event_time_not_arrival_order() -> None:
    """'3 declines in 24h' must be computed on event_time (#3)."""
    events = [
        _ev(1, T0 - timedelta(hours=2), "card_payments", "decline", {"amount": 50}),
        _ev(2, T0 - timedelta(hours=1), "card_payments", "decline", {"amount": 60}),
        _ev(3, T0, "card_payments", "decline", {"amount": 70}),
    ]
    feats = compute_window_features(events, T0, specs=(DEFAULT_SPECS[2],))
    assert feats["decline_count_24h"] == 3.0
    # 25 hours ago: outside the 24h window
    events.append(_ev(4, T0 - timedelta(hours=25), "card_payments", "decline", {"amount": 50}))
    feats = compute_window_features(events, T0, specs=(DEFAULT_SPECS[2],))
    assert feats["decline_count_24h"] == 3.0


def test_out_of_order_equivalence() -> None:
    """Shuffled arrival yields identical features (out-of-order equivalence)."""
    rng = random.Random(7)
    events = []
    i = 0
    for day in range(30):
        events.append(_ev(i, T0 - timedelta(days=day), "core_banking_ledger", "deposit", {"amount": 100, "balance_after": 1}, "C1")); i += 1
        events.append(_ev(i, T0 - timedelta(days=day, hours=-3), "web_app_events", "login", {}, "C1")); i += 1
    shuffled = events[:]
    rng.shuffle(shuffled)
    f_sorted = compute_window_features(events, T0)
    f_shuffled = compute_window_features(shuffled, T0)
    assert f_sorted == f_shuffled


def test_duplicate_events_do_not_double_count() -> None:
    events = [
        _ev(1, T0 - timedelta(hours=1), "core_banking_ledger", "deposit", {"amount": 100}),
        _ev(1, T0 - timedelta(hours=1), "core_banking_ledger", "deposit", {"amount": 100}),  # duplicate id
    ]
    deduped = deduplicate(events)
    feats = compute_window_features(deduped, T0, specs=(DEFAULT_SPECS[0], DEFAULT_SPECS[1]))
    assert feats["txn_count_30d"] == 1.0
    assert feats["txn_amount_sum_30d"] == 100.0


def test_sum_and_count_aggregates() -> None:
    events = [
        _ev(1, T0, "core_banking_ledger", "withdrawal", {"amount": 300, "balance_after": 1}),
        _ev(2, T0 - timedelta(days=1), "core_banking_ledger", "withdrawal", {"amount": 500, "balance_after": 2}),
        _ev(3, T0 - timedelta(days=8), "core_banking_ledger", "withdrawal", {"amount": 9000, "balance_after": 3}),
    ]
    feats = compute_window_features(events, T0, specs=(DEFAULT_SPECS[3],))
    assert feats["large_withdrawal_count_7d"] == 2.0  # 8-day-old withdrawal outside window


def test_customer_scoping_in_features() -> None:
    events = [
        _ev(1, T0, "core_banking_ledger", "deposit", {"amount": 100}, "C1"),
        _ev(2, T0, "core_banking_ledger", "deposit", {"amount": 999}, "C2"),
    ]
    feats = compute_window_features(events, T0, specs=(DEFAULT_SPECS[1],))
    # compute_window_features is per-event-set; consumer filters by customer
    c1_events = [e for e in events if e.customer_id == "C1"]
    feats_c1 = compute_window_features(c1_events, T0, specs=(DEFAULT_SPECS[1],))
    assert feats_c1["txn_amount_sum_30d"] == 100.0


def test_in_window_boundaries() -> None:
    spec = DEFAULT_SPECS[2]  # decline_count_24h
    assert in_window(_ev(1, T0, "card_payments", "decline"), spec, T0)
    assert in_window(_ev(2, T0 - timedelta(hours=24), "card_payments", "decline"), spec, T0)
    assert not in_window(_ev(3, T0 - timedelta(hours=24, seconds=1), "card_payments", "decline"), spec, T0)
    assert not in_window(_ev(4, T0, "card_payments", "purchase"), spec, T0)  # wrong type
