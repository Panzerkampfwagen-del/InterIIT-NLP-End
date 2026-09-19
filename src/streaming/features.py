"""Windowed feature computation (Phase 4) - pure, event-time based.

Rolling-window features over validated envelopes (e.g. ``txn_count_30d``,
``decline_count_24h``, ``login_count_7d``) computed strictly on **event time**
(#3): a feature ``as_of`` T counts only events with
``T - window <= event_time <= T``. Pure functions so out-of-order equivalence
is directly testable (the same event set yields the same features regardless
of arrival order).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from src.ingestion.schemas.events import EventEnvelope


@dataclass(frozen=True)
class WindowSpec:
    """One rolling-window feature definition."""

    name: str
    window_days: float
    sources: frozenset[str]  # source_system values to include
    event_types: frozenset[str] | None = None  # None = all types from the sources
    aggregate: str = "count"  # count | sum | max
    payload_field: str | None = None  # required for sum/max


# The documented baseline features (docs/reference/03 Section 8 historical_baseline).
DEFAULT_SPECS: tuple[WindowSpec, ...] = (
    WindowSpec("txn_count_30d", 30.0, frozenset({"card_payments", "instant_payments", "ach_wire", "core_banking_ledger"})),
    WindowSpec("txn_amount_sum_30d", 30.0, frozenset({"card_payments", "instant_payments", "ach_wire", "core_banking_ledger"}), aggregate="sum", payload_field="amount"),
    WindowSpec("decline_count_24h", 1.0, frozenset({"card_payments"}), event_types=frozenset({"decline"})),
    WindowSpec("large_withdrawal_count_7d", 7.0, frozenset({"core_banking_ledger"}), event_types=frozenset({"withdrawal"})),
    WindowSpec("login_count_7d", 7.0, frozenset({"web_app_events"}), event_types=frozenset({"login"})),
    WindowSpec("search_count_7d", 7.0, frozenset({"web_app_events"}), event_types=frozenset({"search_query"})),
    WindowSpec("support_ticket_count_30d", 30.0, frozenset({"support_logs"}), event_types=frozenset({"ticket_created"})),
)


def _tz(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def in_window(event: EventEnvelope, spec: WindowSpec, as_of: datetime) -> bool:
    """Event-time window membership: as_of - window <= event_time <= as_of."""
    if event.source not in spec.sources:
        return False
    if spec.event_types is not None and event.event_type not in spec.event_types:
        return False
    et = _tz(event.event_time)
    start = _tz(as_of) - timedelta(days=spec.window_days)
    return start <= et <= _tz(as_of)


def compute_window_features(
    events: list[EventEnvelope],
    as_of: datetime,
    specs: tuple[WindowSpec, ...] = DEFAULT_SPECS,
) -> dict[str, float | None]:
    """Compute every window feature for one customer as of ``as_of``.

    Pure over the event set: order-independent (out-of-order equivalence).
    """
    out: dict[str, float | None] = {}
    for spec in specs:
        values: list[float] = []
        for e in events:
            if not in_window(e, spec, as_of):
                continue
            if spec.aggregate == "count":
                values.append(1.0)
            else:
                raw = e.payload.get(spec.payload_field)  # type: ignore[index]
                if raw is None:
                    continue
                values.append(float(raw))
        if not values:
            out[spec.name] = 0.0 if spec.aggregate == "count" else 0.0
            if spec.aggregate != "count" and not values:
                out[spec.name] = None
            continue
        if spec.aggregate == "count":
            out[spec.name] = float(len(values))
        elif spec.aggregate == "sum":
            out[spec.name] = float(sum(values))
        elif spec.aggregate == "max":
            out[spec.name] = max(values)
        else:  # pragma: no cover
            raise ValueError(f"unknown aggregate {spec.aggregate}")
    return out


def deduplicate(events: list[EventEnvelope]) -> list[EventEnvelope]:
    """Drop duplicate event_ids, keeping the first occurrence."""
    seen: set[str] = set()
    out: list[EventEnvelope] = []
    for e in events:
        if e.event_id in seen:
            continue
        seen.add(e.event_id)
        out.append(e)
    return out
