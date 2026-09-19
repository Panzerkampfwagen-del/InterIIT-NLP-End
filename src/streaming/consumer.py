"""Event-time-aware consumer (Phase 4) - Streaming Processing step.

Processes validated envelopes with:

- **deduplication** by event_id (duplicate injection never double-counts);
- **watermark tracking** per source (bounded out-of-orderness, idle timeout);
- **late-arrival handling**: an event whose event_time is behind the watermark
  but within ``allowed_lateness`` is accepted and triggers window-feature
  **recomputation** with a ``superseded_by``-linked baseline writeback (the
  state board's episodic store provides the supersede, per #30);
- events later than allowed lateness are dropped (logged, counted);
- **baseline writeback**: recomputed window features land on the state board's
  ``historical_baseline`` with provenance ``streaming_consumer``.

This is the *production path* component - distinct from the replay simulator
(SIMULATION ONLY), which feeds it from generated scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from src.config import get_settings
from src.ingestion.schemas.events import EventEnvelope
from src.state.board import CustomerStateBoard
from src.state.models import Provenance
from src.streaming.features import DEFAULT_SPECS, WindowSpec, compute_window_features
from src.streaming.watermarks import WatermarkTracker


class Outcome(str, Enum):
    EMITTED = "emitted"                    # in-order (or within out-of-orderness)
    DUPLICATE = "duplicate"                # event_id already processed
    LATE_RECOMPUTED = "late_recomputed"    # behind watermark but within lateness
    TOO_LATE_DROPPED = "too_late_dropped"  # older than allowed lateness


@dataclass(frozen=True)
class ProcessingResult:
    outcome: Outcome
    event_id: str
    customer_id: str
    features: dict[str, Any] | None = None  # recomputed baseline when late


class EventTimeConsumer:
    """Dedup + watermark + lateness + recompute, writing baselines to the board."""

    def __init__(
        self,
        board: CustomerStateBoard,
        tracker: WatermarkTracker | None = None,
        specs: tuple[WindowSpec, ...] = DEFAULT_SPECS,
        allowed_lateness_s: float | None = None,
    ) -> None:
        s = get_settings()
        self.board = board
        self.tracker = tracker or WatermarkTracker()
        self.specs = specs
        self.allowed_lateness = timedelta(seconds=allowed_lateness_s if allowed_lateness_s is not None else s.allowed_lateness_s)
        self._seen: set[str] = set()
        self._max_event_time: dict[str, datetime] = {}  # in-memory per-source max

    # ---------------------------------------------------------------- helpers

    def _is_duplicate(self, event: EventEnvelope) -> bool:
        # event ids are unique PER CUSTOMER (the dataset reuses sequences
        # across customers); dedup key must be (customer_id, event_id)
        key = (event.customer_id, event.event_id)
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    def _is_late(self, event: EventEnvelope) -> bool:
        """Behind the (in-memory) max event time for its source, bounded by
        out-of-orderness; genuinely old events (beyond allowed lateness) are
        dropped."""
        et = event.event_time if event.event_time.tzinfo else event.event_time.replace(tzinfo=UTC)
        source_max = self._max_event_time.get(event.source)
        if source_max is None:
            return False
        return et < source_max - timedelta(seconds=self.tracker.out_of_order_max_s)

    def _is_too_late(self, event: EventEnvelope, now: datetime | None = None) -> bool:
        et = event.event_time if event.event_time.tzinfo else event.event_time.replace(tzinfo=UTC)
        source_max = self._max_event_time.get(event.source)
        if source_max is None:
            return False
        return (source_max - et) > self.allowed_lateness

    def _features(self, event: EventEnvelope, ledger: list[EventEnvelope], horizon: datetime) -> dict[str, Any]:
        customer_events = [e for e in ledger if e.customer_id == event.customer_id]
        # recompute as of the current horizon (max event time seen), so the
        # baseline reflects the full window including all arrived events
        return compute_window_features(customer_events, horizon, self.specs)

    # ----------------------------------------------------------------- process

    def process_batch(
        self,
        events: list[EventEnvelope],
        processing_time: datetime | None = None,
    ) -> list[ProcessingResult]:
        """Process a batch in ARRIVAL order (the real ingestion order).

        Arrival order is authoritative for lateness: an event behind the
        per-source max event time (bounded by out-of-orderness) is late;
        behind by more than allowed lateness -> dropped. Duplicates are
        dropped in-loop (never double-counted). Late events within lateness
        trigger window-feature recomputation **as of the current horizon**
        (max event time seen), with a supersede-linked baseline writeback.
        """
        results: list[ProcessingResult] = []
        ledger: list[EventEnvelope] = []
        horizon: datetime | None = None  # max event time seen across sources
        for e in events:
            et = e.event_time if e.event_time.tzinfo else e.event_time.replace(tzinfo=UTC)
            # duplicate check in-loop (not pre-filtered)
            if self._is_duplicate(e):
                results.append(ProcessingResult(Outcome.DUPLICATE, e.event_id, e.customer_id))
                continue
            prev_max = self._max_event_time.get(e.source)
            is_too_late = prev_max is not None and (prev_max - et) > self.allowed_lateness
            is_late = (
                prev_max is not None
                and et < prev_max - timedelta(seconds=self.tracker.out_of_order_max_s)
                and not is_too_late
            )
            if is_too_late:
                results.append(ProcessingResult(Outcome.TOO_LATE_DROPPED, e.event_id, e.customer_id))
                continue
            ledger.append(e)
            self._max_event_time[e.source] = max(prev_max, et) if prev_max else et
            horizon = max(horizon, et) if horizon else et
            self.tracker.observe(e.source, et, processing_time)
            features = self._features(e, ledger, horizon) if is_late else None
            results.append(
                ProcessingResult(
                    Outcome.LATE_RECOMPUTED if is_late else Outcome.EMITTED,
                    e.event_id, e.customer_id, features,
                )
            )
            if features is not None:
                self._write_baseline(e, features, horizon)
        return results

    def _write_baseline(self, event: EventEnvelope, features: dict[str, Any], horizon: datetime | None = None) -> None:
        """Writeback recomputed features to the state board (supersede via
        episodic store); provenance = streaming_consumer."""
        prov = Provenance(agent="streaming_consumer", event_id=event.event_id)
        self.board.ensure_customer(event.customer_id)
        baseline = self.board.get(event.customer_id, apply_dec=False).historical_baseline
        merged = {**baseline, **features, "computed_at": horizon or event.event_time}
        self.board.update_field(
            event.customer_id,
            "historical_baseline",
            merged,
            None,
            prov,
            event.event_time,
        )
