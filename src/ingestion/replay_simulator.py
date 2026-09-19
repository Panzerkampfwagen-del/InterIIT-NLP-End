"""Event Replay Simulator - *** SIMULATION ONLY *** (Phase 1).

This module is a TEST/SIMULATION component, clearly separated from the
production ingestion path. It reads a scenario (history seed + live stream)
and publishes events to the event bus **through the same producer interface
the production path uses** (``src.ingestion.producers``), adding controlled
impairment so downstream event-time handling can be tested:

- configurable replay velocity (real seconds per simulated day);
- out-of-order injection (a fraction of events emitted out of event_time order);
- duplicate injection (a fraction of events emitted twice);
- late-arrival injection (ingestion_time pushed past event_time, and the event
  emitted a configurable wall-clock delay later than its stream position).

Nothing in this module writes to production state; it only produces to the bus.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import timedelta

from src.ingestion.dataset_loader import Scenario
from src.ingestion.producers.base import EventProducer
from src.ingestion.schemas.events import EventEnvelope


@dataclass(frozen=True)
class ReplayConfig:
    """Controls impairment; deterministic given ``seed``."""

    seconds_per_simulated_day: float = 0.0  # 0 = as-fast-as-possible (batch replay)
    out_of_order_rate: float = 0.0
    duplicate_rate: float = 0.0
    late_rate: float = 0.0
    late_by_seconds: float = 60.0  # simulated seconds added to ingestion_time
    seed: int = 42
    include_history_seed: bool = False  # publish the seed first if True


@dataclass
class ReplayReport:
    """Counts produced by one replay run (asserted on in Phase 1 tests)."""

    published: int = 0
    duplicates_injected: int = 0
    out_of_order_pairs: int = 0
    late_injected: int = 0
    wall_clock_seconds: float = 0.0
    event_ids_published: list[str] = field(default_factory=list)


class ReplaySimulator:
    """*** SIMULATION ONLY *** publishes a scenario to the bus with impairment."""

    def __init__(self, producer: EventProducer, scenario: Scenario, config: ReplayConfig) -> None:
        self.producer = producer
        self.scenario = scenario
        self.config = config
        self._rng = random.Random(config.seed)

    # -- impairment helpers -------------------------------------------------

    def _impair_late(self, event: EventEnvelope) -> EventEnvelope:
        """Push ingestion_time past event_time by ``late_by_seconds``."""
        dt = timedelta(seconds=self.config.late_by_seconds)
        return event.model_copy(
            update={
                "ingestion_time": (event.ingestion_time or event.event_time) + dt,
            }
        )

    def _publish(self, event: EventEnvelope) -> None:
        self.producer.publish(event)
        self.report.published += 1
        self.report.event_ids_published.append(event.event_id)
        if self.config.seconds_per_simulated_day > 0:
            time.sleep(self.config.seconds_per_simulated_day / 86400.0)

    # -- main replay loop ----------------------------------------------------

    def run(self) -> ReplayReport:
        """Replay the scenario in event_time order with configured impairment."""
        self.report = ReplayReport()
        start = time.monotonic()

        stream: list[EventEnvelope] = list(self.scenario.live_stream)
        if self.config.include_history_seed:
            stream = list(self.scenario.history_seed) + stream
        stream.sort(key=lambda e: e.event_time)

        published_order: list[str] = []
        held: EventEnvelope | None = None  # event held back for out-of-order injection
        for event in stream:
            # late-arrival impairment: mark ingestion_time late before publishing
            if self._rng.random() < self.config.late_rate:
                event = self._impair_late(event)
                self.report.late_injected += 1

            # duplicate impairment: publish twice
            if self._rng.random() < self.config.duplicate_rate:
                self._publish(event)
                self.report.duplicates_injected += 1

            # out-of-order impairment: hold this event back so the NEXT event is
            # published before it (breaks event_time order in published stream)
            if held is not None:
                self._publish(event)
                self._publish(held)
                published_order.extend([event.event_id, held.event_id])
                held = None
                self.report.out_of_order_pairs += 1
                continue
            if self._rng.random() < self.config.out_of_order_rate:
                held = event
                continue
            self._publish(event)
            published_order.append(event.event_id)

        if held is not None:  # flush a still-held event at end-of-stream
            self._publish(held)
            published_order.append(held.event_id)

        self.report.wall_clock_seconds = time.monotonic() - start
        self.report.event_ids_published = published_order
        return self.report


def replay_scenario(
    producer: EventProducer,
    scenario: Scenario,
    config: ReplayConfig | None = None,
) -> ReplayReport:
    """Convenience: build a simulator and run it."""
    return ReplaySimulator(producer, scenario, config or ReplayConfig()).run()
