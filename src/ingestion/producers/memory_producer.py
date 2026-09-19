"""In-memory producer for tests and offline pipelines.

Records published messages so tests can assert on impairments and consumers
can drain the "bus" without infrastructure.
"""
from __future__ import annotations

from src.ingestion.producers.base import EventProducer, deserialize_event, serialize_event
from src.ingestion.schemas.events import EventEnvelope


class InMemoryEventProducer(EventProducer):
    """Collects (key, serialized) tuples in memory, in publish order."""

    def __init__(self, topic: str = "c360.events.raw") -> None:
        self.topic = topic
        self.published: list[tuple[str, str]] = []

    def publish(self, event: EventEnvelope) -> None:
        self.published.append((event.customer_id, serialize_event(event)))

    @property
    def events(self) -> list[EventEnvelope]:
        return [deserialize_event(v) for _, v in self.published]

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None
