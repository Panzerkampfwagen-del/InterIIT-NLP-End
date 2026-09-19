"""Producer interface shared by simulation and (future) production paths.

The Phase 1 prompt requires the replay simulator to be clearly labeled as
SIMULATION while publishing through the same producer interface the production
path uses, so impairments (out-of-order/duplicate/late) cannot silently leak
into real ingestion code paths.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from src.ingestion.schemas.events import EventEnvelope


class EventProducer(ABC):
    """Publish validated envelopes to the event bus, keyed by customer_id."""

    topic: str

    @abstractmethod
    def publish(self, event: EventEnvelope) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


def serialize_event(event: EventEnvelope) -> str:
    """Canonical JSON serialization (stable field order)."""
    data = event.model_dump(mode="json")
    return json.dumps(data, sort_keys=True)


def deserialize_event(raw: str | bytes) -> EventEnvelope:
    """Parse one bus message back into a validated envelope."""
    if isinstance(raw, bytes):
        raw = raw.decode()
    return EventEnvelope.model_validate(json.loads(raw))
