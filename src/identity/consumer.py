"""Identity-annotating consumer (Phase 2) - Customer Identification step.

Consumes raw events from the bus, resolves identity, and publishes to the
``c360.events.resolved`` topic carrying ``resolved_customer_id`` plus the
identity confidence. Also usable inline via ``annotate_event`` for the
in-process pipeline path.
"""
from __future__ import annotations

from src.config import get_settings
from src.identity.resolver import IdentityResolver
from src.ingestion.producers.base import EventProducer
from src.ingestion.schemas.events import EventEnvelope

RESOLVED_TOPIC_SUFFIX = "events.resolved"


def annotate_event(event: EventEnvelope, resolver: IdentityResolver) -> EventEnvelope:
    """Return the event enriched with resolved identity (auditable metadata)."""
    resolved, confidence, method = resolver.annotate(event)
    return event.model_copy(
        update={
            "customer_id": resolved,
            "payload": {
                **event.payload,
                "_identity": {
                    "raw_customer_id": event.customer_id,
                    "resolved_customer_id": resolved,
                    "confidence": confidence,
                    "method": method,
                },
            },
        }
    )


class IdentityConsumer:
    """Raw-events consumer that annotates identity and republishes."""

    def __init__(self, resolver: IdentityResolver, producer: EventProducer, topic: str | None = None) -> None:
        self.resolver = resolver
        self.producer = producer
        s = get_settings()
        self.raw_topic = topic or f"{s.kafka_topic_prefix}.events.raw"

    def handle(self, event: EventEnvelope) -> EventEnvelope:
        annotated = annotate_event(event, self.resolver)
        self.producer.publish(annotated)
        return annotated
