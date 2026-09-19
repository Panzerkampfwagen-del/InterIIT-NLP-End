"""Kafka/Redpanda producer (the event-bus path)."""
from __future__ import annotations

from kafka import KafkaProducer

from src.ingestion.producers.base import EventProducer, serialize_event
from src.ingestion.schemas.events import EventEnvelope


class KafkaEventProducer(EventProducer):
    """Publishes validated envelopes to a Redpanda/Kafka topic keyed by customer_id."""

    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        self.topic = topic
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            key_serializer=lambda k: k.encode() if isinstance(k, str) else k,
            value_serializer=lambda v: v.encode() if isinstance(v, str) else v,
            acks="all",
            retries=3,
        )

    def publish(self, event: EventEnvelope) -> None:
        self._producer.send(self.topic, key=event.customer_id, value=serialize_event(event))

    def flush(self) -> None:
        self._producer.flush()

    def close(self) -> None:
        self._producer.flush()
        self._producer.close()
