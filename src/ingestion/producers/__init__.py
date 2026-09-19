"""Producers: Kafka/Redpanda (bus) + in-memory (tests)."""
from src.ingestion.producers.base import EventProducer, deserialize_event, serialize_event
from src.ingestion.producers.kafka_producer import KafkaEventProducer
from src.ingestion.producers.memory_producer import InMemoryEventProducer

__all__ = [
    "EventProducer",
    "KafkaEventProducer",
    "InMemoryEventProducer",
    "serialize_event",
    "deserialize_event",
]
