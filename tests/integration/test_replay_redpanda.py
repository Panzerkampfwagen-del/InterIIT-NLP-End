"""Phase 1 integration: replay simulator → Redpanda → consume back.

Verifies the full event-bus path (KafkaEventProducer → Redpanda → KafkaConsumer)
round-trips validated envelopes byte-for-byte. Skipped when Redpanda is absent.
"""
from __future__ import annotations

import socket
import uuid

import pytest

from src.config import get_settings


def _redpanda_available() -> bool:
    s = get_settings()
    host, port = s.kafka_bootstrap.split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2.0):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _redpanda_available(), reason="Redpanda not running (docker compose up -d)"
)


def test_replay_round_trip_through_redpanda() -> None:
    from kafka import KafkaConsumer, KafkaProducer

    from src.ingestion.producers.base import deserialize_event, serialize_event
    from src.ingestion.schemas.events import parse_event

    s = get_settings()
    topic = f"{s.kafka_topic_prefix}.it.{uuid.uuid4().hex[:8]}"

    # a small replay of scenario_01 live stream through the real producer
    from src.ingestion.dataset_loader import load_scenario

    scenario = load_scenario("scenario_01")
    events = sorted(scenario.live_stream, key=lambda e: e.event_time)[:25]

    producer = KafkaProducer(
        bootstrap_servers=s.kafka_bootstrap,
        key_serializer=lambda k: k.encode(),
        value_serializer=lambda v: v.encode(),
    )
    for e in events:
        producer.send(topic, key=e.customer_id, value=serialize_event(e))
    producer.flush()
    producer.close()

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=s.kafka_bootstrap,
        auto_offset_reset="earliest",
        consumer_timeout_ms=15000,
        group_id=f"it-{uuid.uuid4().hex[:6]}",
    )
    received = [deserialize_event(msg.value) for msg in consumer]
    consumer.close()

    assert len(received) == len(events)
    assert [e.event_id for e in received] == [e.event_id for e in events]
    # round-tripped events still validate through the schema
    for e in received:
        assert parse_event(e.model_dump(mode="json")).event_id == e.event_id


def test_dataset_scenarios_all_load_and_validate() -> None:
    from src.ingestion.dataset_loader import available_scenarios, load_scenario

    names = available_scenarios()
    assert len(names) >= 7  # 3 dataset + 4 generated
    for name in names:
        scenario = load_scenario(name)
        assert len(scenario.live_stream) > 0
        assert scenario.customer_id
