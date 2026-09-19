"""Phase 0 smoke tests: infrastructure connectivity.

Definition of Done (Phase 0): docker-compose services come up healthy and a
smoke test connects to Postgres and Redis. Redpanda connectivity is also
verified here (Kafka API compatibility is an ADR assumption).

These tests require `docker compose up -d`; they are skipped (not failed) when
the infrastructure is intentionally absent so unit suites stay runnable offline.
"""
from __future__ import annotations

import socket

import pytest

from src.config import get_settings


def _port_open(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _infra_available() -> bool:
    s = get_settings()
    pg_host, pg_port = s.pg_dsn.split("//")[1].split("@")[1].split("/")[0].split(":")
    redis_host, redis_port = s.redis_url.split("//")[1].split("/")[0].split(":")
    k_host, k_port = s.kafka_bootstrap.split(":")
    return (
        _port_open(pg_host, int(pg_port))
        and _port_open(redis_host, int(redis_port))
        and _port_open(k_host, int(k_port))
    )


pytestmark = pytest.mark.skipif(
    not _infra_available(), reason="docker compose infrastructure not running"
)


def test_postgres_connects_and_pgvector_enabled() -> None:
    import psycopg

    s = get_settings()
    with psycopg.connect(s.pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone() == (1,)
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            assert cur.fetchone() is not None, "pgvector extension must be available"


def test_redis_connects() -> None:
    import redis

    s = get_settings()
    client = redis.Redis.from_url(s.redis_url, socket_timeout=3.0)
    assert client.ping() is True
    client.set("c360:smoke_test", "ok", ex=60)
    assert client.get("c360:smoke_test") == b"ok"


def test_redpanda_kafka_api_reachable() -> None:
    """The Kafka API must be reachable (Redpanda per ADR; plain Kafka acceptable)."""
    from kafka.admin import KafkaAdminClient

    s = get_settings()
    admin = KafkaAdminClient(bootstrap_servers=s.kafka_bootstrap, request_timeout_ms=5000)
    assert admin is not None
    admin.close()
