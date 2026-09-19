"""Event-time watermark tracking (Phase 4, ADR-02).

Replicates Flink's ``WatermarkStrategy.forBoundedOutOfOrderness`` *semantics*
on top of Postgres - no Flink (scale-conditioned decision, ADR-02):

- per-source ``max_event_time``; the watermark is
  ``min over non-idle sources of (max_event_time - out_of_order_max)``;
- watermark = estimate of "all events before this point have likely arrived";
- idle sources (silent longer than ``idle_source_s`` of processing time) are
  marked idle and excluded, so a silent source cannot stall the watermark
  (Flink's idle-partition timeout semantics);
- watermark lag is observable (Phase 11 timing metrics consume it).

Event time is authoritative: "recent" means *when it happened*, not when we
saw it (#3 in the technical-problems table).
"""
from __future__ import annotations

from datetime import UTC, datetime

import psycopg

from src.config import get_settings


class WatermarkTracker:
    """Per-source event-time tracking with bounded out-of-orderness + idle timeout."""

    def __init__(
        self,
        dsn: str | None = None,
        out_of_order_max_s: float | None = None,
        idle_source_s: float | None = None,
    ) -> None:
        s = get_settings()
        self.dsn = dsn or s.pg_dsn
        self.out_of_order_max_s = out_of_order_max_s if out_of_order_max_s is not None else s.watermark_out_of_order_max_s
        self.idle_source_s = idle_source_s if idle_source_s is not None else s.watermark_idle_source_s
        self._conn: psycopg.Connection | None = None

    def _connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    def init_schema(self) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS watermarks (
                        source_system            TEXT PRIMARY KEY,
                        max_event_time           TIMESTAMPTZ NOT NULL,
                        last_seen_processing_time TIMESTAMPTZ NOT NULL DEFAULT now(),
                        is_idle                  BOOLEAN NOT NULL DEFAULT false
                    )
                    """
                )

    def observe(self, source_system: str, event_time: datetime, processing_time: datetime | None = None) -> float:
        """Record one event from a source; returns the source's watermark."""
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=UTC)
        now = processing_time or datetime.now(UTC)
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO watermarks (source_system, max_event_time, last_seen_processing_time, is_idle)
                    VALUES (%s, %s, %s, false)
                    ON CONFLICT (source_system) DO UPDATE SET
                        max_event_time = GREATEST(watermarks.max_event_time, EXCLUDED.max_event_time),
                        last_seen_processing_time = EXCLUDED.last_seen_processing_time,
                        is_idle = false
                    RETURNING max_event_time
                    """,
                    (source_system, event_time, now),
                )
                max_et = cur.fetchone()[0]
        from datetime import timedelta

        return (max_et - timedelta(seconds=self.out_of_order_max_s)).timestamp()

    def mark_idle_sources(self, now: datetime | None = None) -> list[str]:
        """Mark sources silent for > idle_source_s of processing time as idle."""
        now = now or datetime.now(UTC)
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE watermarks SET is_idle = true
                    WHERE last_seen_processing_time < %s - make_interval(secs => %s)
                    RETURNING source_system
                    """,
                    (now, self.idle_source_s),
                )
                return [r[0] for r in cur.fetchall()]

    def watermark(self, now: datetime | None = None) -> float:
        """Current global watermark: min over non-idle sources.

        When NO source has ever produced (or all are idle), the watermark is
        -inf equivalent (a very small timestamp) - nothing is considered late.
        """
        self.mark_idle_sources(now)
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MIN(max_event_time) FROM watermarks WHERE is_idle = false"
                )
                row = cur.fetchone()
        if row is None or row[0] is None:
            return float("-inf")
        from datetime import timedelta

        return (row[0] - timedelta(seconds=self.out_of_order_max_s)).timestamp()

    def lag(self, now: datetime | None = None) -> float:
        """Observable watermark lag: seconds by which the watermark trails the
        max event time seen (Phase 11 timing metrics)."""
        self.mark_idle_sources(now)
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXTRACT(EPOCH FROM (MAX(max_event_time) - MIN(max_event_time)))
                    FROM watermarks WHERE is_idle = false
                    """
                )
                row = cur.fetchone()
        return float(row[0] or 0.0) + self.out_of_order_max_s
