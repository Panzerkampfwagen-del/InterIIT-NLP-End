"""Durable HITL approval store (Phase 9) - physically blocking, not a log flag.

Per RQ-A2 and docs/reference/04 Phase 9: escalated decisions enter a durable
``pending_approval`` state persisted in Postgres that *physically* blocks
execution (the decision object is only released on an explicit approve).
Restart-durable: a forcibly killed and restarted process finds the same
pending approvals. Approve/reject/modify each produce correct state
transitions feeding the past_interventions feedback loop.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import psycopg

from src.config import get_settings


class HitlStore:
    """Durable pending_approval store over Postgres (physically blocking)."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or get_settings().pg_dsn
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
                    CREATE TABLE IF NOT EXISTS hitl_approvals (
                        approval_id      TEXT PRIMARY KEY,
                        customer_id      TEXT NOT NULL,
                        decision_id      TEXT NOT NULL,
                        proposed_action  TEXT NOT NULL,
                        decision_payload JSONB NOT NULL,
                        status           TEXT NOT NULL DEFAULT 'pending_approval',
                        final_action     TEXT,
                        reviewer         TEXT,
                        note             TEXT,
                        submitted_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                        resolved_at      TIMESTAMPTZ
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_hitl_status ON hitl_approvals(status)")

    def submit(self, customer_id: str, decision_id: str, proposed_action: str, decision_payload: dict[str, Any]) -> str:
        """Enter a durable pending_approval state (blocks execution physically)."""
        approval_id = f"apr_{uuid.uuid4().hex[:12]}"
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hitl_approvals (approval_id, customer_id, decision_id, proposed_action, decision_payload)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    """,
                    (approval_id, customer_id, decision_id, proposed_action,
                     json.dumps(decision_payload, default=str)),
                )
        return approval_id

    def get(self, approval_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT approval_id, customer_id, decision_id, proposed_action, decision_payload,
                           status, final_action, reviewer, note, submitted_at, resolved_at
                    FROM hitl_approvals WHERE approval_id = %s
                    """,
                    (approval_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return self._row(row)

    def pending(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT approval_id, customer_id, decision_id, proposed_action, decision_payload,
                           status, final_action, reviewer, note, submitted_at, resolved_at
                    FROM hitl_approvals WHERE status = 'pending_approval' ORDER BY submitted_at
                    """
                )
                rows = cur.fetchall()
        return [self._row(r) for r in rows]

    def approve(self, approval_id: str, reviewer: str, note: str = "") -> dict[str, Any] | None:
        """pending_approval -> approved (releases the proposed action)."""
        return self._resolve(approval_id, "approved", reviewer, note, final_action=None)

    def reject(self, approval_id: str, reviewer: str, note: str = "") -> dict[str, Any] | None:
        """pending_approval -> rejected (final action: explicit no_action)."""
        return self._resolve(approval_id, "rejected", reviewer, note, final_action="no_action")

    def modify(self, approval_id: str, reviewer: str, modified_action: str, note: str = "") -> dict[str, Any] | None:
        """pending_approval -> modified (final action: the modified action)."""
        return self._resolve(approval_id, "modified", reviewer, note, final_action=modified_action)

    def _resolve(self, approval_id: str, status: str, reviewer: str, note: str, final_action: str | None) -> dict[str, Any] | None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE hitl_approvals
                    SET status = %s, final_action = COALESCE(%s, proposed_action), reviewer = %s, note = %s, resolved_at = now()
                    WHERE approval_id = %s AND status = 'pending_approval'
                    RETURNING approval_id, customer_id, decision_id, proposed_action, decision_payload,
                              status, final_action, reviewer, note, submitted_at, resolved_at
                    """,
                    (status, final_action, reviewer, note, approval_id),
                )
                row = cur.fetchone()
        return self._row(row) if row else None

    def _row(self, row) -> dict[str, Any]:
        payload = row[4]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return {
            "approval_id": row[0],
            "customer_id": row[1],
            "decision_id": row[2],
            "proposed_action": row[3],
            "decision_payload": payload,
            "status": row[5],
            "final_action": row[6],
            "reviewer": row[7],
            "note": row[8],
            "submitted_at": row[9],
            "resolved_at": row[10],
        }
