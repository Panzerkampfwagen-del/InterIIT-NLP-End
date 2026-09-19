"""Explainability (Phase 10) - "Why did you recommend this?" without post-hoc
reconstruction.

Per docs/reference/04 Phase 10 and the PS: explanations are assembled
**deterministically** from the evidence snapshot stored at decision time -
zero LLM calls, zero state re-derivation. Two ``explain()`` calls for the same
``decision_id`` return byte-identical output (determinism test).

The snapshot is written once per decision into an append-only decision log
(reproducible decision logs #30): given a decision_id, the exact evidence,
guardrail outcome, route result, and state snapshot are recoverable.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import psycopg

from src.action.selector import ActionDecision
from src.config import get_settings
from src.guardrails.machine import GuardrailResult
from src.state.models import StateSnapshot


def _jsonify(obj: Any) -> Any:
    """Convert dataclasses / enums / datetimes / pydantic models to JSON-safe."""
    import dataclasses

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value") and hasattr(obj, "name") and not isinstance(obj, (str, int)):  # enums
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonify(v) for k, v in dataclasses.asdict(obj).items()}
    if hasattr(obj, "model_dump"):
        return _jsonify(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    return str(obj)


class DecisionNotFound(KeyError):
    pass


class Explainer:
    """Decision-log writer + deterministic explanation assembly (zero LLM)."""

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
                    CREATE TABLE IF NOT EXISTS decision_log (
                        decision_id      TEXT PRIMARY KEY,
                        customer_id      TEXT NOT NULL,
                        as_of_event_time TIMESTAMPTZ NOT NULL,
                        decision         JSONB NOT NULL,
                        guardrail        JSONB,
                        route            JSONB,
                        state_snapshot   JSONB,
                        trace_id         TEXT,
                        created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )

    def snapshot(
        self,
        decision_id: str,
        customer_id: str,
        as_of_event_time: datetime,
        decision: ActionDecision,
        guardrail: GuardrailResult | None,
        route: dict[str, Any] | None,
        state_snapshot: StateSnapshot,
        trace_id: str | None = None,
    ) -> None:
        """Write the evidence snapshot once (append-only decision log)."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO decision_log
                        (decision_id, customer_id, as_of_event_time, decision, guardrail, route, state_snapshot, trace_id)
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                    ON CONFLICT (decision_id) DO NOTHING
                    """,
                    (
                        decision_id,
                        customer_id,
                        as_of_event_time,
                        json.dumps(_jsonify(decision), default=str),
                        json.dumps(_jsonify(guardrail), default=str) if guardrail else None,
                        json.dumps(_jsonify(route), default=str) if route else None,
                        json.dumps(_jsonify(state_snapshot), default=str),
                        trace_id,
                    ),
                )

    def explain(self, decision_id: str) -> dict[str, Any]:
        """Deterministically assemble the explanation from the stored snapshot.

        ZERO LLM calls, zero state re-derivation: everything comes from the
        evidence snapshot written at decision time. Two calls return identical
        output (determinism).
        """
        row = self._load(decision_id)
        decision = row["decision"]
        guardrail = row["guardrail"] or {}
        route = row["route"] or {}

        evidence = decision.get("evidence", [])
        explanation = {
            "decision_id": decision_id,
            "customer_id": row["customer_id"],
            "as_of_event_time": row["as_of_event_time"].isoformat() if isinstance(row["as_of_event_time"], datetime) else row["as_of_event_time"],
            "final_action": route.get("final_action") or decision.get("action"),
            "status": route.get("status", "allowed"),
            "action_confidence": decision.get("action_confidence"),
            "confidence_band": decision.get("confidence_band"),
            "reasoning_summary": decision.get("reasoning_summary"),
            "inferred_state": decision.get("inferred_state"),
            "evidence": [
                {
                    "source": e.get("source"),
                    "agent": e.get("agent"),
                    "excerpt": e.get("excerpt"),
                    "timestamp": e.get("timestamp"),
                    "resolvable_refs": e.get("refs", []),
                }
                for e in evidence
            ],
            "guardrail": {
                "outcome": guardrail.get("outcome"),
                "tier": guardrail.get("tier"),
                "reasons": guardrail.get("reasons", []),
            },
            "hitl": {
                "approval_id": route.get("approval_id"),
                "status": route.get("status"),
            } if route.get("status") == "pending_approval" else None,
            "trace_id": row["trace_id"],
        }
        return explanation

    def explain_json(self, decision_id: str) -> str:
        """Byte-stable JSON rendering (determinism test compares this)."""
        return json.dumps(self.explain(decision_id), sort_keys=True, default=str)

    def _load(self, decision_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT decision_id, customer_id, as_of_event_time, decision, guardrail, route, state_snapshot, trace_id
                    FROM decision_log WHERE decision_id = %s
                    """,
                    (decision_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise DecisionNotFound(decision_id)
        decision = row[3]
        guardrail = row[4]
        route = row[5]
        if isinstance(decision, str):
            decision = json.loads(decision)
        if isinstance(guardrail, str):
            guardrail = json.loads(guardrail)
        if isinstance(route, str):
            route = json.loads(route)
        return {
            "decision_id": row[0],
            "customer_id": row[1],
            "as_of_event_time": row[2],
            "decision": decision,
            "guardrail": guardrail,
            "route": route,
            "state_snapshot": row[6],
            "trace_id": row[7],
        }
