"""Customer State Board (Phase 3) - the one place all agents read/write.

Backed by Postgres (ADR-03) with:

- **Optimistic concurrency**: every write is a compare-and-swap keyed on
  ``state_version``; a concurrent writer gets a ``VersionConflict`` and
  retries - no silent last-write-wins.
- **Episodic store (append-only)**: every field write inserts a row with
  ``valid_from`` (event time); superseding writes set ``superseded_by`` on the
  prior row rather than overwriting it - this is what makes decision logs
  reproducible (#30): given an ``as_of`` instant, the exact state can be
  reconstructed.
- **Conflict log**: when two agents write conflicting values to the same field
  in the same decision cycle, both values are preserved under ``conflicts[]``
  and routed to the Synthesis/Critique step; the losing value is never deleted.
- **Lazy decay** at read time (see ``src/state/decay.py``).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import psycopg

from src.config import get_settings
from src.state.models import (
    Finding,
    Intervention,
    Provenance,
    StateConflict,
    StateSnapshot,
)


class VersionConflict(Exception):
    """Optimistic-concurrency failure: state_version moved under us."""

    def __init__(self, customer_id: str, expected_version: int) -> None:
        super().__init__(
            f"state_version conflict for {customer_id}: expected {expected_version}"
        )
        self.customer_id = customer_id
        self.expected_version = expected_version


class CustomerStateBoard:
    """Postgres-backed state board: CAS writes + episodic history + decay."""

    RECENT_EVENTS_RING = 50

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
                    CREATE TABLE IF NOT EXISTS customer_state (
                        customer_id                 TEXT PRIMARY KEY,
                        state                       JSONB NOT NULL DEFAULT '{}'::jsonb,
                        state_version               BIGINT NOT NULL DEFAULT 0,
                        last_updated_event_time     TIMESTAMPTZ,
                        last_updated_processing_time TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS state_episodic (
                        episode_id    BIGSERIAL PRIMARY KEY,
                        customer_id   TEXT NOT NULL,
                        field_path    TEXT NOT NULL,
                        value         JSONB NOT NULL,
                        confidence    REAL,
                        provenance    JSONB,
                        valid_from    TIMESTAMPTZ NOT NULL,
                        valid_to      TIMESTAMPTZ,
                        superseded_by BIGINT REFERENCES state_episodic(episode_id),
                        observed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_state_episodic_lookup ON state_episodic(customer_id, field_path, valid_from)"
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS state_conflicts (
                        conflict_id  BIGSERIAL PRIMARY KEY,
                        customer_id  TEXT NOT NULL,
                        field_path   TEXT NOT NULL,
                        agents       TEXT[] NOT NULL,
                        values       JSONB NOT NULL,
                        resolved_by  TEXT,
                        resolution   JSONB,
                        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                        resolved_at  TIMESTAMPTZ
                    )
                    """
                )

    # ------------------------------------------------------------------ reads

    def _row_to_snapshot(self, row) -> StateSnapshot:
        state = row[1] if not isinstance(row[1], str) else json.loads(row[1])
        state["state_version"] = row[2]
        state["last_updated_event_time"] = row[3]
        state["last_updated_processing_time"] = row[4]
        state.setdefault("customer_id", row[0])
        return StateSnapshot.model_validate(state)

    def get(self, customer_id: str, apply_dec: bool = True, now: datetime | None = None) -> StateSnapshot:
        """Current state snapshot with lazy decay applied at read time."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT customer_id, state, state_version, last_updated_event_time, last_updated_processing_time FROM customer_state WHERE customer_id = %s",
                    (customer_id,),
                )
                row = cur.fetchone()
        if row is None:
            return StateSnapshot(customer_id=customer_id)
        snap = self._row_to_snapshot(row)
        if apply_dec:
            from src.state.decay import apply_decay

            s = get_settings()
            snap = apply_decay(
                snap,
                now or datetime.now(UTC),
                default_half_life_days=s.default_half_life_days,
                min_confidence=s.decay_min_confidence,
            )
        return snap

    def version(self, customer_id: str) -> int:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT state_version FROM customer_state WHERE customer_id = %s", (customer_id,))
                row = cur.fetchone()
        return row[0] if row else 0

    # ----------------------------------------------------------------- writes

    def ensure_customer(self, customer_id: str) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO customer_state (customer_id, state) VALUES (%s, %s::jsonb) ON CONFLICT (customer_id) DO NOTHING",
                    (customer_id, json.dumps({"customer_id": customer_id})),
                )

    def update_field(
        self,
        customer_id: str,
        field_path: str,
        value: Any,
        confidence: float | None,
        provenance: Provenance,
        event_time: datetime,
        expected_version: int | None = None,
        conflict_with: str | None = None,
    ) -> StateSnapshot:
        """CAS write of one field + episodic insert + supersede of the prior row.

        Raises ``VersionConflict`` when the state moved concurrently (caller
        retries with the fresh version - optimistic concurrency, never silent
        last-write-wins). When ``conflict_with`` names another agent already
        holding a different value for this field in this cycle, both values
        are preserved in the conflict log for arbitration.
        """
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT state, state_version FROM customer_state WHERE customer_id = %s FOR UPDATE",
                (customer_id,),
            )
            row = cur.fetchone()
            if row is None:
                state: dict[str, Any] = {"customer_id": customer_id}
                current_version = 0
            else:
                state = row[0] if not isinstance(row[0], str) else json.loads(row[0])
                current_version = row[1]
            if expected_version is None:
                expected_version = current_version
            if expected_version != current_version:
                raise VersionConflict(customer_id, expected_version)

            # conflict detection: different agent, different value, same cycle.
            # Inlined INSERT (same transaction as the field write); calling
            # self.record_conflict here would open a nested `with` on the same
            # connection and close it mid-write.
            existing_value = _get_path(state, field_path)
            if (
                conflict_with
                and existing_value is not None
                and existing_value != value
            ):
                cur.execute(
                    """
                    INSERT INTO state_conflicts (customer_id, field_path, agents, values)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (customer_id, field_path, [conflict_with, provenance.agent],
                     json.dumps(_jsonable([existing_value, value]))),
                )

            # 1. supersede the prior episodic row (never overwrite)
            cur.execute(
                """
                SELECT episode_id FROM state_episodic
                WHERE customer_id = %s AND field_path = %s AND valid_to IS NULL
                ORDER BY episode_id DESC LIMIT 1
                """,
                (customer_id, field_path),
            )
            prior = cur.fetchone()
            # 2. insert the new episodic row
            cur.execute(
                """
                INSERT INTO state_episodic (customer_id, field_path, value, confidence, provenance, valid_from)
                VALUES (%s, %s, %s::jsonb, %s, %s::jsonb, %s)
                RETURNING episode_id
                """,
                (customer_id, field_path, json.dumps(_jsonable(value)), confidence,
                 json.dumps(provenance.model_dump(mode="json")), event_time),
            )
            new_episode = cur.fetchone()[0]
            if prior is not None:
                cur.execute(
                    "UPDATE state_episodic SET valid_to = %s, superseded_by = %s WHERE episode_id = %s",
                    (event_time, new_episode, prior[0]),
                )

            # 3. write the field into the state JSONB. Self-contained values
            # (findings, risks: provenance/confidence carried inside the dict)
            # are written as-is; scalar inferred fields (life_phase) get sibling
            # _provenance/_confidence/_valid_from keys per the documented model.
            _set_path(state, field_path, value)
            if not _self_contained(value):
                _set_path(state, f"{field_path}_provenance", provenance.model_dump(mode="json"))
                if confidence is not None:
                    _set_path(state, f"{field_path}_confidence", confidence)
                    _set_path(state, f"{field_path}_valid_from", event_time)

            # 4. CAS bump of state_version
            cur.execute(
                """
                INSERT INTO customer_state (customer_id, state, state_version, last_updated_event_time, last_updated_processing_time)
                VALUES (%s, %s::jsonb, 1, %s, now())
                ON CONFLICT (customer_id) DO UPDATE SET
                    state = EXCLUDED.state,
                    state_version = customer_state.state_version + 1,
                    last_updated_event_time = EXCLUDED.last_updated_event_time,
                    last_updated_processing_time = now()
                WHERE customer_state.state_version = %s
                RETURNING state_version
                """,
                (customer_id, json.dumps(_jsonable(state)), event_time, current_version),
            )
            bumped = cur.fetchone()
            if bumped is None:
                raise VersionConflict(customer_id, current_version)
        return self.get(customer_id, apply_dec=False)

    def append_event_ref(self, customer_id: str, event_ref: dict[str, Any]) -> None:
        """Append to the recent-events ring buffer (bounded, refs not payloads)."""
        snap = self.get(customer_id, apply_dec=False)
        ring = [e for e in snap.recent_events if e.get("event_id") != event_ref.get("event_id")]
        ring.append(event_ref)
        snap.recent_events = ring[-self.RECENT_EVENTS_RING:]
        self._write_full_state(snap)

    def record_finding(self, customer_id: str, domain: str, finding: Finding, event_time: datetime) -> StateSnapshot:
        return self.update_field(
            customer_id,
            f"findings.{domain}",
            finding.model_dump(mode="json"),
            finding.confidence,
            finding.provenance,
            event_time,
        )

    def record_intervention(self, customer_id: str, intervention: Intervention) -> None:
        snap = self.get(customer_id, apply_dec=False)
        snap.past_interventions = [i for i in snap.past_interventions if i.decision_id != intervention.decision_id]
        snap.past_interventions.append(intervention)
        self._write_full_state(snap)

    def record_conflict(
        self,
        customer_id: str,
        field_path: str,
        agents: list[str],
        values: list[Any],
        resolved_by: str | None = None,
        resolution: Any = None,
    ) -> int:
        """Append to the conflict log (both values preserved, never deleted)."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO state_conflicts (customer_id, field_path, agents, values, resolved_by, resolution)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb)
                    RETURNING conflict_id
                    """,
                    (customer_id, field_path, agents,
                     json.dumps(_jsonable(values)), resolved_by,
                     json.dumps(_jsonable(resolution)) if resolution is not None else None),
                )
                return cur.fetchone()[0]

    def resolve_conflict_by_field(self, customer_id: str, field_path: str, resolved_by: str, resolution: Any) -> None:
        """Resolve ALL pending conflicts for one field path (critique step)."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE state_conflicts
                    SET resolved_by = %s, resolution = %s::jsonb, resolved_at = now()
                    WHERE customer_id = %s AND field_path = %s AND resolved_at IS NULL
                    """,
                    (resolved_by, json.dumps(_jsonable(resolution)), customer_id, field_path),
                )

    def resolve_conflict(self, conflict_id: int, resolved_by: str, resolution: Any) -> None:
        """Record arbitration; the losing value stays in the log for audit."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE state_conflicts SET resolved_by = %s, resolution = %s::jsonb, resolved_at = now() WHERE conflict_id = %s",
                    (resolved_by, json.dumps(_jsonable(resolution)), conflict_id),
                )

    def pending_conflicts(self, customer_id: str) -> list[StateConflict]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT conflict_id, field_path, agents, values, resolved_by, resolution, created_at, resolved_at
                    FROM state_conflicts WHERE customer_id = %s AND resolved_at IS NULL ORDER BY conflict_id
                    """,
                    (customer_id,),
                )
                rows = cur.fetchall()
        out: list[StateConflict] = []
        for rid, path, agents, values, resolved_by, resolution, created, resolved in rows:
            out.append(
                StateConflict(
                    field_path=path,
                    agents=list(agents),
                    values=values if not isinstance(values, str) else json.loads(values),
                    resolved_by=resolved_by,
                    resolution=resolution,
                    resolved_at=resolved,
                )
            )
        return out

    # --------------------------------------------------- reproducible history

    def reconstruct(self, customer_id: str, as_of: datetime) -> StateSnapshot:
        """Reconstruct the exact state as it looked at ``as_of`` (decision logs).

        Walks the episodic history: for each field path, the latest row with
        ``valid_from <= as_of`` - reproducible byte-for-byte from stored history.
        """
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (field_path)
                        field_path, value, confidence, provenance, valid_from
                    FROM state_episodic
                    WHERE customer_id = %s AND valid_from <= %s
                    ORDER BY field_path, valid_from DESC, episode_id DESC
                    """,
                    (customer_id, as_of),
                )
                rows = cur.fetchall()
        state: dict[str, Any] = {"customer_id": customer_id}
        for field_path, value, confidence, provenance, valid_from in rows:
            _set_path(state, field_path, value)
            if not _self_contained(value):
                _set_path(state, f"{field_path}_provenance", provenance)
                if confidence is not None:
                    _set_path(state, f"{field_path}_confidence", confidence)
                    _set_path(state, f"{field_path}_valid_from", valid_from)
        return StateSnapshot.model_validate(state)

    # ------------------------------------------------------------------ internals

    def _write_full_state(self, snap: StateSnapshot) -> None:
        """Whole-snapshot write (ring buffer, interventions); still version-bumped."""
        data = snap.model_dump(mode="json", exclude={"state_version", "last_updated_event_time", "last_updated_processing_time"})
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO customer_state (customer_id, state, state_version, last_updated_processing_time)
                    VALUES (%s, %s::jsonb, 1, now())
                    ON CONFLICT (customer_id) DO UPDATE SET
                        state = EXCLUDED.state,
                        state_version = customer_state.state_version + 1,
                        last_updated_processing_time = now()
                    """,
                    (snap.customer_id, json.dumps(_jsonable(data))),
                )


def _get_path(obj: dict, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_path(obj: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _self_contained(value: Any) -> bool:
    """True when a dict value carries its own provenance/confidence metadata
    (findings, active risks/opportunities, recommended action)."""
    return isinstance(value, dict) and ("provenance" in value or "confidence" in value)


def _jsonable(value: Any) -> Any:
    """Convert datetimes/enums/pydantic models into JSON-safe structures."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "value") and hasattr(value, "name"):
        return value.value  # enums
    return value
