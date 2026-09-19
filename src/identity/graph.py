"""In-Postgres identity graph (Phase 2, RQ-I1 / ADR-03).

Append-only ``identity_records`` + ``identity_graph_edges``; connected
components (transitive closure: A<->B, B<->C => A<->C) computed with an
in-memory union-find loaded from the edge table - auditable, deterministic,
and no second stateful system (the dedicated temporal-KG store is a documented
future upgrade in ADR-03).
"""
from __future__ import annotations

import psycopg

from src.identity.matcher import IdentityRecord, MatchResult


class IdentityGraph:
    """Records + edges + union-find closure over Postgres."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def init_schema(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS identity_records (
                    record_id            BIGSERIAL PRIMARY KEY,
                    source_system        TEXT NOT NULL,
                    source_customer_id   TEXT NOT NULL,
                    canonical_customer_id TEXT NOT NULL,
                    full_name            TEXT,
                    email                TEXT,
                    phone                TEXT,
                    dob                  DATE,
                    kyc_doc_number       TEXT,
                    match_method         TEXT NOT NULL DEFAULT 'first_seen',
                    match_confidence     REAL NOT NULL DEFAULT 1.0,
                    match_reason         TEXT NOT NULL DEFAULT '',
                    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (source_system, source_customer_id)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_identity_records_canonical ON identity_records(canonical_customer_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_identity_records_email ON identity_records(lower(email))")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_identity_records_phone ON identity_records(phone)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_identity_records_doc ON identity_records(kyc_doc_number)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS identity_graph_edges (
                    edge_id     BIGSERIAL PRIMARY KEY,
                    record_a    BIGINT NOT NULL REFERENCES identity_records(record_id),
                    record_b    BIGINT NOT NULL REFERENCES identity_records(record_id),
                    method      TEXT NOT NULL,
                    confidence  REAL NOT NULL,
                    reason      TEXT NOT NULL DEFAULT '',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    def _load_records(self, conn: psycopg.Connection) -> dict[int, tuple[IdentityRecord, str]]:
        """record_id -> (attributes, canonical_customer_id)."""
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT record_id, source_system, source_customer_id, canonical_customer_id,
                       full_name, email, phone, dob, kyc_doc_number
                FROM identity_records
                """
            )
            out: dict[int, tuple[IdentityRecord, str]] = {}
            for row in cur.fetchall():
                rec = IdentityRecord(
                    source_customer_id=row[2],
                    source_system=row[1],
                    full_name=row[4],
                    email=row[5],
                    phone=row[6],
                    dob=row[7],
                    kyc_doc_number=row[8],
                )
                out[row[0]] = (rec, row[3])
            return out

    def register_first_seen(self, conn: psycopg.Connection, record: IdentityRecord, canonical: str) -> int:
        """Insert a first-seen record mapped to its own canonical id (idempotent)."""
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO identity_records
                    (source_system, source_customer_id, canonical_customer_id, full_name, email, phone, dob, kyc_doc_number, match_method, match_confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'first_seen', 1.0)
                ON CONFLICT (source_system, source_customer_id) DO NOTHING
                RETURNING record_id
                """,
                (record.source_system, record.source_customer_id, canonical, record.full_name, record.email, record.phone, record.dob, record.kyc_doc_number),
            )
            row = cur.fetchone()
            if row is not None:
                return row[0]
            cur.execute(
                "SELECT record_id FROM identity_records WHERE source_system = %s AND source_customer_id = %s",
                (record.source_system, record.source_customer_id),
            )
            return cur.fetchone()[0]

    def add_edge(self, conn: psycopg.Connection, record_a: int, record_b: int, result: MatchResult) -> None:
        """Append one audited merge edge (never deleted)."""
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO identity_graph_edges (record_a, record_b, method, confidence, reason)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (record_a, record_b, result.method.value, result.confidence, result.reason),
            )

    def components(self, conn: psycopg.Connection) -> dict[str, str]:
        """source_customer_id -> canonical_customer_id via transitive closure.

        Union-find over all records + edges: the canonical representative of a
        component is the lexicographically smallest source_customer_id in it.
        """
        records = self._load_records(conn)
        parent: dict[int, int] = {rid: rid for rid in records}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        with conn.cursor() as cur:
            cur.execute("SELECT record_a, record_b FROM identity_graph_edges")
            for a, b in cur.fetchall():
                union(a, b)

        # choose representative (smallest source_customer_id) per component
        best: dict[int, str] = {}
        for rid, (rec, _canon) in records.items():
            root = find(rid)
            cur_id = rec.source_customer_id
            if root not in best or cur_id < best[root]:
                best[root] = cur_id
        return {rec.source_customer_id: best[find(rid)] for rid, (rec, _c) in records.items()}
