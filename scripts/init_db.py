#!/usr/bin/env python3
"""Initialize the Postgres instance for Agentic Customer 360.

Phase 0: enables pgvector and creates the migration-tracking table.
Later phases register their DDL here incrementally (identity, state,
retrieval, HITL, decision log) so a fresh clone boots to a working schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

from src.config import get_settings


def main() -> int:
    settings = get_settings()
    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    migration_id   TEXT PRIMARY KEY,
                    applied_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    description    TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                INSERT INTO schema_migrations (migration_id, description)
                VALUES ('0001_phase0_baseline', 'pgvector extension + migration table')
                ON CONFLICT (migration_id) DO NOTHING
                """
            )
            # Phase 2: identity graph schema
            from src.identity.graph import IdentityGraph
            IdentityGraph(".").init_schema(conn)
            cur.execute(
                """
                INSERT INTO schema_migrations (migration_id, description)
                VALUES ('0002_phase2_identity', 'identity_records + identity_graph_edges')
                ON CONFLICT (migration_id) DO NOTHING
                """
            )
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
            if row is None:
                print("FATAL: pgvector extension not available", file=sys.stderr)
                return 1
    print("db init ok: pgvector enabled, migration table ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
