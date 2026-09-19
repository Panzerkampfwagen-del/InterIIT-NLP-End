"""Phase 2 integration: resolver + identity graph over Postgres.

Requires docker compose infrastructure; skipped when absent.
"""
from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from src.config import get_settings


def _infra_available() -> bool:
    import socket

    s = get_settings()
    host, port = s.pg_dsn.split("//")[1].split("@")[1].split("/")[0].split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2.0):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _infra_available(), reason="Postgres not running")


@pytest.fixture()
def resolver():
    from src.identity.graph import IdentityGraph
    from src.identity.resolver import IdentityResolver

    s = get_settings()
    graph = IdentityGraph(s.pg_dsn)
    with psycopg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE identity_graph_edges, identity_records RESTART IDENTITY")
    r = IdentityResolver(graph, s.pg_dsn)
    yield r
    r.close()


def psycopg_connect():
    import psycopg

    return psycopg.connect(get_settings().pg_dsn, autocommit=True)


def _event(cid: str, source: str, payload: dict | None = None):
    from src.ingestion.schemas.events import parse_event

    t = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
    return parse_event(
        {
            "event_id": f"EVT_{abs(hash((cid, source, str(payload)))) % 100000:06d}",
            "event_time": t,
            "ingestion_time": t,
            "customer_id": cid,
            "account_id": "ACC_CHK_001",
            "source_system": source,
            "event_type": "deposit" if source == "core_banking_ledger" else "kyc_update",
            "schema_version": "1.0",
            "payload": payload or {"amount": 100, "balance_after": 1000, "transaction_type": "salary_credit"},
        }
    )


def test_first_seen_maps_to_itself(resolver) -> None:
    r = resolver.resolve("CUST_A", "core_banking_ledger")
    assert r.resolved_customer_id == "CUST_A" and r.match_method.value == "first_seen"
    # idempotent second resolution
    r2 = resolver.resolve("CUST_A", "core_banking_ledger")
    assert r2.resolved_customer_id == "CUST_A" and r2.merged is False


def test_email_merge_links_records(resolver) -> None:
    resolver.resolve("CUST_A", "core_banking_ledger", record=_rec("CUST_A", "dchen@x.com", "David Chen"))
    r = resolver.resolve("CUST_B", "support_logs", record=_rec("CUST_B", "dchen@x.com", "David Chen"))
    assert r.merged is True and r.resolved_customer_id == "CUST_A"


def test_transitive_closure_via_union_find(resolver) -> None:
    """A<->B (email), B<->C (phone) => A,C share canonical id."""
    from src.identity.graph import IdentityGraph

    resolver.resolve("CUST_A", "core_banking_ledger", record=_rec("CUST_A", "a@x.com", "Ann Lee", phone="555-1111"))
    resolver.resolve("CUST_B", "support_logs", record=_rec("CUST_B", "a@x.com", "Ann Lee"))
    resolver.resolve("CUST_C", "card_payments", record=_rec("CUST_C", "c@y.com", "Ann Lee", phone="555-1111"))

    with psycopg_connect() as conn:
        comps = IdentityGraph(get_settings().pg_dsn).components(conn)
    canon = comps["CUST_A"]
    assert comps["CUST_B"] == canon and comps["CUST_C"] == canon


def test_false_merge_regression_through_resolver(resolver) -> None:
    """Two distinct people sharing a surname must keep separate canonical ids."""
    resolver.resolve("CUST_P", "core_banking_ledger", record=_rec("CUST_P", "d.chen@one.com", "David Chen", dob=date(1985, 3, 10), phone="555-0101"))
    r = resolver.resolve("CUST_Q", "support_logs", record=_rec("CUST_Q", "deb.chen@two.org", "Deborah Chen", dob=date(1979, 11, 2), phone="555-0909"))
    assert r.merged is False and r.resolved_customer_id == "CUST_Q"


def test_annotate_event_carries_identity_metadata(resolver) -> None:
    from src.identity.consumer import annotate_event

    event = _event("CUST_A", "core_banking_ledger")
    annotated = annotate_event(event, resolver)
    assert annotated.customer_id == "CUST_A"
    meta = annotated.payload.get("_identity")
    assert meta is not None and meta["resolved_customer_id"] == "CUST_A"


def _rec(cid, email, name, phone=None, dob=None):
    from src.identity.matcher import IdentityRecord

    return IdentityRecord(cid, "test", full_name=name, email=email, phone=phone, dob=dob)
