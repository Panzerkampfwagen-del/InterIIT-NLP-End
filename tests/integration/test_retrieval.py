"""Phase 6 integration: incremental indexing, hybrid search, zero leakage."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.config import get_settings


def _infra() -> bool:
    import socket

    s = get_settings()
    host, port = s.pg_dsn.split("//")[1].split("@")[1].split("/")[0].split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2.0):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _infra(), reason="Postgres not running")


@pytest.fixture()
def env():
    from src.retrieval.indexer import RetrievalIndexer
    from src.retrieval.search import HybridSearcher

    ix = RetrievalIndexer()
    ix.init_schema()
    hs = HybridSearcher()
    with ix._connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE retrieval_chunks RESTART IDENTITY")
    yield ix, hs
    ix.close()
    hs.close()


NOW = datetime.now(UTC)


def _ticket_event(cid, text, event_id="EVT_000001", event_time=None):
    from src.ingestion.schemas.events import parse_event

    return parse_event(
        {
            "event_id": event_id,
            "event_time": event_time or NOW,
            "ingestion_time": event_time or NOW,
            "customer_id": cid,
            "account_id": None,
            "source_system": "support_logs",
            "event_type": "ticket_created",
            "schema_version": "1.0",
            "payload": {"channel": "chat", "category": "payment_arrangements", "raw_text": text, "resolution_status": "open"},
        }
    )


def test_fresh_ticket_indexed_and_retrievable(env) -> None:
    """Freshness (RQ-R2 #16): a ticket indexed 30s ago is retrievable."""
    ix, hs = env
    event = _ticket_event("CUST_A", "I have been in the hospital and my income dropped. Can I set up a payment plan?", event_time=NOW - timedelta(seconds=30))
    n = ix.index_event(event)
    assert n >= 1
    hits = hs.search(__import__("src.retrieval.query_builder", fromlist=["scoped_query"]).scoped_query("medical hardship payment plan", "CUST_A"), now=NOW)
    assert len(hits) >= 1
    assert hits[0].source == "support_logs"
    assert "hospital" in hits[0].chunk_text


def test_zero_cross_customer_leakage(env) -> None:
    """Adversarial isolation: customer A's query never returns B's chunks."""
    ix, hs = env
    from src.retrieval.query_builder import scoped_query

    ix.index_event(_ticket_event("CUST_B", "Standing instruction mortgage schedule question", event_id="EVT_B1"))
    ix.index_event(_ticket_event("CUST_A", "hospital hardship payment plan request", event_id="EVT_A1"))
    hits = hs.search(scoped_query("mortgage standing instruction", "CUST_A"), now=NOW)
    sources = {h.customer_id for h in hits}
    assert "CUST_B" not in sources, "CROSS-CUSTOMER LEAKAGE"
    # all hits are either CUST_A's or global (policy) chunks
    assert all(h.customer_id in ("CUST_A", None) for h in hits)


def test_unscoped_search_impossible_in_searcher(env) -> None:
    """The searcher requires a RetrievalQuery - unscoped queries cannot exist."""
    from src.retrieval.query_builder import UnscopedQueryError

    with pytest.raises(UnscopedQueryError):
        __import__("src.retrieval.query_builder", fromlist=["RetrievalQuery"]).RetrievalQuery(text="anything")


def test_recency_reranking_favors_fresh_chunk(env) -> None:
    """RQ-R2: score = similarity - lambda*age; fresh chunk outranks old equal-relevance."""
    ix, hs = env
    from src.retrieval.query_builder import scoped_query

    ix.index_event(_ticket_event("CUST_A", "hardship plan payment arrangement", event_id="EVT_OLD", event_time=NOW - timedelta(days=90)))
    ix.index_event(_ticket_event("CUST_A", "hardship plan payment arrangement", event_id="EVT_NEW", event_time=NOW - timedelta(seconds=10)))
    hits = hs.search(scoped_query("hardship plan", "CUST_A"), now=NOW)
    assert len(hits) >= 2
    by_source_id = {h.source_id for h in hits}
    assert "EVT_NEW" in by_source_id
    # the fresh chunk scores strictly higher than the 90-day-old identical chunk
    scores = {h.source_id: h.final_score for h in hits}
    assert scores["EVT_NEW"] > scores["EVT_OLD"]


def test_policy_doc_global_and_versioned(env) -> None:
    """Batch path: policy docs indexed globally, retrievable from customer queries."""
    ix, hs = env
    from src.retrieval.query_builder import global_query, scoped_query

    ix.index_policy_doc("offer_policy", "Personalized offers require eligibility and no active compliance hold", version="v1")
    # global query retrieves it
    hits = hs.search(global_query("offer eligibility compliance hold"), now=NOW)
    assert any(h.source == "policy_docs" for h in hits)
    # customer query may include global chunks
    hits = hs.search(scoped_query("offer eligibility", "CUST_A"), now=NOW)
    assert any(h.source == "policy_docs" for h in hits)


def test_upsert_is_incremental_no_duplicate_rows(env) -> None:
    """RQ-R1: same source re-indexed -> upsert (not a new row)."""
    ix, hs = env
    e = _ticket_event("CUST_A", "hardship plan request", event_id="EVT_UP1")
    ix.index_event(e)
    ix.index_event(e)  # same event_id -> same upsert_key
    with ix._connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM retrieval_chunks WHERE source_id = 'EVT_UP1'")
            assert cur.fetchone()[0] == 1


def test_support_agent_indexes_ticket_text(env) -> None:
    """Phase 6 wiring: support agent indexes ticket text on analyze."""
    from src.agents.support_agent import SupportAgent
    from src.state.board import CustomerStateBoard
    from tests.unit.test_signal_agents import _ev, _state

    ix, hs = env
    board = CustomerStateBoard()
    board.init_schema()
    with board._connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE customer_state, state_episodic, state_conflicts RESTART IDENTITY")
    agent = SupportAgent(board=board, llm=None, indexer=ix)
    event = _ev(1, NOW, "support_logs", "ticket_created", {
        "channel": "chat", "category": "payment_arrangements",
        "raw_text": "hardship plan request after hospital stay",
    })
    agent.analyze_and_record(event, _state(), event_time=NOW)
    from src.retrieval.query_builder import scoped_query

    hits = hs.search(scoped_query("hardship plan hospital", "C1"), now=NOW)
    assert any(h.source == "support_logs" for h in hits)
    board.close()
