"""Phase 6: structurally-enforced scoping + embeddings (no DB)."""
from __future__ import annotations

import pytest

from src.retrieval.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL_VERSION, cosine, embed
from src.retrieval.query_builder import (
    RetrievalQuery,
    UnscopedQueryError,
    global_query,
    scoped_query,
)


def test_unscoped_query_structurally_rejected() -> None:
    """The adversarial-isolation core: no caller can construct an unscoped query."""
    with pytest.raises(UnscopedQueryError):
        RetrievalQuery(text="hospital income dropped")
    with pytest.raises(UnscopedQueryError):
        RetrievalQuery(text="x", customer_id=None, global_scope=False)


def test_scoped_query_allowed_and_carries_scope_clause() -> None:
    q = scoped_query("payment plan", "CUST_A")
    assert q.customer_id == "CUST_A"
    clause = q.scope_clause
    assert "customer_id = %(customer_id)s" in clause
    # global chunks allowed via explicit OR clause - but never other customers
    assert "OR customer_id IS NULL" in clause


def test_strict_scoping_excludes_global_chunks() -> None:
    q = scoped_query("payment plan", "CUST_A", include_global_chunks=False)
    assert q.scope_clause == "customer_id = %(customer_id)s"


def test_global_scope_must_be_explicit() -> None:
    q = global_query("offer eligibility policy")
    assert q.global_scope and q.scope_clause == "customer_id IS NULL"


def test_empty_text_rejected() -> None:
    with pytest.raises(UnscopedQueryError):
        scoped_query("  ", "CUST_A")


def test_embeddings_deterministic_and_normalized() -> None:
    v1 = embed("I have been in the hospital and my income dropped")
    v2 = embed("I have been in the hospital and my income dropped")
    assert v1 == v2  # deterministic
    assert len(v1) == EMBEDDING_DIM
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-9  # L2-normalized
    assert EMBEDDING_MODEL_VERSION == "hashed-ngram-v1"


def test_embeddings_similar_text_closer_than_unrelated() -> None:
    v_med = embed("hospital emergency medical hardship income dropped")
    v_q = embed("medical hardship plan income")
    v_unrelated = embed("standing instruction mortgage payment schedule")
    assert cosine(v_med, v_q) > cosine(v_med, v_unrelated)
