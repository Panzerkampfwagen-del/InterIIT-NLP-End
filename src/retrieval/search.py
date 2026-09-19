"""Hybrid retrieval (Phase 6, RQ-R2 / ADR-04).

Vector (pgvector cosine) + lexical (Postgres FTS) with **recency re-ranking**:
``final = hybrid - lambda * age_days`` - a fresh but weakly-relevant chunk
does not drown out a highly-relevant older one (RQ-R2 finding). Every query
flows through ``RetrievalQuery`` - unscoped retrieval is structurally rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg

from src.config import get_settings
from src.retrieval.embeddings import embed
from src.retrieval.query_builder import RetrievalQuery


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    customer_id: str | None
    source: str
    source_id: str
    chunk_text: str
    vector_sim: float
    lexical_rank: float
    recency_penalty: float
    final_score: float
    event_time: datetime
    embedding_model_version: str


class HybridSearcher:
    """Hybrid vector+lexical search with recency re-ranking (scoped only)."""

    def __init__(self, dsn: str | None = None, recency_lambda: float | None = None) -> None:
        s = get_settings()
        self.dsn = dsn or s.pg_dsn
        self.recency_lambda = recency_lambda if recency_lambda is not None else s.retrieval_recency_lambda
        self._conn: psycopg.Connection | None = None

    def _connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    def search(self, query: RetrievalQuery, now: datetime | None = None) -> list[SearchHit]:
        """Hybrid search; the query's mandatory scope clause is always applied."""
        now = now or datetime.now(UTC)
        vec = embed(query.text)
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
        params: dict = {"top_k": query.top_k, "vec": vec_literal, "qtext": query.text, "now": now}
        if query.customer_id is not None:
            params["customer_id"] = query.customer_id

        sql = f"""
            WITH scoped AS (
                SELECT chunk_id, customer_id, source, source_id, chunk_text,
                       embedding, event_time, embedding_model_version,
                       1 - (embedding <=> %(vec)s::vector) AS vector_sim,
                       ts_rank(tsv, websearch_to_tsquery('english', %(qtext)s)) AS lexical_rank
                FROM retrieval_chunks
                WHERE {query.scope_clause}
            ), scored AS (
                SELECT *,
                    (0.7 * vector_sim + 0.3 * lexical_rank) AS hybrid
                FROM scoped
            )
            SELECT chunk_id, customer_id, source, source_id, chunk_text,
                   vector_sim, lexical_rank, hybrid, event_time, embedding_model_version
            FROM scored
            ORDER BY (hybrid - %(lambda)s * GREATEST(0, EXTRACT(EPOCH FROM (%(now)s - event_time)) / 86400.0)) DESC
            LIMIT %(top_k)s
        """
        params["lambda"] = self.recency_lambda
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        hits: list[SearchHit] = []
        for row in rows:
            chunk_id, customer_id, source, source_id, chunk_text, vsim, lrank, hybrid, event_time, emv = row
            age_days = max(0.0, (now - event_time).total_seconds() / 86400.0)
            penalty = self.recency_lambda * age_days
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    customer_id=customer_id,
                    source=source,
                    source_id=source_id,
                    chunk_text=chunk_text,
                    vector_sim=float(vsim),
                    lexical_rank=float(lrank),
                    recency_penalty=float(penalty),
                    final_score=float(hybrid) - float(penalty),
                    event_time=event_time,
                    embedding_model_version=emv,
                )
            )
        return hits
