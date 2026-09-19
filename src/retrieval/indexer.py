"""Incremental retrieval indexer (Phase 6, RQ-R1 / ADR-04).

Streaming path (RQ-R1): as tickets/notes/findings land on the event bus, they
are chunked, embedded, and **upserted** into pgvector - never a full reindex.
Static policy/eligibility documents take a separate batch path and are
versioned (hybrid freshness per the RisingWave/Airbyte finding).

Every chunk carries ``embedding_model_version`` and source timestamps so
evidence citations are resolvable (#24) and a model upgrade can re-embed
selectively.
"""
from __future__ import annotations

from datetime import datetime

import psycopg

from src.config import get_settings
from src.ingestion.schemas.events import EventEnvelope
from src.retrieval.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL_VERSION, embed

MAX_CHUNK_CHARS = 600


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Deterministic whitespace-aware chunking (short corpus; 1-2 chunks)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    words = text.split()
    cur: list[str] = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > max_chars and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class RetrievalIndexer:
    """Incremental chunk + embed + upsert into pgvector."""

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
                    f"""
                    CREATE TABLE IF NOT EXISTS retrieval_chunks (
                        chunk_id                BIGSERIAL PRIMARY KEY,
                        customer_id             TEXT,
                        source                  TEXT NOT NULL,
                        source_id               TEXT NOT NULL,
                        chunk_no                INT NOT NULL DEFAULT 0,
                        chunk_text              TEXT NOT NULL,
                        tsv                     tsvector,
                        embedding               vector({EMBEDDING_DIM}) NOT NULL,
                        embedding_model_version TEXT NOT NULL,
                        event_time              TIMESTAMPTZ NOT NULL,
                        indexed_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
                        upsert_key              TEXT NOT NULL UNIQUE
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_chunks_customer ON retrieval_chunks(customer_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_chunks_tsv ON retrieval_chunks USING gin(tsv)")
                # cosine ANN index (pgvector; upgrade path to a dedicated vector DB per ADR-03)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_chunks_embedding ON retrieval_chunks USING hnsw (embedding vector_cosine_ops)")

    def _upsert(self, customer_id: str | None, source: str, source_id: str, chunks: list[str], event_time: datetime) -> int:
        conn = self._connection()
        count = 0
        with conn.cursor() as cur:
            for no, chunk in enumerate(chunks):
                vec = embed(chunk)
                cur.execute(
                    """
                    INSERT INTO retrieval_chunks
                        (customer_id, source, source_id, chunk_no, chunk_text, tsv, embedding, embedding_model_version, event_time, upsert_key)
                    VALUES (%s, %s, %s, %s, %s,
                            to_tsvector('english', %s),
                            %s::vector, %s, %s, %s)
                    ON CONFLICT (upsert_key) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        tsv = EXCLUDED.tsv,
                        embedding = EXCLUDED.embedding,
                        embedding_model_version = EXCLUDED.embedding_model_version,
                        event_time = EXCLUDED.event_time,
                        indexed_at = now()
                    """,
                    (customer_id, source, source_id, no, chunk, chunk,
                     "[" + ",".join(f"{v:.6f}" for v in vec) + "]",
                     EMBEDDING_MODEL_VERSION, event_time,
                     f"{source}:{source_id}:{no}"),
                )
                count += 1
        return count

    def index_event(self, event: EventEnvelope, text_field: str = "raw_text") -> int:
        """Streaming path: index an event's unstructured text (upsert, incremental)."""
        text = event.payload.get(text_field) if isinstance(event.payload, dict) else None
        if not text or not str(text).strip():
            return 0
        chunks = chunk_text(str(text))
        return self._upsert(event.customer_id, event.source, event.event_id, chunks, event.event_time)

    def index_finding(self, customer_id: str, domain: str, summary: str, event_time: datetime, decision_id: str | None = None) -> int:
        """Index a synthesized finding (RQ-R1: newly-inferred findings upserted)."""
        if not summary.strip():
            return 0
        source_id = f"finding_{domain}_{decision_id or event_time.isoformat()}"
        return self._upsert(customer_id, "findings", source_id, chunk_text(summary), event_time)

    def index_policy_doc(self, doc_id: str, text: str, version: str = "v1") -> int:
        """Batch path for static policy/reference docs (global, versioned)."""
        chunks = chunk_text(text)
        return self._upsert(None, "policy_docs", f"{doc_id}_{version}", chunks, datetime(2026, 1, 1, tzinfo=datetime.now().tzinfo))
