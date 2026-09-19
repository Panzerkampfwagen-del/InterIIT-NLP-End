"""Identity resolver (Phase 2) - the Customer Identification step.

Resolves a raw source-system customer id (plus any record attributes carried
by the event payload) to one canonical ``customer_id`` via:

1. cache/registry lookup (already-seen ids keep their canonical mapping);
2. deterministic exact-match on email/phone/KYC doc (RQ-I1 pipeline order);
3. Fellegi-Sunter probabilistic fallback - merge only above threshold, with
   the confidence stored and auditable in the identity graph.

Events whose payload carries identity attributes (loan_kyc changes, support
transcripts with contact info) enrich the graph; events without attributes
resolve through the registry only. **Never merges below threshold** - the
false-merge protection is structural, not prompt-level.
"""
from __future__ import annotations

from dataclasses import dataclass

import psycopg

from src.identity.graph import IdentityGraph
from src.identity.matcher import IdentityRecord, MatchMethod, match


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving one raw id - always auditable."""

    resolved_customer_id: str
    match_method: MatchMethod
    confidence: float
    merged: bool  # True when this call created a new graph edge


def extract_identity_record(event) -> IdentityRecord | None:
    """Pull identity attributes out of an event payload, when present.

    loan_kyc address/marital/dependents changes carry old/new values; support
    transcripts may mention contact channels. Returns None when the payload
    carries no usable identity attributes.
    """
    payload = event.payload or {}
    source = event.source_system.value if hasattr(event.source_system, "value") else str(event.source_system)

    if source == "loan_kyc":
        return IdentityRecord(
            source_customer_id=event.customer_id,
            source_system=source,
            email=_as_str(payload.get("new_value")) if payload.get("event_subtype") == "email_change" else None,
            phone=_as_str(payload.get("new_value")) if payload.get("event_subtype") == "phone_change" else None,
        )
    if source == "support_logs":
        text = payload.get("raw_text") or ""
        email = _find_email(text)
        phone = _find_phone(text)
        if email or phone:
            return IdentityRecord(source_customer_id=event.customer_id, source_system=source, email=email, phone=phone)
    return None


def _as_str(v) -> str | None:
    return str(v) if v is not None else None


def _find_email(text: str) -> str | None:
    import re

    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    return m.group(0) if m else None


def _find_phone(text: str) -> str | None:
    import re

    m = re.search(r"(?:\+?\d[\d\s().-]{8,}\d)", text)
    return m.group(0) if m else None


class IdentityResolver:
    """Stateful resolver over the identity graph (Postgres-backed)."""

    def __init__(self, graph: IdentityGraph, dsn: str) -> None:
        self.graph = graph
        self.dsn = dsn
        self._conn: psycopg.Connection | None = None

    def _connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    def resolve(
        self,
        source_customer_id: str,
        source_system: str,
        record: IdentityRecord | None = None,
    ) -> Resolution:
        """Resolve one raw id to its canonical customer id."""
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT record_id, canonical_customer_id FROM identity_records WHERE source_system = %s AND source_customer_id = %s",
                (source_system, source_customer_id),
            )
            row = cur.fetchone()
        if row is not None:
            return Resolution(row[1], MatchMethod.FIRST_SEEN, 1.0, merged=False)

        canonical = source_customer_id
        merged = False
        method, confidence = MatchMethod.FIRST_SEEN, 1.0

        if record is not None:
            # deterministic / probabilistic candidate lookup over existing records
            candidates = self._candidates(record)
            best: tuple[int, str, object] | None = None  # (record_id, canonical, match_result)
            for rid, cand in candidates:
                if cand.source_customer_id == source_customer_id:
                    continue
                result = match(record, cand)
                if result.matched and (best is None or result.confidence > best[2].confidence):
                    best = (rid, cand.source_customer_id, result)
            if best is not None:
                canonical = best[1]
                result = best[2]
                method, confidence, merged = result.method, result.confidence, True
                new_rid = self.graph.register_first_seen(conn, record, canonical)
                self.graph.add_edge(conn, best[0], new_rid, result)
                return Resolution(canonical, method, confidence, merged=True)

        self.graph.register_first_seen(conn, record or IdentityRecord(source_customer_id, source_system), canonical)
        return Resolution(canonical, method, confidence, merged=merged)

    def _candidates(self, record: IdentityRecord) -> list[tuple[int, IdentityRecord]]:
        """Blocking: candidate records sharing any deterministic key or name prefix.

        Brute-force pairwise comparison is quadratic (RQ-I1); blocking on
        email/phone/doc/lastname keeps the candidate set small.
        """
        conn = self._connection()
        clauses: list[str] = []
        params: list[str] = []
        if record.canonical_email:
            clauses.append("lower(email) = %s")
            params.append(record.canonical_email)
        if record.canonical_phone:
            clauses.append("phone = %s")
            params.append(record.canonical_phone)
        if record.kyc_doc_number:
            clauses.append("kyc_doc_number = %s")
            params.append(record.kyc_doc_number)
        if record.full_name:
            clauses.append("lower(full_name) = lower(%s)")
            params.append(record.full_name)
        if not clauses:
            return []
        where = " OR ".join(clauses)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT record_id, source_system, source_customer_id, full_name, email, phone, dob, kyc_doc_number
                FROM identity_records WHERE {where} LIMIT 50
                """,
                params,
            )
            rows = cur.fetchall()
        out: list[tuple[int, IdentityRecord]] = []
        for rid, _src, scid, name, email, phone, dob, doc in rows:
            out.append((rid, IdentityRecord(scid, _src, name, email, phone, dob, doc)))
        return out

    def annotate(self, event):
        """Return (resolved_customer_id, confidence, method) for one event."""
        record = extract_identity_record(event)
        source = event.source_system.value if hasattr(event.source_system, "value") else str(event.source_system)
        resolution = self.resolve(event.customer_id, source, record)
        return resolution.resolved_customer_id, resolution.confidence, resolution.match_method.value
