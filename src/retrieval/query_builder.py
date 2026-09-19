"""Retrieval query builder (Phase 6) - **structurally enforced scoping**.

This turns the PS's named cross-customer-contamination failure mode into a
design constraint (novelty point 5, docs/reference/06): an unscoped retrieval
call is a *type error* - ``RetrievalQuery`` refuses to construct without either
a ``customer_id`` or an explicit ``global_scope=True``, and the SQL it emits
always carries the scoping predicate. No caller can forget the WHERE clause.
"""
from __future__ import annotations

from dataclasses import dataclass


class UnscopedQueryError(ValueError):
    """Raised when a retrieval query would scan across customers."""


@dataclass(frozen=True)
class RetrievalQuery:
    """A retrieval query that CANNOT be constructed unscoped."""

    text: str
    customer_id: str | None = None
    global_scope: bool = False  # policy/reference retrieval only, explicit
    top_k: int = 8
    include_global_chunks: bool = True  # scoped queries may see policy chunks

    def __post_init__(self) -> None:
        if self.customer_id is None and not self.global_scope:
            raise UnscopedQueryError(
                "retrieval query must be scoped to a customer_id OR explicitly "
                "declared global_scope=True - unscoped cross-customer retrieval "
                "is structurally rejected"
            )
        if not self.text or not self.text.strip():
            raise UnscopedQueryError("retrieval query text must not be empty")

    @property
    def scope_clause(self) -> str:
        """Mandatory scoping predicate - always present in the emitted SQL."""
        if self.global_scope:
            return "customer_id IS NULL"  # policy/reference chunks only
        # customer-scoped (+ optionally global policy chunks); NEVER other customers
        if self.include_global_chunks:
            return "(customer_id = %(customer_id)s OR customer_id IS NULL)"
        return "customer_id = %(customer_id)s"


def scoped_query(text: str, customer_id: str, top_k: int = 8, include_global_chunks: bool = True) -> RetrievalQuery:
    """Build a customer-scoped query (the normal path for signal agents)."""
    return RetrievalQuery(text=text, customer_id=customer_id, top_k=top_k, include_global_chunks=include_global_chunks)


def global_query(text: str, top_k: int = 8) -> RetrievalQuery:
    """Build an explicitly global (policy/reference) query."""
    return RetrievalQuery(text=text, global_scope=True, top_k=top_k)
