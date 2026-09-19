"""State board data models (Phase 3, per docs/reference/03 Section 8).

Design principles: every field is provenanced (which agent/event produced it),
timestamped (event time, not just processing time), versioned (updates never
silently destroy the previous value), and confidence-scored where inferred.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """Which agent/event produced a field value."""

    agent: str
    decision_id: str | None = None
    event_id: str | None = None


class Finding(BaseModel):
    """A per-signal-agent finding written to the shared state board."""

    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    as_of: datetime
    provenance: Provenance
    flags: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class RiskOrOpportunity(BaseModel):
    """An active risk or opportunity on the state board."""

    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    opened_at: datetime
    evidence_refs: list[str] = Field(default_factory=list)


class StateConflict(BaseModel):
    """A conflicting write preserved for arbitration (never overwritten)."""

    field_path: str
    agents: list[str]
    values: list[Any]
    resolved_by: str | None = None
    resolution: Any = None
    resolved_at: datetime | None = None


class Intervention(BaseModel):
    """A past intervention with its outcome (feedback loop)."""

    decision_id: str
    action: str
    outcome: str
    closed_at: datetime | None = None


class StateSnapshot(BaseModel):
    """The full documented state-board structure for one customer."""

    customer_id: str
    schema_version: str = "1.2"
    current_state: dict[str, Any] = Field(default_factory=dict)
    recent_events: list[dict[str, Any]] = Field(default_factory=list)  # ring buffer, refs only
    findings: dict[str, Finding] = Field(default_factory=dict)
    historical_baseline: dict[str, Any] = Field(default_factory=dict)
    active_risks: list[RiskOrOpportunity] = Field(default_factory=list)
    active_opportunities: list[RiskOrOpportunity] = Field(default_factory=list)
    past_interventions: list[Intervention] = Field(default_factory=list)
    conflicts: list[StateConflict] = Field(default_factory=list)
    eligibility: dict[str, Any] = Field(default_factory=dict)
    recommended_action: dict[str, Any] = Field(default_factory=dict)
    decay_metadata: dict[str, dict[str, float]] = Field(default_factory=dict)
    state_version: int = 0
    last_updated_event_time: datetime | None = None
    last_updated_processing_time: datetime | None = None

    def finding(self, domain: str) -> Finding | None:
        return self.findings.get(domain)
