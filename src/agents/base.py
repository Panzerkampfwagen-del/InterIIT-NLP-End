"""Signal agent base (Phase 5).

Every signal agent follows the same contract (per RQ-L1 and the PS):

- **deterministic/statistical detection first** - flags come from rules,
  thresholds, and z-scores against baselines; the LLM is never the detector;
- **LLM only for narrative synthesis** over already-detected candidates;
- findings written to the shared state board with provenance and evidence
  refs (resolvable to events), via ``record_finding``;
- **fallback contract**: LLM unavailable ⇒ the deterministic summary flows
  onward - a finding is never lost to an LLM failure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.agents.llm import LLMClient
from src.ingestion.schemas.events import EventEnvelope
from src.state.board import CustomerStateBoard
from src.state.models import Finding, Provenance


class SignalAgent(ABC):
    """Base class for the four specialist signal agents."""

    name: str
    domain: str  # findings.<domain> on the state board

    def __init__(self, board: CustomerStateBoard, llm: LLMClient | None = None) -> None:
        self.board = board
        self.llm = llm

    @abstractmethod
    def detect(self, event: EventEnvelope, state) -> list[dict[str, Any]]:
        """Deterministic/statistical detection: return flag dicts.

        Each flag: {"type", "confidence", "evidence": [event ids], "detail"}.
        Never calls the LLM.
        """

    def narrate(self, event: EventEnvelope, flags: list[dict[str, Any]]) -> str:
        """Deterministic summary; LLM enhances the narrative when available.

        Fallback contract: LLM failure keeps the deterministic summary.
        """
        base = self._deterministic_summary(event, flags)
        if self.llm is None or not flags:
            return base
        enhanced = self._llm_narrative(event, flags)
        return enhanced if enhanced is not None else base

    def _deterministic_summary(self, event: EventEnvelope, flags: list[dict[str, Any]]) -> str:
        if not flags:
            return f"{self.domain}: no notable signals."
        parts = [f"{f['type']} (conf {f['confidence']:.2f})" for f in flags]
        return f"{self.domain}: {len(flags)} flag(s): " + "; ".join(parts)

    def _llm_narrative(self, event: EventEnvelope, flags: list[dict[str, Any]]) -> str | None:
        if self.llm is None:
            return None
        system = (
            "You are a bank signal-analysis summarizer. Given detected signal flags, "
            "write a 2-sentence factual summary of what is happening in this customer's "
            "account. Do not invent signals that are not listed. Output JSON: {\"summary\": ...}"
        )
        user = f"Agent: {self.name}. Detected flags: {json_dumps(flags)}. Latest event: {event.source}/{event.event_type}."
        result = self.llm.complete(system, user)
        if result is None:
            return None
        summary = result.get("summary")
        return summary if isinstance(summary, str) and summary.strip() else None

    def analyze_and_record(
        self,
        event: EventEnvelope,
        state,
        event_time: datetime | None = None,
    ) -> Finding:
        """Detect → narrate → write finding to the shared state board."""
        flags = self.detect(event, state)
        summary = self.narrate(event, flags)
        now = event_time or event.event_time
        finding = Finding(
            summary=summary,
            confidence=max((f["confidence"] for f in flags), default=0.3),
            as_of=now,
            provenance=Provenance(agent=self.name, event_id=event.event_id),
            flags=flags,
            evidence_refs=[eid for f in flags for eid in f.get("evidence", [])],
        )
        self.board.ensure_customer(event.customer_id)
        self.board.record_finding(event.customer_id, self.domain, finding, now)
        return finding


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(_jsonable(value), default=str)


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value
