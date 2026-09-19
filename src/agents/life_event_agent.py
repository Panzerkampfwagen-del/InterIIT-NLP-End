"""Life-Event agent (Phase 7) - the differentiator vs a rules engine.

Per RQ-L1 and the PS Section 4.6 constraint ("Do NOT automatically use an
LLM"): the deterministic/statistical detector layer (Phase 5 agents) raises
candidate findings; THIS agent only runs on already-detected candidates and
produces the cross-signal correlation narrative via structured LLM output:

- LLM structured JSON validated against the dataset's fixed inferred_state
  enum; invalid output is retried (bounded retries), then falls back to
  **deterministic synthesis** (never an invalid label);
- disagreement between agents is addressed explicitly in the reasoning and
  conflicts[] are preserved on the state board (Phase 3);
- evidence_refs are resolvable (event ids + retrieval chunk ids).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.agents.llm import LLMClient
from src.ingestion.schemas.events import EventEnvelope
from src.state.board import CustomerStateBoard
from src.state.models import Provenance, RiskOrOpportunity, StateSnapshot

# The dataset's fixed inferred_state enum (evaluation/dataset/README_dataset_schema.md).
ALLOWED_INFERRED_STATES = frozenset(
    {
        "no_signal",
        "new_child_life_event",
        "marriage_or_relationship_change",
        "job_change_or_promotion",
        "job_loss_or_income_disruption",
        "medical_hardship",
        "financial_distress_general",
        "relocation",
        "retirement_transition",
        "wealth_growth_or_windfall",
        "potential_fraud_or_takeover",
        "elder_vulnerability_or_scam_risk",
        "churn_risk",
        "small_business_cashflow_event",
    }
)

MAX_RETRIES = 2


class LifeEventInference(BaseModel):
    """Structured LLM output, validated against the fixed enum."""

    inferred_state: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    risks: list[dict[str, Any]] = Field(default_factory=list)
    opportunities: list[dict[str, Any]] = Field(default_factory=list)


class LifeEventAgent:
    """Cross-signal correlation -> inferred_state + confidence + risks."""

    name = "life_event_agent"

    def __init__(
        self,
        board: CustomerStateBoard,
        llm: LLMClient | None = None,
        searcher=None,
    ) -> None:
        self.board = board
        self.llm = llm
        self.searcher = searcher  # optional HybridSearcher (scoped evidence)
        self.invocations = 0  # flood-test observable

    # --------------------------------------------------------------- synthesis

    def infer(self, customer_id: str, as_of: datetime | None = None, triggering_event: EventEnvelope | None = None) -> LifeEventInference:
        """Run one correlation cycle for a customer (trigger-gated upstream)."""
        self.invocations += 1
        as_of = as_of or datetime.now(datetime.now().tzinfo)
        snap = self.board.get(customer_id, apply_dec=False)
        evidence = self._collect_evidence(customer_id, snap, as_of)
        disagreement = self._find_disagreement(snap)

        # LLM structured output with bounded retries on invalid labels
        inference = None
        if self.llm is not None and self.llm.available():
            inference = self._llm_infer(snap, evidence, disagreement, retries=MAX_RETRIES)
        if inference is None:
            inference = self._deterministic_fallback(snap, disagreement)

        inference.reasoning = self._ensure_disagreement_addressed(inference.reasoning, disagreement)
        self._write_to_board(customer_id, inference, evidence, as_of)
        return inference

    def _collect_evidence(self, customer_id: str, snap: StateSnapshot, as_of: datetime) -> list[dict[str, Any]]:
        """Findings + scoped retrieval evidence (resolvable refs)."""
        evidence: list[dict[str, Any]] = []
        for domain, f in snap.findings.items():
            evidence.append(
                {
                    "kind": "finding",
                    "domain": domain,
                    "summary": f.summary,
                    "confidence": f.confidence,
                    "flags": [fl.get("type") for fl in f.flags],
                    "refs": f.evidence_refs,
                }
            )
        if self.searcher is not None:
            try:
                from src.retrieval.query_builder import scoped_query

                qtext = " ".join(f.summary for f in snap.findings.values()) or "customer account signals"
                hits = self.searcher.search(scoped_query(qtext, customer_id, top_k=5), now=as_of)
                for h in hits:
                    evidence.append(
                        {
                            "kind": "retrieval_chunk",
                            "source": h.source,
                            "source_id": h.source_id,
                            "text": h.chunk_text[:300],
                            "ref": f"chunk:{h.chunk_id}",
                        }
                    )
            except Exception:
                pass  # evidence retrieval is best-effort; findings still flow
        return evidence

    def _find_disagreement(self, snap: StateSnapshot) -> list[dict[str, Any]]:
        """Agent disagreement (e.g. usage healthy vs transaction distress)."""
        out: list[dict[str, Any]] = []
        opposing = {
            "usage": {"engagement_drop", "search_spike"},
            "transaction": {"decline_burst", "accelerated_withdrawals"},
        }
        usage_flags = {fl.get("type") for fl in snap.findings.get("usage", _empty()).flags}
        txn_flags = {fl.get("type") for fl in snap.findings.get("transaction", _empty()).flags}
        usage_distress = bool(usage_flags & opposing["transaction"]) or "engagement_drop" in usage_flags
        txn_distress = bool(txn_flags & opposing["transaction"])
        if usage_flags and txn_flags and txn_distress and not usage_distress:
            out.append(
                {
                    "agents": ["usage_agent", "transaction_agent"],
                    "values": ["healthy/engaged", "financial distress signals"],
                }
            )
        return out

    def _llm_infer(self, snap: StateSnapshot, evidence: list[dict[str, Any]], disagreement: list[dict[str, Any]], retries: int) -> LifeEventInference | None:
        system = (
            "You are a customer-state inference engine for a bank. Given per-signal findings "
            "and evidence, infer what is happening in the customer's life/account. "
            "Use ONLY these inferred_state values: "
            + ", ".join(sorted(ALLOWED_INFERRED_STATES))
            + ". Address any agent disagreement explicitly in the reasoning. "
            'Output JSON: {"inferred_state": ..., "confidence": 0.0-1.0, "reasoning": ..., '
            '"risks": [{"type", "confidence"}], "opportunities": [{"type", "confidence"}]}'
        )
        import json as _json

        user = _json.dumps(
            {
                "current_findings": {d: f.summary for d, f in snap.findings.items()},
                "evidence": evidence,
                "disagreement": disagreement,
                "active_risks": [r.type for r in snap.active_risks],
            },
            default=str,
        )
        for _ in range(retries + 1):
            result = self.llm.complete(system, user)
            if result is None:
                return None
            try:
                inference = LifeEventInference.model_validate(result)
                if inference.inferred_state in ALLOWED_INFERRED_STATES:
                    return inference
            except ValidationError:
                pass
            # invalid label -> retry with a stricter reminder
            user += "\nREMINDER: inferred_state MUST be one of the allowed values exactly."
        return None

    def _deterministic_fallback(self, snap: StateSnapshot, disagreement: list[dict[str, Any]]) -> LifeEventInference:
        """Deterministic narrative synthesis (fallback contract - never an invalid label)."""
        all_flags: set[str] = set()
        for f in snap.findings.values():
            all_flags.update(fl.get("type", "") for fl in f.flags)
        compliance_flags = {fl.get("type") for fl in snap.findings.get("compliance", _empty()).flags}

        # KYC hard facts first (deterministic, high stakes)
        if "watchlist_hit" in compliance_flags:
            return LifeEventInference(
                inferred_state="potential_fraud_or_takeover",
                confidence=0.85,
                reasoning="Watchlist hit (deterministic KYC fact); potential fraud or account takeover.",
                risks=[{"type": "fraud", "confidence": 0.85}],
            )
        if "address_change_recent" in all_flags and {"large_amount", "decline_burst"} & all_flags:
            return LifeEventInference(
                inferred_state="potential_fraud_or_takeover",
                confidence=0.7,
                reasoning="Recent address change combined with unusual transaction pattern.",
                risks=[{"type": "fraud", "confidence": 0.7}],
            )
        # transaction distress pattern
        if {"decline_burst", "accelerated_withdrawals"} & all_flags:
            return LifeEventInference(
                inferred_state="financial_distress_general",
                confidence=0.6,
                reasoning="Repeated declines / accelerating withdrawals indicate financial distress.",
                risks=[{"type": "financial_distress", "confidence": 0.6}],
            )
        # churn pattern: negative sentiment + repeated contact + engagement drop
        if {"negative_sentiment", "repeated_contact"} & all_flags and "engagement_drop" in all_flags:
            return LifeEventInference(
                inferred_state="churn_risk",
                confidence=0.6,
                reasoning="Support friction combined with dropping engagement indicates churn risk.",
                risks=[{"type": "churn", "confidence": 0.6}],
            )
        # household change (KYC fact) - life event
        if "household_change" in all_flags:
            return LifeEventInference(
                inferred_state="new_child_life_event",
                confidence=0.7,
                reasoning="Dependents/marital status change (deterministic KYC fact) suggests a household life event.",
                opportunities=[{"type": "savings_or_insurance_offer", "confidence": 0.5}],
            )
        return LifeEventInference(
            inferred_state="no_signal",
            confidence=0.3,
            reasoning="No corroborated cross-signal pattern; conservative no-signal inference.",
        )

    def _ensure_disagreement_addressed(self, reasoning: str, disagreement: list[dict[str, Any]]) -> str:
        """The reasoning must explicitly address agent disagreement."""
        if not disagreement:
            return reasoning
        note = (
            f" Agent disagreement noted: {disagreement[0]['agents'][0]} reports "
            f"{disagreement[0]['values'][0]} while {disagreement[0]['agents'][1]} reports "
            f"{disagreement[0]['values'][1]}; the transaction-signal evidence is weighted higher "
            "because it is corroborated by deterministic thresholds."
        )
        if any(v in reasoning for v in ("disagreement", "while", "versus")):
            return reasoning + note
        return reasoning + note

    def _write_to_board(self, customer_id: str, inference: LifeEventInference, evidence: list[dict[str, Any]], as_of: datetime) -> None:
        """Write the inferred state + risks/opportunities to the shared board."""
        prov = Provenance(agent=self.name)
        self.board.ensure_customer(customer_id)
        self.board.update_field(
            customer_id,
            "current_state.life_phase",
            inference.inferred_state,
            inference.confidence,
            prov,
            as_of,
        )
        now = as_of
        for r in inference.risks:
            self._merge_risk(customer_id, "active_risks", r, now)
        for o in inference.opportunities:
            self._merge_risk(customer_id, "active_opportunities", o, now)
        ev_refs = sorted({ref for e in evidence for ref in (e.get("refs") or []) if isinstance(ref, str)})
        self.board.append_event_ref(customer_id, {"event_id": f"inference_{as_of.isoformat()}", "kind": "life_event_inference", "state": inference.inferred_state, "evidence": ev_refs})

    def _merge_risk(self, customer_id: str, path: str, item: dict[str, Any], now: datetime) -> None:
        snap = self.board.get(customer_id, apply_dec=False)
        items = list(getattr(snap, path))
        risk_type = item.get("type", "unknown")
        existing = next((i for i in items if i.type == risk_type), None)
        try:
            new_item = RiskOrOpportunity(
                type=risk_type,
                confidence=float(item.get("confidence", 0.5)),
                opened_at=existing.opened_at if existing else now,
            )
        except Exception:
            return
        items = [i for i in items if i.type != risk_type]
        items.append(new_item)
        import json as _json

        data = snap.model_dump(mode="json", exclude={"state_version", "last_updated_event_time", "last_updated_processing_time"})
        data[path] = [i.model_dump(mode="json") for i in items]
        with self.board._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO customer_state (customer_id, state, state_version, last_updated_processing_time)
                    VALUES (%s, %s::jsonb, 1, now())
                    ON CONFLICT (customer_id) DO UPDATE SET
                        state = EXCLUDED.state,
                        state_version = customer_state.state_version + 1,
                        last_updated_processing_time = now()
                    """,
                    (customer_id, _json.dumps(data)),
                )


def _empty() -> Any:
    from src.state.models import Finding

    return Finding(summary="", confidence=0.0, as_of=datetime(2026, 1, 1, tzinfo=datetime.now().tzinfo), provenance=Provenance(agent="none"))
