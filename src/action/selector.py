"""Action selection engine (Phase 8) - the literal terminal decision.

Bounded enum (dataset's fixed action values), justified selection function,
explicit NO_ACTION - never silence (zero-silence test). Selection is
**deterministic** from (inferred_state, risks, opportunities, eligibility,
past interventions): the LLM never picks the action (bounded enum, #20).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.action.eligibility import compute_eligibility
from src.state.models import StateSnapshot


class Action(str, Enum):
    """The dataset's fixed action enum (bounded - never open-ended text)."""

    NO_ACTION = "no_action"
    PROACTIVE_RETENTION_OUTREACH = "proactive_retention_outreach"
    RELATIONSHIP_MANAGER_ESCALATION = "relationship_manager_escalation"
    PERSONALIZED_OFFER = "personalized_offer"
    SUPPORT_INTERVENTION = "support_intervention"
    COMPLIANCE_FRAUD_HOLD = "compliance_fraud_hold"


ALLOWED_ACTIONS = frozenset(a.value for a in Action)


def confidence_band(confidence: float) -> str:
    """Map confidence to the dataset's fixed confidence_band enum."""
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


@dataclass(frozen=True)
class ActionDecision:
    """The terminal decision object - every cycle produces one (zero silence)."""

    action: str  # bounded enum value
    action_confidence: float
    confidence_band: str
    reasoning_summary: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    eligibility: dict[str, Any] = field(default_factory=dict)
    inferred_state: str | None = None


class ActionSelector:
    """Deterministic action selection with a justified selection function."""

    def select(self, snapshot: StateSnapshot, now: datetime | None = None) -> ActionDecision:
        """Select the action deterministically; NO_ACTION is explicit, never silence."""
        now = now or datetime.now(datetime.now().tzinfo)
        eligibility = compute_eligibility(snapshot, now)
        state = snapshot.current_state.get("life_phase") or "no_signal"
        conf = float(snapshot.current_state.get("life_phase_confidence") or 0.0)
        evidence = self._collect_evidence(snapshot)

        # selection function (ordered, deterministic, documented):
        if eligibility["compliance_hold"] or state == "potential_fraud_or_takeover" and conf >= 0.6:
            return self._decision(
                Action.COMPLIANCE_FRAUD_HOLD, max(conf, 0.8),
                f"Compliance/fraud hold: inferred_state={state} (conf {conf:.2f}) or active compliance hold.",
                evidence, eligibility, state, now,
            )
        if state == "medical_hardship" and conf >= 0.5:
            return self._decision(
                Action.SUPPORT_INTERVENTION, conf,
                f"Medical hardship (conf {conf:.2f}) with support friction justifies a support intervention.",
                evidence, eligibility, state, now,
            )
        if state == "churn_risk" and conf >= 0.5:
            tier = eligibility["customer_value_tier"]
            if tier == "high" and eligibility["retention_outreach"]:
                return self._decision(
                    Action.RELATIONSHIP_MANAGER_ESCALATION, conf,
                    f"Churn risk (conf {conf:.2f}) for a high-value customer requires RM escalation.",
                    evidence, eligibility, state, now,
                )
            if eligibility["retention_outreach"]:
                return self._decision(
                    Action.PROACTIVE_RETENTION_OUTREACH, conf,
                    f"Churn risk (conf {conf:.2f}) warrants proactive retention outreach.",
                    evidence, eligibility, state, now,
                )
            return self._decision(
                Action.NO_ACTION, conf,
                f"Churn risk present (conf {conf:.2f}) but a recent identical intervention exists; holding.",
                evidence, eligibility, state, now,
            )
        if state in {"new_child_life_event", "wealth_growth_or_windfall", "retirement_transition", "marriage_or_relationship_change"} and eligibility["personalized_offer"]:
            return self._decision(
                Action.PERSONALIZED_OFFER, conf,
                f"Life event {state} (conf {conf:.2f}) with offer eligibility justifies a personalized offer.",
                evidence, eligibility, state, now,
            )
        if state == "financial_distress_general" and conf >= 0.6:
            support_flags = snapshot.findings.get("support")
            if support_flags and support_flags.confidence >= 0.5:
                return self._decision(
                    Action.SUPPORT_INTERVENTION, conf,
                    f"Financial distress (conf {conf:.2f}) with open support friction justifies a support intervention.",
                    evidence, eligibility, state, now,
                )
            return self._decision(
                Action.NO_ACTION, conf,
                f"Financial distress signals present (conf {conf:.2f}) but no corroborating support context; holding.",
                evidence, eligibility, state, now,
            )
        # explicit NO_ACTION - never silence
        return self._decision(
            Action.NO_ACTION, conf,
            f"No intervention-worthy condition: inferred_state={state} (conf {conf:.2f}); explicit no-action.",
            evidence, eligibility, state, now,
        )

    def _decision(
        self,
        action: Action,
        confidence: float,
        reasoning: str,
        evidence: list[dict[str, Any]],
        eligibility: dict[str, Any],
        state: str,
        now: datetime,
    ) -> ActionDecision:
        c = round(min(1.0, max(0.0, confidence)), 4)
        return ActionDecision(
            action=action.value,  # validated bounded enum value
            action_confidence=c,
            confidence_band=confidence_band(c),
            reasoning_summary=reasoning,
            evidence=evidence,
            eligibility=eligibility,
            inferred_state=state,
        )

    def _collect_evidence(self, snapshot: StateSnapshot) -> list[dict[str, Any]]:
        """Evidence assembled from stored findings (resolvable refs)."""
        evidence: list[dict[str, Any]] = []
        for domain, f in snapshot.findings.items():
            evidence.append(
                {
                    "source": f"findings.{domain}",
                    "agent": f.provenance.agent,
                    "excerpt": f.summary[:200],
                    "timestamp": f.as_of.isoformat() if f.as_of else None,
                    "refs": f.evidence_refs,
                }
            )
        return evidence
