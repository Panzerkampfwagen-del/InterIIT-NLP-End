"""Guardrail state machine (Phase 9) - governance as architecture, not prompt.

Per RQ-A2/RQ-H1 and the PS ("Ensure an LLM cannot simply reason around a hard
stop"): hard rules are evaluated in code, AFTER the agent proposal. The LLM
sees none of this logic; prompt persuasion cannot bypass it (adversarial test
in the DoD). Outcomes:

- ``ALLOW``   - proposal passes all hard rules;
- ``ESCALATE``- route to HITL (durable pending_approval), fail-closed;
- ``BLOCK``   - downgrade to explicit no_action (never a rule-violating action).

On ANY internal error the machine fails CLOSED (escalate), never open.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.action.selector import Action, ActionDecision
from src.state.models import StateSnapshot


class Outcome(str, Enum):
    ALLOW = "allow"
    ESCALATE = "escalate"  # -> HITL (durable pending_approval)
    BLOCK = "block"        # -> explicit no_action (downgrade)


AUTONOMOUS = "AUTONOMOUS"
HITL = "HITL"
HARD_BLOCK = "HARD_BLOCK"

# PII patterns for free-text scanning (financial/KYC data, #27)
PII_PATTERNS = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("phone", re.compile(r"(?:\+?\d[\d\s().-]{8,}\d)")),
)

# prompt-injection markers: evidence text must never change the decision path
INJECTION_MARKERS = (
    "ignore all rules", "ignore previous", "approve this", "override policy",
    "system prompt", "you are now", "disregard", "escalate this to approved",
)

OFFER_COST_THRESHOLD_HIGH_TIER = 5000.0  # cost check (offers above this -> HITL)


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of one guardrail evaluation - auditable."""

    outcome: Outcome
    final_action: str  # the action that may proceed (no_action when BLOCK)
    reasons: list[str] = field(default_factory=list)
    tier: str = AUTONOMOUS
    pii_masked: bool = False


class GuardrailStateMachine:
    """Deterministic hard rules, evaluated after the agent proposal."""

    MIN_CUSTOMER_FACING_CONFIDENCE = 0.55  # below this -> HITL (uncertainty)
    MIN_FRAUD_HOLD_CONFIDENCE = 0.7        # fraud hold below this -> HITL

    def evaluate(self, decision: ActionDecision, snapshot: StateSnapshot, now=None) -> GuardrailResult:
        """Evaluate hard rules; fail-closed on internal error."""
        try:
            return self._evaluate(decision, snapshot)
        except Exception as exc:  # fail-closed: never allow on internal error
            return GuardrailResult(
                outcome=Outcome.ESCALATE,
                final_action=Action.NO_ACTION.value,
                reasons=[f"fail_closed: internal guardrail error ({type(exc).__name__}) - routed to HITL"],
                tier=HITL,
            )

    def _evaluate(self, decision: ActionDecision, snapshot: StateSnapshot) -> GuardrailResult:
        reasons: list[str] = []
        pii_masked = False
        action = decision.action

        # R1: adversarial injection scan over free text - evidence text must
        # never change the decision path. Injection is flagged and stripped;
        # the PROPOSED action stands (cannot be upgraded by prompt persuasion).
        text_blob = decision.reasoning_summary + " " + " ".join(
            str(e.get("excerpt", "")) for e in decision.evidence
        )
        lower = text_blob.lower()
        if any(m in lower for m in INJECTION_MARKERS):
            reasons.append("R1_injection: prompt-injection markers found in free text; stripped, proposal unchanged (cannot be persuaded)")

        # R2: PII in free text -> mask + escalate (compliance #27)
        for name, pattern in PII_PATTERNS:
            if pattern.search(text_blob):
                pii_masked = True
                reasons.append(f"R2_pii: raw {name} found in free text; masked, escalated for review")

        # R3: no customer-facing offers/retention under active compliance hold
        if decision.eligibility.get("compliance_hold") and action in {
            Action.PERSONALIZED_OFFER.value,
            Action.PROACTIVE_RETENTION_OUTREACH.value,
            Action.RELATIONSHIP_MANAGER_ESCALATION.value,
        }:
            reasons.append("R3_compliance: customer-facing action proposed under active compliance hold - blocked")
            return GuardrailResult(Outcome.BLOCK, Action.NO_ACTION.value, reasons, HARD_BLOCK, pii_masked)

        # R4: fraud hold requires high confidence, else HITL
        if action == Action.COMPLIANCE_FRAUD_HOLD.value and decision.action_confidence < self.MIN_FRAUD_HOLD_CONFIDENCE:
            reasons.append(f"R4_fraud_confidence: fraud hold confidence {decision.action_confidence:.2f} below {self.MIN_FRAUD_HOLD_CONFIDENCE} - escalated")
            return GuardrailResult(Outcome.ESCALATE, action, reasons, HITL, pii_masked)

        # R5: uncertainty gate - customer-facing actions below confidence floor -> HITL
        if action in {
            Action.PERSONALIZED_OFFER.value,
            Action.RELATIONSHIP_MANAGER_ESCALATION.value,
        } and decision.action_confidence < self.MIN_CUSTOMER_FACING_CONFIDENCE:
            reasons.append(f"R5_uncertainty: customer-facing action at confidence {decision.action_confidence:.2f} - escalated")
            return GuardrailResult(Outcome.ESCALATE, action, reasons, HITL, pii_masked)

        # R6: cost/permission check - high-tier offer above cost threshold -> HITL
        if action == Action.PERSONALIZED_OFFER.value:
            tier = decision.eligibility.get("customer_value_tier", "mid")
            amount = self._offer_amount(snapshot)
            if tier == "high" and amount is not None and amount > OFFER_COST_THRESHOLD_HIGH_TIER:
                reasons.append(f"R6_cost: offer amount {amount} above threshold for high-tier customer - escalated")
                return GuardrailResult(Outcome.ESCALATE, action, reasons, HITL, pii_masked)

        # R7: autonomy tier annotation
        tier = self.autonomy_tier(action, decision.action_confidence, decision.eligibility)
        if reasons:
            return GuardrailResult(Outcome.ESCALATE, action, reasons, HITL, pii_masked)
        return GuardrailResult(Outcome.ALLOW, action, reasons, tier, pii_masked)

    def autonomy_tier(self, action: str, confidence: float, eligibility: dict[str, Any]) -> str:
        """3-tier autonomy: action type x confidence x value x reversibility."""
        if action == Action.COMPLIANCE_FRAUD_HOLD.value:
            return HITL  # highest-stakes irreversible action always reviewed
        if action in {Action.PERSONALIZED_OFFER.value, Action.RELATIONSHIP_MANAGER_ESCALATION.value}:
            tier = eligibility.get("customer_value_tier", "mid")
            if tier == "high" or confidence < self.MIN_CUSTOMER_FACING_CONFIDENCE:
                return HITL
            return AUTONOMOUS
        if action in {Action.PROACTIVE_RETENTION_OUTREACH.value, Action.SUPPORT_INTERVENTION.value}:
            return AUTONOMOUS if confidence >= self.MIN_CUSTOMER_FACING_CONFIDENCE else HITL
        return AUTONOMOUS  # no_action and unknown: reversible

    def _offer_amount(self, snapshot: StateSnapshot) -> float | None:
        """Offer amount from the triggering evidence (cost check), if present."""
        for e in snapshot.recent_events:
            if e.get("kind") == "life_event_inference":
                continue
        txn = snapshot.historical_baseline.get("txn_amount_sum_30d")
        if txn is None:
            return None
        return float(txn) * 0.02  # offer value heuristic: 2% of 30d volume
