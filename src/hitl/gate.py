"""HITL gate (Phase 9) - escalation-on-risk-signal with a real approval gate.

Escalated decisions (guardrail ESCALATE) enter the durable pending_approval
store - execution is *physically* blocked until a human approves. Approved
actions feed the ``past_interventions`` feedback loop on the state board.
"""
from __future__ import annotations

from typing import Any

from src.action.selector import Action, ActionDecision
from src.guardrails.machine import GuardrailResult, Outcome
from src.hitl.store import HitlStore
from src.state.board import CustomerStateBoard
from src.state.models import Intervention


class HitlGate:
    """Routes guardrail outcomes; approved actions feed the feedback loop."""

    def __init__(self, store: HitlStore, board: CustomerStateBoard) -> None:
        self.store = store
        self.board = board

    def route(self, customer_id: str, decision: ActionDecision, guardrail: GuardrailResult, decision_id: str) -> dict[str, Any]:
        """Route one decision through the guardrail outcome.

        Returns the final action + status. BLOCK -> explicit no_action;
        ESCALATE -> durable pending_approval (execution blocked);
        ALLOW -> action proceeds (and is recorded as an intervention).
        """
        if guardrail.outcome == Outcome.BLOCK:
            return {
                "status": "blocked",
                "final_action": Action.NO_ACTION.value,
                "reasons": guardrail.reasons,
            }
        if guardrail.outcome == Outcome.ESCALATE:
            approval_id = self.store.submit(
                customer_id,
                decision_id,
                decision.action,
                {
                    "action": decision.action,
                    "action_confidence": decision.action_confidence,
                    "confidence_band": decision.confidence_band,
                    "reasoning_summary": decision.reasoning_summary,
                    "evidence": decision.evidence,
                    "guardrail_reasons": guardrail.reasons,
                    "tier": guardrail.tier,
                },
            )
            return {
                "status": "pending_approval",  # execution physically blocked
                "final_action": None,
                "approval_id": approval_id,
                "reasons": guardrail.reasons,
                "tier": guardrail.tier,
            }
        # ALLOW: action proceeds; record the intervention (feedback loop)
        self.board.ensure_customer(customer_id)
        self.board.record_intervention(
            customer_id,
            Intervention(decision_id=decision_id, action=decision.action, outcome="executed"),
        )
        return {
            "status": "allowed",
            "final_action": guardrail.final_action,
            "tier": guardrail.tier,
            "reasons": guardrail.reasons,
        }

    def resolve(self, approval_id: str, reviewer: str, action: str = "approve", modified_action: str | None = None, note: str = "") -> dict[str, Any] | None:
        """Human resolution: approve | reject | modify (correct state transitions)."""
        if action == "approve":
            row = self.store.approve(approval_id, reviewer, note)
        elif action == "reject":
            row = self.store.reject(approval_id, reviewer, note)
        elif action == "modify":
            if modified_action is None:
                raise ValueError("modify requires modified_action")
            row = self.store.modify(approval_id, reviewer, modified_action, note)
        else:
            raise ValueError(f"unknown HITL action {action}")
        if row is None:
            return None
        # approved/modified decisions feed past_interventions (feedback loop)
        if row["status"] in {"approved", "modified"}:
            self.board.ensure_customer(row["customer_id"])
            self.board.record_intervention(
                row["customer_id"],
                Intervention(
                    decision_id=row["decision_id"],
                    action=row["final_action"],
                    outcome=f"hitl_{row['status']}",
                ),
            )
        return row
