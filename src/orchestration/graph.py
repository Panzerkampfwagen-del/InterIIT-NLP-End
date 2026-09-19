"""Top-level orchestration graph (Phase 13) - the full intended path.

Live/Replayed Event -> Customer Identification -> Streaming Processing ->
Specialized Agent Analysis -> Shared Customer State -> Cross-Agent Correlation
-> Life-Event/Risk/Opportunity Inference -> Action Decision -> Policy/
Guardrails -> HITL when required -> Final Action OR Explicit No-Action ->
Evidence + Explanation + Trace -> Evaluation Output.

Typed state-graph semantics (ADR-01): explicit nodes, deterministic order,
persisted checkpoints (state board + decision log), replayable traces.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from src.action.selector import ActionSelector
from src.agents.kyc_agent import KycAgent
from src.agents.life_event_agent import LifeEventAgent
from src.agents.support_agent import SupportAgent
from src.agents.transaction_agent import TransactionAgent
from src.agents.usage_agent import UsageAgent
from src.config import Settings, get_settings
from src.explainability.explainer import Explainer
from src.guardrails.machine import GuardrailStateMachine
from src.hitl.gate import HitlGate
from src.hitl.store import HitlStore
from src.identity.resolver import IdentityResolver
from src.ingestion.dataset_loader import Scenario
from src.ingestion.schemas.events import EventEnvelope
from src.retrieval.indexer import RetrievalIndexer
from src.retrieval.search import HybridSearcher
from src.state.board import CustomerStateBoard
from src.state.models import Provenance
from src.streaming.baselines import compute_baselines
from src.streaming.consumer import EventTimeConsumer
from src.streaming.consumer import Outcome as StreamOutcome
from src.streaming.features import DEFAULT_SPECS
from src.streaming.watermarks import WatermarkTracker


class CritiqueStep:
    """Deterministic conflict-arbitration node (ADR-01: the narrow
    disagreement-arbitration sub-step; losing values stay in the conflict log)."""

    name = "critique_agent"

    def arbitrate(self, board: CustomerStateBoard, customer_id: str) -> None:
        """Resolve pending conflicts deterministically: the later-proposed
        (transaction-signal) value wins; the loser remains in the log for audit."""
        for conflict in board.pending_conflicts(customer_id):
            values = conflict.values
            resolution = values[1] if len(values) >= 2 else (values[0] if values else None)
            board.resolve_conflict_by_field(customer_id, conflict.field_path, self.name, resolution)


@dataclass
class EventOutcome:
    """One processed event's outcome (per hop, auditable)."""

    event_id: str
    customer_id: str
    identity_resolved: bool
    stream_outcome: str
    agents_run: list[str] = field(default_factory=list)
    trigger_fired: bool = False
    inference: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    decision_id: str | None = None
    trace_id: str | None = None


@dataclass
class ScenarioResult:
    """End-to-end result for one scenario (the demo/evaluation artifact)."""

    scenario_id: str
    customer_id: str
    events_processed: int
    events_failed: int
    decisions: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str | None = None
    final_decision: dict[str, Any] | None = None


_ALLOWED_STATES = frozenset(
    {
        "no_signal", "new_child_life_event", "marriage_or_relationship_change",
        "job_change_or_promotion", "job_loss_or_income_disruption", "medical_hardship",
        "financial_distress_general", "relocation", "retirement_transition",
        "wealth_growth_or_windfall", "potential_fraud_or_takeover",
        "elder_vulnerability_or_scam_risk", "churn_risk", "small_business_cashflow_event",
    }
)


def agent_sources(agent) -> set[str]:
    """The source systems each signal agent consumes."""
    return {
        TransactionAgent: {"card_payments", "core_banking_ledger", "instant_payments", "ach_wire"},
        UsageAgent: {"web_app_events"},
        SupportAgent: {"support_logs"},
        KycAgent: {"loan_kyc"},
    }.get(type(agent), set())


class Customer360Pipeline:
    """Wires every phase into one runnable graph."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm=None,
        board: CustomerStateBoard | None = None,
    ) -> None:
        s = settings or get_settings()
        self.settings = s
        self.board = board or CustomerStateBoard(s.pg_dsn)
        self.board.init_schema()
        self.tracker = WatermarkTracker(s.pg_dsn)
        self.tracker.init_schema()
        self.stream = EventTimeConsumer(self.board, self.tracker, allowed_lateness_s=s.allowed_lateness_s)
        self.resolver = IdentityResolver(self._identity_graph(), s.pg_dsn)
        self.indexer = RetrievalIndexer(s.pg_dsn)
        self.indexer.init_schema()
        self.searcher = HybridSearcher(s.pg_dsn)
        self.life_event = LifeEventAgent(self.board, llm=llm, searcher=self.searcher)
        self.action_selector = ActionSelector()
        self.guardrail = GuardrailStateMachine()
        self.hitl_store = HitlStore(s.pg_dsn)
        self.hitl_store.init_schema()
        self.hitl_gate = HitlGate(self.hitl_store, self.board)
        self.explainer = Explainer(s.pg_dsn)
        self.explainer.init_schema()
        self.critique = CritiqueStep()
        self._agents = [
            TransactionAgent(self.board, llm=None),
            UsageAgent(self.board, llm=None),
            SupportAgent(self.board, llm=None, indexer=self.indexer),
            KycAgent(self.board, llm=None),
        ]
        self._trigger = None  # per-scenario (cooldown state resets)
        self._pending_checkpoints: list[datetime] = []
        self._checkpoint_grace_days = 2.0

    def _identity_graph(self):
        from src.identity.graph import IdentityGraph

        g = IdentityGraph(self.settings.pg_dsn)
        with self.board._connection() as conn:
            g.init_schema(conn)
        return g

    # ------------------------------------------------------------- history seed

    def load_history(self, scenario: Scenario) -> None:
        """Seed the customer's state from history: identity, baselines, ring.

        History seeds the state (findings/baselines/recent events); the
        decision path (trigger -> inference -> action -> HITL) runs on the
        live window only.
        """
        for event in scenario.history_seed:
            resolved, _conf, _method = self.resolver.annotate(event)
            event = event.model_copy(update={"customer_id": resolved})
            self.stream.process_batch([event], processing_time=event.event_time)
            self.board.ensure_customer(resolved)
            self.board.append_event_ref(
                resolved,
                {
                    "event_id": event.event_id,
                    "source_system": event.source,
                    "event_type": event.event_type,
                    "event_time": event.event_time.isoformat(),
                },
            )
            if event.source == "support_logs":
                self.indexer.index_event(event)

        history_events = list(scenario.history_seed)
        if not history_events:
            return
        history_end = max(e.event_time for e in history_events)
        baseline = compute_baselines(
            history_events, DEFAULT_SPECS, history_end=history_end, customer_id=scenario.customer_id
        )
        baseline_doc: dict[str, Any] = {
            "txn_count_30d": baseline.get("txn_count_30d"),
            "txn_count_30d_mean": (baseline.get("txn_count_30d") or {}).get("mean"),
            "txn_count_30d_std": (baseline.get("txn_count_30d") or {}).get("std"),
            **baseline,
            "computed_at": datetime.now(UTC).isoformat(),
        }
        self.board.ensure_customer(scenario.customer_id)
        self.board.update_field(
            scenario.customer_id,
            "historical_baseline",
            baseline_doc,
            None,
            Provenance(agent="streaming_consumer"),
            history_end,
        )

    # -------------------------------------------------------------- event path

    def process_event(self, event: EventEnvelope, decide: bool = True) -> EventOutcome:
        """One event through the full graph (all hops, auditable)."""
        return self._process_event_inner(event, decide)

    def _process_event_inner(self, event: EventEnvelope, decide: bool) -> EventOutcome:
        # 1. Customer Identification (identity resolution)
        resolved, _conf, _method = self.resolver.annotate(event)
        event = event.model_copy(update={"customer_id": resolved})

        # 2. Streaming Processing (dedup, watermark, features)
        stream_results = self.stream.process_batch([event], processing_time=event.event_time)
        stream_outcome = stream_results[0].outcome.value if stream_results else StreamOutcome.DUPLICATE.value
        if stream_outcome == StreamOutcome.DUPLICATE.value:
            return EventOutcome(event.event_id, resolved, True, stream_outcome)

        # ring buffer update (agents detect on recent refs)
        self.board.ensure_customer(resolved)
        self.board.append_event_ref(
            resolved,
            {
                "event_id": event.event_id,
                "source_system": event.source,
                "event_type": event.event_type,
                "event_time": event.event_time.isoformat(),
            },
        )

        # 3. Specialized Agent Analysis (signal agents; findings -> shared state)
        snap = self.board.get(resolved, apply_dec=False)
        agents_run: list[str] = []
        for agent in self._agents:
            if event.source in agent_sources(agent):
                agent.analyze_and_record(event, snap, event_time=event.event_time)
                agents_run.append(agent.name)
                snap = self.board.get(resolved, apply_dec=False)

        outcome = EventOutcome(event.event_id, resolved, True, stream_outcome, agents_run)
        if not decide:
            return outcome

        # 4. Cross-Agent Correlation: trigger evaluator (cost-disciplined invocation)
        trigger = self._current_trigger()
        trig = trigger.evaluate(resolved, as_of=event.event_time)
        outcome.trigger_fired = trig.fire
        if trig.fire:
            # 5. Life-Event Inference + critique (conflict arbitration)
            inference = self.life_event.infer(resolved, as_of=event.event_time, triggering_event=event)
            self.critique.arbitrate(self.board, resolved)
            outcome.inference = {
                "inferred_state": inference.inferred_state,
                "confidence": inference.confidence,
                "reasoning": inference.reasoning,
            }

        # 6. Action Decision -> Policy/Guardrails -> HITL gate -> final decision
        snap = self.board.get(resolved, apply_dec=False)
        decision = self.action_selector.select(snap, now=event.event_time)
        guardrail = self.guardrail.evaluate(decision, snap)
        decision_id = f"dec_{uuid.uuid4().hex[:10]}"
        routed = self.hitl_gate.route(resolved, decision, guardrail, decision_id)

        # 7. Evidence + Explanation + Trace (decision-log snapshot)
        self.explainer.snapshot(
            decision_id=decision_id,
            customer_id=resolved,
            as_of_event_time=event.event_time,
            decision=decision,
            guardrail=guardrail,
            route=routed,
            state_snapshot=snap,
            trace_id=self._trace_id(),
        )
        outcome.decision = {
            "decision_id": decision_id,
            "action": decision.action,
            "action_confidence": decision.action_confidence,
            "confidence_band": decision.confidence_band,
            "reasoning_summary": decision.reasoning_summary,
            "status": routed["status"],
            "final_action": routed.get("final_action") or decision.action,
            "guardrail_reasons": guardrail.reasons,
            "tier": guardrail.tier,
        }
        outcome.decision_id = decision_id
        outcome.trace_id = self._trace_id()
        return outcome

    def _current_trigger(self):
        from src.orchestration.trigger import TriggerEvaluator

        if self._trigger is None:
            self._trigger = TriggerEvaluator(self.board)
        return self._trigger

    def _trace_id(self) -> str:
        return uuid.uuid4().hex

    # ------------------------------------------------------------- scenario run

    def _reset_watermarks(self) -> None:
        """Clear the watermark table for a fresh replay window."""
        with self.tracker._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE watermarks")

    def run_scenario(self, scenario: Scenario, checkpoint_times: list[datetime] | None = None) -> ScenarioResult:
        """Run one scenario end-to-end: history seed -> live stream -> checkpoints.

        Checkpoints (the dataset's inferred-events schema) are emitted when the
        live stream's event time crosses each checkpoint time (+grace).
        """
        # Each scenario replay is an independent stream: reset in-memory dedup
        # and the watermark table (production runs one persistent consumer;
        # replays start a fresh stream window).
        self.stream = EventTimeConsumer(
            self.board, self.tracker, allowed_lateness_s=self.settings.allowed_lateness_s
        )
        self._reset_watermarks()
        self.load_history(scenario)
        self._trigger = None  # fresh cooldown per scenario

        result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            customer_id=scenario.customer_id,
            events_processed=0,
            events_failed=0,
        )

        live = sorted(scenario.live_stream, key=lambda e: e.event_time)
        self._pending_checkpoints = sorted(checkpoint_times or self._default_checkpoints(scenario))

        for event in live:
            self._emit_due_checkpoints(scenario, event.event_time, result)
            outcome = self.process_event(event, decide=True)
            result.events_processed += 1
            if outcome.decision is not None:
                result.decisions.append(outcome.decision)

        self._emit_due_checkpoints(scenario, None, result)  # flush remaining
        result.final_decision = result.decisions[-1] if result.decisions else None
        if result.decisions:
            result.trace_id = result.decisions[-1].get("decision_id")
        return result

    def _default_checkpoints(self, scenario: Scenario) -> list[datetime]:
        """Periodic checkpoints when no ground truth schedules them."""
        start = scenario.replay_config.get("simulated_start")
        end = scenario.replay_config.get("simulated_end")
        if not start or not end:
            return []
        s = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        e = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        out: list[datetime] = []
        cur = s + timedelta(days=14)
        while cur < e:
            out.append(cur)
            cur += timedelta(days=14)
        return out

    def _emit_due_checkpoints(self, scenario: Scenario, event_time: datetime | None, result: ScenarioResult) -> None:
        """Emit checkpoints due at/before event_time (+grace), dataset schema."""
        grace = timedelta(days=self._checkpoint_grace_days)
        remaining: list[datetime] = []
        for cp_time in self._pending_checkpoints:
            due = event_time is None or event_time > cp_time + grace
            if due:
                result.checkpoints.append(self._build_checkpoint(scenario, cp_time))
            else:
                remaining.append(cp_time)
        self._pending_checkpoints = remaining

    def _build_checkpoint(self, scenario: Scenario, cp_time: datetime) -> dict[str, Any]:
        """One checkpoint in the dataset's inferred-events schema."""
        snap = self.board.get(scenario.customer_id, apply_dec=True, now=cp_time)
        decision = self.action_selector.select(snap, now=cp_time)
        state = snap.current_state.get("life_phase") or "no_signal"
        conf = float(snap.current_state.get("life_phase_confidence") or 0.0)
        refs = sorted({r for item in decision.evidence for r in (item.get("refs") or [])})
        return {
            "as_of_time": cp_time.isoformat(),
            "inferred_state": state if state in _ALLOWED_STATES else "no_signal",
            "confidence_band": decision.confidence_band if conf >= 0.1 else "low",
            "action": decision.action,
            "action_subtype": None,
            "hitl_status": "auto_approved",
            "notes": decision.reasoning_summary,
            "evidence_refs": refs,
        }
