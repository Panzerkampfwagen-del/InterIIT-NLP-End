"""Trigger evaluator (Phase 7) - when is reasoning worth its cost?

Per docs/reference/06 novelty point 1 and RQ-A1 (routing calls have real
token/latency cost): the Life-Event agent fires ONLY on a correlation-worthy
condition - never on every event (flood protection via a per-customer cooldown), and not on single weak signals:

1. >= 2 DISTINCT domains updated within the correlation window (fan-in), OR
2. a single finding crosses the high-severity confidence threshold, OR
3. a KYC hard flag (watchlist hit - deterministic, confidence 1.0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from src.state.board import CustomerStateBoard


@dataclass(frozen=True)
class TriggerDecision:
    """Outcome of one trigger evaluation - auditable."""

    fire: bool
    reason: str
    conditions: list[str] = field(default_factory=list)


class TriggerEvaluator:
    """Decides when the Life-Event agent should be invoked for a customer."""

    def __init__(
        self,
        board: CustomerStateBoard,
        correlation_window_min: float | None = None,
        high_severity_conf: float | None = None,
        cooldown_min: float | None = None,
    ) -> None:
        self.board = board
        self.correlation_window = timedelta(minutes=correlation_window_min if correlation_window_min is not None else 15.0)
        self.high_severity_conf = high_severity_conf if high_severity_conf is not None else 0.75
        self.cooldown = timedelta(minutes=cooldown_min if cooldown_min is not None else 30.0)
        self._last_fired: dict[str, datetime] = {}

    def evaluate(self, customer_id: str, as_of: datetime | None = None) -> TriggerDecision:
        """Evaluate trigger conditions against the state board's findings."""
        as_of = as_of or datetime.now(UTC)
        snap = self.board.get(customer_id, apply_dec=False)
        conditions: list[str] = []

        # 1. >= 2 finding updates within the correlation window (any domain;
        # repeated same-domain updates also correlate - the PS's rolling window)
        updated_domains = [
            domain
            for domain, f in snap.findings.items()
            if _age_s(f.as_of, as_of) <= self.correlation_window.total_seconds()
        ]
        if len(updated_domains) >= 2:
            conditions.append(
                f"correlation_fan_in: {len(updated_domains)} domains updated in window ({sorted(updated_domains)})"
            )
        elif len(updated_domains) == 1:
            # same-domain repeated updates count: the finding was refreshed
            # within the window after an earlier refresh (2 updates in window)
            domain = updated_domains[0]
            f = snap.findings[domain]
            prior_updates = getattr(self, "_update_counts", {}).get((customer_id, domain), 0)
            if prior_updates >= 1:
                conditions.append(
                    f"correlation_repeated: {domain} finding refreshed {prior_updates + 1}x within window"
                )
            self._update_counts = getattr(self, "_update_counts", {})
            self._update_counts[(customer_id, domain)] = prior_updates + 1

        # 2. a single finding crosses high-severity confidence
        high = [
            (domain, f)
            for domain, f in snap.findings.items()
            if f.confidence >= self.high_severity_conf
        ]
        for domain, f in high:
            conditions.append(f"high_severity: {domain} finding confidence {f.confidence:.2f}")

        # 3. KYC hard flag (watchlist hit - deterministic fact)
        compliance = snap.findings.get("compliance")
        if compliance and any(fl.get("type") == "watchlist_hit" for fl in compliance.flags):
            conditions.append("kyc_hard_flag: watchlist hit")

        if not conditions:
            return TriggerDecision(False, "no correlation-worthy condition", [])

        # cooldown: flood protection (never per-event invocation)
        last = self._last_fired.get(customer_id)
        if last is not None and _age_s(last, as_of) < self.cooldown.total_seconds():
            return TriggerDecision(
                False,
                f"cooldown active ({self.cooldown}) after fire at {last.isoformat()}",
                conditions,
            )
        self._last_fired[customer_id] = as_of
        return TriggerDecision(True, "; ".join(conditions), conditions)


def _age_s(since: datetime, now: datetime) -> float:
    if isinstance(since, str):
        since = datetime.fromisoformat(since.replace("Z", "+00:00"))
    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return max(0.0, (now - since).total_seconds())
