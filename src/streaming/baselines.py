"""Historical baseline statistics (Phase 5) - pure, from history seed.

Computes per-customer rolling-window mean/std by sampling weekly windows over
the history seed (docs/reference/03: ``historical_baseline`` carries
``txn_count_30d_mean`` / ``txn_count_30d_std``). Statistical-first detection
(RQ-L1) compares current counts against these baselines via z-scores; the LLM
is never the detector.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from statistics import fmean, stdev

from src.ingestion.schemas.events import EventEnvelope
from src.streaming.features import WindowSpec, in_window


def _tz(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _window_value(events: list[EventEnvelope], spec: WindowSpec, as_of: datetime, customer_id: str | None = None) -> float | None:
    vals: list[float] = []
    for e in events:
        if customer_id is not None and e.customer_id != customer_id:
            continue
        if not in_window(e, spec, as_of):
            continue
        if spec.aggregate == "count":
            vals.append(1.0)
        else:
            raw = e.payload.get(spec.payload_field)  # type: ignore[index]
            if raw is not None:
                vals.append(float(raw))
    if not vals:
        # A window with zero events is a VALID observation for count/sum
        # (skipping it would bias the baseline toward active windows).
        if spec.aggregate in ("count", "sum"):
            return 0.0
        return None  # max over an empty window is not meaningful
    if spec.aggregate == "count":
        return float(len(vals))
    if spec.aggregate == "sum":
        return float(sum(vals))
    return max(vals)


def compute_baselines(
    history_events: list[EventEnvelope],
    specs: tuple[WindowSpec, ...],
    history_end: datetime,
    sample_step_days: float = 7.0,
    customer_id: str | None = None,
) -> dict[str, dict[str, float]]:
    """Per-feature mean/std sampled weekly over the history window.

    Returns ``{feature_name: {"mean": m, "std": s}}`` (std 0 for stable
    features; never None). Pure and deterministic.
    """
    end = _tz(history_end)
    # history span: from earliest event to history_end
    ets = [_tz(e.event_time) for e in history_events]
    if not ets:
        return {}
    start = min(ets)
    span_days = (end - start).total_seconds() / 86400.0
    n_samples = max(1, int(span_days / sample_step_days))
    out: dict[str, dict[str, float]] = {}
    for spec in specs:
        samples: list[float] = []
        for i in range(1, n_samples + 1):  # exclude t=start (empty window at the very beginning)
            as_of = start + timedelta(days=span_days * i / n_samples)
            v = _window_value(history_events, spec, as_of, customer_id)
            if v is not None:
                samples.append(v)
        if not samples:
            continue
        mean = fmean(samples)
        std = stdev(samples) if len(samples) > 1 else 0.0
        out[spec.name] = {"mean": round(mean, 4), "std": round(std, 4)}
    return out


def zscore(current: float, baseline: dict[str, float]) -> float | None:
    """z = (current - mean) / std; None when std is 0 (no variance to detect)."""
    std = baseline.get("std", 0.0)
    if std <= 0:
        return None
    return (current - baseline.get("mean", 0.0)) / std
