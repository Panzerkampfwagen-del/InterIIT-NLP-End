"""Evaluation metric calculators (Phase 12) - pure, fixture-verifiable.

RAGAS-style stage decomposition (RQ-E1): retrieval quality and generation
quality are scored SEPARATELY so a bad decision is diagnosable as a retrieval
failure vs a reasoning failure vs a policy failure - never one blended number.
Every calculator here is a pure function with a hand-computed fixture test
(verified BEFORE running the harness on real pipeline output).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

# --------------------------------------------------------------- base metrics


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision / recall / F1 from confusion counts (hand-computable)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def classification_accuracy(inferred: list[str], expected: list[str]) -> float:
    """Fraction of correctly classified states."""
    if not expected:
        return 0.0
    correct = sum(1 for i, e in zip(inferred, expected) if i == e)
    return round(correct / len(expected), 4)


def expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int = 5) -> float:
    """ECE: sum over bins of |avg_confidence - accuracy| weighted by bin share.

    Hand-computable: perfectly calibrated within a bin contributes 0; a bin
    with avg conf 0.9 and accuracy 1.0 contributes 0.1 * (bin share).
    """
    if not confidences or len(confidences) != len(correct):
        return 1.0
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    total = len(confidences)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        idx = [
            i
            for i, c in enumerate(confidences)
            if (lo <= c < hi) or (b == n_bins - 1 and lo <= c <= hi)
        ]
        if not idx:
            continue
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        ece += abs(avg_conf - acc) * (len(idx) / total)
    return round(ece, 4)


# ------------------------------------------------- dataset checkpoint scoring

ALLOWED_INFERRED_STATES = frozenset(
    {
        "no_signal", "new_child_life_event", "marriage_or_relationship_change",
        "job_change_or_promotion", "job_loss_or_income_disruption", "medical_hardship",
        "financial_distress_general", "relocation", "retirement_transition",
        "wealth_growth_or_windfall", "potential_fraud_or_takeover",
        "elder_vulnerability_or_scam_risk", "churn_risk", "small_business_cashflow_event",
    }
)
ALLOWED_ACTIONS = frozenset(
    {
        "no_action", "proactive_retention_outreach", "relationship_manager_escalation",
        "personalized_offer", "support_intervention", "compliance_fraud_hold",
    }
)
ALLOWED_BANDS = frozenset({"low", "medium", "high"})
ALLOWED_HITL = frozenset({"auto_approved", "escalated", "human_approved", "human_rejected", "human_modified"})


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def match_checkpoints(
    inferred: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Match inferred checkpoints to expected ones by nearest as_of_time.

    Each expected checkpoint matches at most one inferred checkpoint (closest
    in time); unmatched inferred checkpoints pair with None.
    """
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    used: set[int] = set()
    for cp in inferred:
        t = _parse_time(cp.get("as_of_time"))
        best: int | None = None
        best_delta = None
        for j, exp in enumerate(expected):
            if j in used:
                continue
            delta = abs((_parse_time(exp["as_of_time"]) - t).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = j
        if best is not None:
            used.add(best)
            pairs.append((cp, expected[best]))
        else:
            pairs.append((cp, None))
    return pairs


def score_checkpoint_pair(
    inferred_cp: dict[str, Any], expected_cp: dict[str, Any] | None
) -> dict[str, Any]:
    """Score one (inferred, expected) checkpoint pair on the 3 dataset criteria:
    accuracy of inference, timeliness (confidence band at the right moment),
    appropriate action (+ hitl routing)."""
    if expected_cp is None:
        return {
            "as_of_time": inferred_cp.get("as_of_time"),
            "matched": False,
            "state_match": False,
            "band_match": False,
            "action_match": False,
            "hitl_match": False,
            "score": 0.0,
        }
    state_ok = inferred_cp.get("inferred_state") == expected_cp.get("expected_inferred_state")
    band_ok = inferred_cp.get("confidence_band") == expected_cp.get("expected_confidence_band")
    action_ok = inferred_cp.get("action") == expected_cp.get("expected_action")
    hitl_expected = expected_cp.get("expected_hitl_status")
    hitl_ok = True if hitl_expected is None else inferred_cp.get("hitl_status") == hitl_expected
    score = 0.4 * float(state_ok) + 0.25 * float(band_ok) + 0.25 * float(action_ok) + 0.1 * float(hitl_ok)
    return {
        "as_of_time": inferred_cp.get("as_of_time"),
        "matched": True,
        "state_match": state_ok,
        "band_match": band_ok,
        "action_match": action_ok,
        "hitl_match": hitl_ok,
        "score": round(score, 4),
    }


def score_checkpoints(
    inferred: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate checkpoint scoring: per-stage scores, not one blended number."""
    pairs = match_checkpoints(inferred, expected)
    per_cp = [score_checkpoint_pair(i, e) for i, e in pairs]
    matched = [p for p in per_cp if p["matched"]]
    if matched:
        state_acc = sum(1 for p in matched if p["state_match"]) / len(matched)
        band_acc = sum(1 for p in matched if p["band_match"]) / len(matched)
        action_acc = sum(1 for p in matched if p["action_match"]) / len(matched)
        hitl_acc = sum(1 for p in matched if p["hitl_match"]) / len(matched)
        mean_score = sum(p["score"] for p in matched) / len(matched)
    else:
        state_acc = band_acc = action_acc = hitl_acc = mean_score = 0.0
    return {
        "n_inferred": len(inferred),
        "n_expected": len(expected),
        "n_matched": len(matched),
        "inference_accuracy": round(state_acc, 4),
        "timeliness_band_accuracy": round(band_acc, 4),
        "action_accuracy": round(action_acc, 4),
        "hitl_routing_accuracy": round(hitl_acc, 4),
        "mean_checkpoint_score": round(mean_score, 4),
        "per_checkpoint": per_cp,
    }


# ------------------------------------------------------ action/NO_ACTION P/R


def no_action_precision_recall(
    decisions: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> dict[str, Any]:
    """NO_ACTION precision/recall scored SEPARATELY from action accuracy
    (silence inflation would be worse than no metric at all).

    decisions/expected: [{"as_of_time", "action"}] aligned by index.
    """
    tp = fp = fn = 0
    for d, e in zip(decisions, expected):
        d_na = d.get("action") == "no_action"
        e_na = e.get("action") == "no_action"
        if d_na and e_na:
            tp += 1
        elif d_na and not e_na:
            fn += 1  # missed an action (silence where action was expected)
        elif not d_na and e_na:
            fp += 1  # over-reaction (action where no_action was expected)
    return precision_recall_f1(tp, fp, fn)


# ------------------------------------------------------- evidence grounding


def evidence_grounding(evidence_refs: list[str], known_event_ids: set[str]) -> float:
    """Fraction of evidence refs resolvable to real event ids (groundedness).

    Zero groundedness on an inference is a red flag (unverifiable decision).
    """
    if not evidence_refs:
        return 0.0
    grounded = sum(1 for r in evidence_refs if r in known_event_ids)
    return round(grounded / len(evidence_refs), 4)


def red_herring_filtering(evidence_refs: list[str], red_herring_ids: set[str]) -> float:
    """Fraction of evidence that is NOT a known red herring (1.0 = clean)."""
    if not evidence_refs:
        return 1.0
    clean = sum(1 for r in evidence_refs if r not in red_herring_ids)
    return round(clean / len(evidence_refs), 4)
