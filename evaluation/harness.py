"""Evaluation harness (Phase 12) - RAGAS-style stage decomposition.

Ingests ``inferred-events.json`` (the pipeline's checkpoint output) + the
provided ``ground_truth.json`` and emits a per-stage metrics report (never one
blended number). Scenarios without ground truth are scored only on internal
consistency checks (schema validity, evidence groundedness, guardrail
correctness) rather than forced accuracy.

Stage decomposition (RQ-E1):
- inference stage: inferred_state accuracy + confidence-band timeliness;
- action stage: action accuracy + NO_ACTION precision/recall SEPARATE;
- consistency stage: schema validity + evidence groundedness.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.metrics.calculators import (
    ALLOWED_ACTIONS,
    ALLOWED_BANDS,
    ALLOWED_HITL,
    ALLOWED_INFERRED_STATES,
    evidence_grounding,
    red_herring_filtering,
    score_checkpoints,
)


def validate_checkpoint_schema(cp: dict[str, Any]) -> list[str]:
    """Schema validity of one checkpoint (fixed enums, required fields)."""
    errors: list[str] = []
    if "as_of_time" not in cp:
        errors.append("missing as_of_time")
    if cp.get("inferred_state") not in ALLOWED_INFERRED_STATES:
        errors.append(f"invalid inferred_state: {cp.get('inferred_state')}")
    if cp.get("action") not in ALLOWED_ACTIONS:
        errors.append(f"invalid action: {cp.get('action')}")
    if cp.get("confidence_band") not in ALLOWED_BANDS:
        errors.append(f"invalid confidence_band: {cp.get('confidence_band')}")
    if cp.get("hitl_status") not in ALLOWED_HITL:
        errors.append(f"invalid hitl_status: {cp.get('hitl_status')}")
    return errors


def run_harness(
    inferred_checkpoints: list[dict[str, Any]],
    ground_truth: dict[str, Any] | None = None,
    known_event_ids: set[str] | None = None,
    red_herring_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run the decomposed evaluation over one scenario's checkpoints."""
    known_event_ids = known_event_ids or set()
    red_herring_ids = red_herring_ids or set()

    # consistency stage (always scored - no ground truth needed)
    schema_errors: list[dict[str, Any]] = []
    for cp in inferred_checkpoints:
        errs = validate_checkpoint_schema(cp)
        if errs:
            schema_errors.append({"as_of_time": cp.get("as_of_time"), "errors": errs})

    all_refs = sorted({ref for cp in inferred_checkpoints for ref in (cp.get("evidence_refs") or [])})
    grounding = evidence_grounding(all_refs, known_event_ids)
    herring = red_herring_filtering(all_refs, red_herring_ids)

    report: dict[str, Any] = {
        "stages": {
            "consistency": {
                "schema_valid": len(schema_errors) == 0,
                "schema_errors": schema_errors,
                "evidence_grounding": grounding,
                "red_herring_filtering": herring,
            },
        },
        "has_ground_truth": ground_truth is not None,
    }

    if ground_truth is not None:
        expected_cps = ground_truth.get("checkpoints", [])
        checkpoint_scores = score_checkpoints(inferred_checkpoints, expected_cps)
        report["stages"]["inference"] = {
            "inference_accuracy": checkpoint_scores["inference_accuracy"],
            "timeliness_band_accuracy": checkpoint_scores["timeliness_band_accuracy"],
            "n_matched": checkpoint_scores["n_matched"],
            "n_expected": checkpoint_scores["n_expected"],
        }
        report["stages"]["action"] = {
            "action_accuracy": checkpoint_scores["action_accuracy"],
            "hitl_routing_accuracy": checkpoint_scores["hitl_routing_accuracy"],
            "mean_checkpoint_score": checkpoint_scores["mean_checkpoint_score"],
        }
        report["per_checkpoint"] = checkpoint_scores["per_checkpoint"]
    else:
        report["stages"]["inference"] = {"note": "no ground truth - scored on internal consistency only"}
        report["stages"]["action"] = {"note": "no ground truth - scored on internal consistency only"}

    return report


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def format_report(report: dict[str, Any]) -> str:
    """Human-readable summary (legible to a non-technical reviewer)."""
    lines = ["=" * 60, "EVALUATION REPORT (RAGAS-style stage decomposition)", "=" * 60]
    consistency = report["stages"]["consistency"]
    lines.append(f"Schema valid:          {consistency['schema_valid']}")
    lines.append(f"Evidence grounding:    {consistency['evidence_grounding']:.2f}")
    lines.append(f"Red-herring filtering: {consistency['red_herring_filtering']:.2f}")
    if report.get("has_ground_truth"):
        inf = report["stages"]["inference"]
        act = report["stages"]["action"]
        lines.append("")
        lines.append("-- Inference stage (vs ground truth) --")
        lines.append(f"Inference accuracy:    {inf['inference_accuracy']:.2f} ({inf['n_matched']}/{inf['n_expected']} checkpoints)")
        lines.append(f"Timeliness (band):     {inf['timeliness_band_accuracy']:.2f}")
        lines.append("-- Action stage (vs ground truth) --")
        lines.append(f"Action accuracy:       {act['action_accuracy']:.2f}")
        lines.append(f"HITL routing accuracy: {act['hitl_routing_accuracy']:.2f}")
        lines.append(f"Mean checkpoint score: {act['mean_checkpoint_score']:.2f}")
    else:
        lines.append("No ground truth available: scored on internal consistency only.")
    lines.append("=" * 60)
    return "\n".join(lines)
