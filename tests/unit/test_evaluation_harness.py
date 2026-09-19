"""Phase 12: harness end-to-end over real ground truth + CLI."""
from __future__ import annotations

import json

from evaluation.harness import format_report, run_harness


def test_harness_runs_over_dataset_ground_truth() -> None:
    """Harness runs end-to-end over the provided dataset's ground truth with
    synthetic inferred checkpoints and produces per-stage scores."""
    from pathlib import Path

    gt_path = Path(__file__).resolve().parents[2] / "evaluation" / "dataset" / "scenario_01" / "ground_truth.json"
    gt = json.loads(gt_path.read_text())

    # synthetic inferred checkpoints aligned with the ground truth checkpoints
    inferred = []
    for cp in gt["checkpoints"]:
        inferred.append(
            {
                "as_of_time": cp["as_of_time"],
                "inferred_state": cp["expected_inferred_state"],
                "confidence_band": cp["expected_confidence_band"],
                "action": cp["expected_action"],
                "hitl_status": cp.get("expected_hitl_status", "auto_approved"),
                "evidence_refs": gt.get("signal_events", [])[:3],
            }
        )

    report = run_harness(
        inferred,
        gt,
        known_event_ids=set(gt.get("signal_events", [])) | set(gt.get("red_herring_events", [])),
        red_herring_ids=set(gt.get("red_herring_events", [])),
    )
    assert report["has_ground_truth"] is True
    # per-stage scores present (not one blended number)
    assert "inference" in report["stages"] and "action" in report["stages"] and "consistency" in report["stages"]
    assert report["stages"]["inference"]["inference_accuracy"] == 1.0
    assert report["stages"]["action"]["action_accuracy"] == 1.0
    assert report["stages"]["consistency"]["schema_valid"] is True
    assert report["stages"]["consistency"]["evidence_grounding"] == 1.0
    # human-readable summary renders
    text = format_report(report)
    assert "EVALUATION REPORT" in text and "Inference stage" in text


def test_harness_without_ground_truth_scores_consistency_only() -> None:
    """Scenarios without ground truth: internal consistency, no forced accuracy."""
    inferred = [
        {"as_of_time": "2026-05-01T00:00:00Z", "inferred_state": "no_signal", "confidence_band": "low", "action": "no_action", "hitl_status": "auto_approved", "evidence_refs": []},
    ]
    report = run_harness(inferred, None)
    assert report["has_ground_truth"] is False
    assert "note" in report["stages"]["inference"]
    assert report["stages"]["consistency"]["schema_valid"] is True


def test_harness_flags_invalid_checkpoint_schema() -> None:
    """Silently inflated scores are worse than no metric: invalid enums flagged."""
    inferred = [
        {"as_of_time": "2026-05-01T00:00:00Z", "inferred_state": "totally_custom_label", "confidence_band": "extreme", "action": "do_something", "hitl_status": "maybe"},
    ]
    report = run_harness(inferred, None)
    assert report["stages"]["consistency"]["schema_valid"] is False
    assert len(report["stages"]["consistency"]["schema_errors"]) == 1
    errors = report["stages"]["consistency"]["schema_errors"][0]["errors"]
    assert any("inferred_state" in e for e in errors)
    assert any("action" in e for e in errors)
