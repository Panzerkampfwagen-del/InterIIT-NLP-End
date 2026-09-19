"""Phase 12: fixture-verified metric tests - hand-computed expected values.

These fixtures are verified BEFORE the harness runs on real pipeline output
(DoD: every metric calculator has a hand-computed expected value).
"""
from __future__ import annotations

from evaluation.metrics.calculators import (
    classification_accuracy,
    evidence_grounding,
    expected_calibration_error,
    match_checkpoints,
    no_action_precision_recall,
    precision_recall_f1,
    red_herring_filtering,
    score_checkpoint_pair,
    score_checkpoints,
)


def test_precision_recall_f1_hand_computed() -> None:
    # tp=8, fp=2, fn=2 -> precision=8/10=0.8, recall=8/10=0.8, f1=0.8
    assert precision_recall_f1(8, 2, 2) == {"precision": 0.8, "recall": 0.8, "f1": 0.8}
    # tp=3, fp=1, fn=1 -> p=0.75, r=0.75, f1=0.75
    assert precision_recall_f1(3, 1, 1) == {"precision": 0.75, "recall": 0.75, "f1": 0.75}
    # tp=0 -> all zeros (never divide-by-zero)
    assert precision_recall_f1(0, 5, 5) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    # perfect: tp=4, fp=0, fn=0 -> 1.0
    assert precision_recall_f1(4, 0, 0) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_classification_accuracy_hand_computed() -> None:
    # 3 of 4 correct -> 0.75
    inferred = ["medical_hardship", "churn_risk", "no_signal", "financial_distress_general"]
    expected = ["medical_hardship", "churn_risk", "medical_hardship", "financial_distress_general"]
    assert classification_accuracy(inferred, expected) == 0.75
    # 4 of 4 -> 1.0
    assert classification_accuracy(expected, expected) == 1.0
    # 0 of 4 -> 0.0
    assert classification_accuracy(["no_signal"] * 4, expected) == 0.0


def test_ece_hand_computed() -> None:
    # Bin 1 (0.8-1.0]: conf 0.9 x2, acc 1.0 -> |0.9 - 1.0| = 0.1, share 2/3
    # Bin 2 (0.0-0.2]: conf 0.1 x1, acc 0.0 -> |0.1 - 0.0| = 0.1, share 1/3
    # ECE = 0.1*(2/3) + 0.1*(1/3) = 0.1
    confidences = [0.9, 0.9, 0.1]
    correct = [True, True, False]
    ece = expected_calibration_error(confidences, correct, n_bins=5)
    assert abs(ece - 0.1) < 1e-9
    # perfectly calibrated within bins: conf 1.0 x2 correct, conf 0.0 x1 wrong
    # bin (0.8-1.0]: avg conf 1.0, acc 1.0 -> 0; bin (0.0-0.2]: avg 0.0, acc 0.0 -> 0
    assert expected_calibration_error([1.0, 1.0, 0.0], [True, True, False], n_bins=5) == 0.0
    # all wrong with high confidence: bin avg 0.9, acc 0 -> 0.9
    assert expected_calibration_error([0.9, 0.9], [False, False], n_bins=5) == 0.9


def test_checkpoint_pair_scoring_hand_computed() -> None:
    # perfect match -> score 1.0 (0.4 + 0.25 + 0.25 + 0.1)
    inferred = {"as_of_time": "2026-03-26T00:00:00Z", "inferred_state": "medical_hardship", "confidence_band": "high", "action": "support_intervention", "hitl_status": "escalated"}
    expected = {"as_of_time": "2026-03-26T00:00:00Z", "expected_inferred_state": "medical_hardship", "expected_confidence_band": "high", "expected_action": "support_intervention", "expected_hitl_status": "escalated"}
    result = score_checkpoint_pair(inferred, expected)
    assert result["state_match"] and result["band_match"] and result["action_match"] and result["hitl_match"]
    assert result["score"] == 1.0

    # state correct only -> 0.4
    inferred2 = dict(inferred, confidence_band="low", action="no_action", hitl_status="auto_approved")
    result2 = score_checkpoint_pair(inferred2, expected)
    assert result2["state_match"] is True and result2["band_match"] is False
    assert result2["score"] == 0.4

    # no hitl expected -> hitl_match True (not penalized)
    expected3 = {k: v for k, v in expected.items() if k != "expected_hitl_status"}
    result3 = score_checkpoint_pair(inferred, expected3)
    assert result3["hitl_match"] is True
    assert result3["score"] == 1.0


def test_checkpoint_matching_by_nearest_time() -> None:
    inferred = [
        {"as_of_time": "2026-02-15T00:00:00Z", "inferred_state": "medical_hardship"},
        {"as_of_time": "2026-03-12T00:00:00Z", "inferred_state": "medical_hardship"},
    ]
    expected = [
        {"as_of_time": "2026-02-16T00:00:00Z", "expected_inferred_state": "medical_hardship"},
        {"as_of_time": "2026-03-13T00:00:00Z", "expected_inferred_state": "medical_hardship"},
    ]
    pairs = match_checkpoints(inferred, expected)
    assert len(pairs) == 2
    assert all(e is not None for _, e in pairs)
    # nearest matching: first inferred pairs with first expected
    assert pairs[0][1]["as_of_time"] == "2026-02-16T00:00:00Z"


def test_score_checkpoints_aggregate_hand_computed() -> None:
    inferred = [
        {"as_of_time": "2026-02-15T00:00:00Z", "inferred_state": "medical_hardship", "confidence_band": "low", "action": "no_action", "hitl_status": "auto_approved"},
        {"as_of_time": "2026-03-26T00:00:00Z", "inferred_state": "medical_hardship", "confidence_band": "high", "action": "support_intervention", "hitl_status": "escalated"},
    ]
    expected = [
        {"as_of_time": "2026-02-15T00:00:00Z", "expected_inferred_state": "medical_hardship", "expected_confidence_band": "low", "expected_action": "no_action"},
        {"as_of_time": "2026-03-26T00:00:00Z", "expected_inferred_state": "medical_hardship", "expected_confidence_band": "high", "expected_action": "support_intervention", "expected_hitl_status": "escalated"},
    ]
    result = score_checkpoints(inferred, expected)
    assert result["n_matched"] == 2
    assert result["inference_accuracy"] == 1.0
    assert result["timeliness_band_accuracy"] == 1.0
    assert result["action_accuracy"] == 1.0
    assert result["hitl_routing_accuracy"] == 1.0
    assert result["mean_checkpoint_score"] == 1.0


def test_no_action_precision_recall_separate() -> None:
    """NO_ACTION P/R scored separately (silence inflation would be worse)."""
    # 2 true no_actions, 1 over-reaction (action where no_action expected),
    # 1 missed action (no_action where action expected)
    decisions = [
        {"action": "no_action"},
        {"action": "no_action"},
        {"action": "personalized_offer"},  # fp
        {"action": "no_action"},  # fn (action was expected)
    ]
    expected = [
        {"action": "no_action"},
        {"action": "no_action"},
        {"action": "no_action"},
        {"action": "support_intervention"},
    ]
    result = no_action_precision_recall(decisions, expected)
    # tp=2, fp=1, fn=1 -> p=2/3, r=2/3, f1=2/3
    assert result["precision"] == 0.6667
    assert result["recall"] == 0.6667
    assert result["f1"] == 0.6667


def test_evidence_grounding_hand_computed() -> None:
    known = {"EVT_1", "EVT_2", "EVT_3"}
    # 2 of 3 resolvable -> 0.6667
    assert evidence_grounding(["EVT_1", "EVT_2", "EVT_FAKE"], known) == 0.6667
    # all resolvable -> 1.0
    assert evidence_grounding(["EVT_1"], known) == 1.0
    # no evidence -> 0.0 (unverifiable decision is a red flag)
    assert evidence_grounding([], known) == 0.0


def test_red_herring_filtering_hand_computed() -> None:
    herrings = {"EVT_9"}
    # 1 of 3 evidence is a red herring -> 2/3 clean
    assert red_herring_filtering(["EVT_1", "EVT_2", "EVT_9"], herrings) == 0.6667
    # clean -> 1.0
    assert red_herring_filtering(["EVT_1"], herrings) == 1.0
    # no evidence -> 1.0 (vacuously clean)
    assert red_herring_filtering([], herrings) == 1.0
