"""Phase 13: end-to-end integration - every scenario through the full pipeline.

DoD: every synthetic scenario produces a valid, schema-conformant decision
object end-to-end with zero unhandled exceptions (N scenarios run, N
succeeded), and the Phase 12 harness runs cleanly over the real output.
"""
from __future__ import annotations

import json

import pytest

from src.config import get_settings


def _infra() -> bool:
    import socket

    s = get_settings()
    host, port = s.pg_dsn.split("//")[1].split("@")[1].split("/")[0].split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2.0):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _infra(), reason="Postgres not running")


@pytest.fixture(scope="module")
def pipeline():
    from src.agents.llm import FakeLLMClient
    from src.orchestration.graph import Customer360Pipeline
    from src.state.board import CustomerStateBoard

    b = CustomerStateBoard()
    b.init_schema()
    with b._connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE customer_state, state_episodic, state_conflicts, watermarks, identity_records, identity_graph_edges, retrieval_chunks, hitl_approvals, decision_log RESTART IDENTITY"
            )
    p = Customer360Pipeline(settings=get_settings(), llm=FakeLLMClient(fail=True), board=b)
    yield p
    b.close()


SCENARIOS = ["scenario_01", "scenario_02", "scenario_03", "conflicting_signals", "ambiguous"]

# Each scenario runs EXACTLY ONCE per module (the pipeline's in-memory dedup
# correctly suppresses duplicate re-runs); results are shared across tests.
_RESULTS: dict = {}


def _run_all(pipeline) -> dict:
    from src.ingestion.dataset_loader import load_scenario

    if not _RESULTS:
        for name in SCENARIOS:
            scenario = load_scenario(name)
            gt_times = None
            if scenario.ground_truth and scenario.ground_truth.get("checkpoints"):
                from datetime import datetime

                gt_times = [
                    datetime.fromisoformat(cp["as_of_time"].replace("Z", "+00:00"))
                    for cp in scenario.ground_truth["checkpoints"]
                ]
            _RESULTS[name] = (scenario, pipeline.run_scenario(scenario, checkpoint_times=gt_times))
    return _RESULTS


def test_all_scenarios_run_end_to_end(pipeline) -> None:
    """Every scenario produces schema-conformant checkpoints with zero unhandled exceptions."""
    from evaluation.metrics.calculators import (
        ALLOWED_ACTIONS,
        ALLOWED_BANDS,
        ALLOWED_HITL,
        ALLOWED_INFERRED_STATES,
    )

    results = _run_all(pipeline)
    for name, (_scenario, result) in results.items():
        assert result.events_processed > 0, f"{name}: no events processed"
        assert result.events_failed == 0, f"{name}: {result.events_failed} events failed"
        # every emitted checkpoint is schema-conformant (dataset enums)
        for cp in result.checkpoints:
            assert cp["inferred_state"] in ALLOWED_INFERRED_STATES, f"{name}: invalid state {cp['inferred_state']}"
            assert cp["action"] in ALLOWED_ACTIONS, f"{name}: invalid action {cp['action']}"
            assert cp["confidence_band"] in ALLOWED_BANDS, f"{name}: invalid band {cp['confidence_band']}"
            assert cp["hitl_status"] in ALLOWED_HITL, f"{name}: invalid hitl {cp['hitl_status']}"
            assert cp["notes"], f"{name}: checkpoint without notes (silence)"
        assert len(result.checkpoints) >= 1, f"{name}: zero checkpoints (silence)"
        assert result.final_decision is not None, f"{name}: no decisions made"
        assert result.final_decision["decision_id"].startswith("dec_")


def test_scenario_01_checkpoints_match_ground_truth_schedule(pipeline) -> None:
    """The medical_hardship scenario: one checkpoint per ground-truth moment."""
    results = _run_all(pipeline)
    scenario, result = results["scenario_01"]
    gt_times = scenario.ground_truth["checkpoints"] if scenario.ground_truth else []
    assert result.events_processed > 0
    assert len(result.checkpoints) == len(gt_times)


def test_harness_runs_over_real_pipeline_output(pipeline) -> None:
    """The Phase 12 harness runs cleanly over this phase's real output."""
    from evaluation.harness import run_harness

    _scenario, result = _run_all(pipeline)["scenario_01"]
    gt = _scenario.ground_truth or {}
    report = run_harness(
        result.checkpoints,
        gt,
        known_event_ids=set(gt.get("signal_events", [])) | set(gt.get("red_herring_events", [])),
        red_herring_ids=set(gt.get("red_herring_events", [])),
    )
    assert report["stages"]["consistency"]["schema_valid"] is True
    assert report["has_ground_truth"] is True


def test_demo_script_writes_inferred_events_file(pipeline) -> None:
    """scripts/run_demo.py writes the dataset-schema checkpoints file."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "evaluation" / "inferred_events"
    venv_python = root / ".venv" / "bin" / "python"
    python_bin = str(venv_python) if venv_python.exists() else sys.executable
    proc = subprocess.run(
        [python_bin, str(root / "scripts" / "run_demo.py"), "--scenario", "scenario_01"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"demo failed: {proc.stderr[-2000:]}"
    out_file = out_dir / "scenario_01_checkpoints.json"
    assert out_file.is_file(), "demo did not write the inferred-events file"
    checkpoints = json.loads(out_file.read_text())
    assert len(checkpoints) >= 1
    for cp in checkpoints:
        assert "as_of_time" in cp and "inferred_state" in cp and "action" in cp
