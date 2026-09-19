#!/usr/bin/env python3
"""Demo entrypoint (Phase 13): run one named scenario end-to-end through the
full pipeline (replay simulator path -> graph) and print/export the final
decision object, its trace_id, and a reference to its full explanation.

Also emits the dataset-schema inferred-events checkpoints file - the terminal
artifact evaluators score against (see evaluation/dataset/README schema).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.ingestion.dataset_loader import load_scenario  # noqa: E402
from src.orchestration.graph import Customer360Pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one scenario end-to-end")
    parser.add_argument("--scenario", default="scenario_01", help="scenario directory name")
    parser.add_argument("--out-dir", default=None, help="where to write the inferred-events checkpoints")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    gt_checkpoint_times = None
    if scenario.ground_truth and scenario.ground_truth.get("checkpoints"):
        from datetime import datetime

        gt_checkpoint_times = [
            datetime.fromisoformat(cp["as_of_time"].replace("Z", "+00:00"))
            for cp in scenario.ground_truth["checkpoints"]
        ]

    pipeline = Customer360Pipeline(settings=get_settings(), llm=None)  # deterministic demo (no live LLM)
    result = pipeline.run_scenario(scenario, checkpoint_times=gt_checkpoint_times)

    # terminal artifact: inferred-events checkpoints in the dataset schema
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "evaluation" / "inferred_events"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.scenario}_checkpoints.json"
    out_path.write_text(json.dumps(result.checkpoints, indent=2, default=str))

    print("=" * 60)
    print(f"SCENARIO {args.scenario} - end-to-end run complete")
    print("=" * 60)
    print(f"Customer:        {result.customer_id}")
    print(f"Events processed: {result.events_processed} (failed: {result.events_failed})")
    print(f"Decisions made:  {len(result.decisions)}")
    print(f"Checkpoints:     {len(result.checkpoints)} (written to {out_path})")
    if result.final_decision:
        fd = result.final_decision
        print("")
        print("-- FINAL DECISION --")
        print(f"  action:     {fd['final_action']}")
        print(f"  confidence: {fd['action_confidence']:.2f} ({fd['confidence_band']})")
        print(f"  status:     {fd['status']}")
        print(f"  reasoning:  {fd['reasoning_summary']}")
        print("")
        print("-- EXPLANATION --")
        print(f"  trace_id:   {result.trace_id}")
        print(f"  decision_id: {fd['decision_id']}")
        print("  full explanation: .venv/bin/python -c \"from src.explainability import Explainer; print(Explainer().explain_json('" + fd["decision_id"] + "'))\"")
    print("")
    print("-- CHECKPOINTS (inferred-events output) --")
    for cp in result.checkpoints:
        print(f"  {cp['as_of_time']}: {cp['inferred_state']} ({cp['confidence_band']}) -> {cp['action']} [{cp['hitl_status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
