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
from src.logging_setup import get_logger  # noqa: E402
from src.orchestration.graph import Customer360Pipeline  # noqa: E402

log = get_logger("scripts.run_demo")


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

    log.info(
        "demo_complete",
        extra={
            "scenario": args.scenario,
            "customer_id": result.customer_id,
            "events_processed": result.events_processed,
            "events_failed": result.events_failed,
            "decisions": len(result.decisions),
            "checkpoints": len(result.checkpoints),
            "checkpoints_path": str(out_path),
        },
    )
    if result.final_decision:
        fd = result.final_decision
        log.info(
            "demo_final_decision",
            extra={
                "decision_id": fd["decision_id"],
                "trace_id": result.trace_id,
                "final_action": fd["final_action"],
                "action_confidence": fd["action_confidence"],
                "confidence_band": fd["confidence_band"],
                "status": fd["status"],
                "reasoning_summary": fd["reasoning_summary"],
            },
        )
    log.info(
        "demo_checkpoints",
        extra={
            "count": len(result.checkpoints),
            "items": [
                {
                    "as_of_time": cp["as_of_time"],
                    "inferred_state": cp["inferred_state"],
                    "confidence_band": cp["confidence_band"],
                    "action": cp["action"],
                    "hitl_status": cp["hitl_status"],
                }
                for cp in result.checkpoints
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
