#!/usr/bin/env python3
"""Evaluation CLI (Phase 12): run the harness over inferred-events output vs
the provided ground truth; emits JSON + a human-readable summary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.harness import format_report, load_json, run_harness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation harness")
    parser.add_argument("--inferred", required=True, help="path to inferred-events checkpoints JSON")
    parser.add_argument("--ground-truth", default=None, help="path to ground_truth.json (optional)")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args()

    inferred = load_json(Path(args.inferred))
    if isinstance(inferred, dict):  # allow wrapped output
        inferred = inferred.get("checkpoints", [])
    ground_truth = load_json(Path(args.ground_truth)) if args.ground_truth else None

    gt = ground_truth or {}
    report = run_harness(
        inferred,
        ground_truth,
        known_event_ids=set(gt.get("signal_events", [])) | set(gt.get("red_herring_events", [])),
        red_herring_ids=set(gt.get("red_herring_events", [])),
    )

    print(format_report(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
