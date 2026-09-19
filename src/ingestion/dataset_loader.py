"""Scenario loader for the provided Customer 360 evaluation dataset.

Loads ``entities.json``, ``history_seed.jsonl``, ``live_stream.jsonl``,
``replay_config.json``, ``instructions.md`` and (where available)
``ground_truth.json`` for one scenario directory under ``evaluation/dataset``.
All events are validated through the Phase 1 schema on load.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.schemas.events import EventEnvelope


class ScenarioNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class Scenario:
    """One loaded scenario: reference data, seeded history, live stream, config."""

    scenario_id: str
    entities: dict
    history_seed: tuple[EventEnvelope, ...]
    live_stream: tuple[EventEnvelope, ...]
    replay_config: dict
    ground_truth: dict | None = None
    instructions: str | None = None
    source_dir: Path | None = None
    generated: bool = False

    @property
    def customer_id(self) -> str:
        return self.entities["customer_id"]

    @property
    def profile(self) -> dict:
        return self.entities.get("profile", {})

    @property
    def accounts(self) -> list[dict]:
        return self.entities.get("accounts", [])

    @property
    def signal_event_ids(self) -> set[str]:
        if not self.ground_truth:
            return set()
        return set(self.ground_truth.get("signal_events", []))

    @property
    def red_herring_event_ids(self) -> set[str]:
        if not self.ground_truth:
            return set()
        return set(self.ground_truth.get("red_herring_events", []))


def _read_jsonl(path: Path) -> tuple[EventEnvelope, ...]:
    events: list[EventEnvelope] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(EventEnvelope.model_validate(json.loads(line)))
            except Exception as exc:  # surface the offending line clearly
                raise ValueError(f"{path}:{line_no}: invalid event: {exc}") from exc
    return tuple(events)


def load_scenario(name: str, dataset_root: Path | None = None) -> Scenario:
    """Load a scenario by directory name (e.g. ``scenario_01``)."""
    root = dataset_root or Path(__file__).resolve().parents[2] / "evaluation"
    for base in (root / "dataset", root / "scenarios"):
        scenario_dir = base / name
        if scenario_dir.is_dir():
            break
    else:
        raise ScenarioNotFoundError(
            f"scenario {name!r} not found under {root}/dataset or {root}/scenarios"
        )

    entities = json.loads((scenario_dir / "entities.json").read_text())
    history_seed = _read_jsonl(scenario_dir / "history_seed.jsonl")
    live_stream = _read_jsonl(scenario_dir / "live_stream.jsonl")
    replay_config = json.loads((scenario_dir / "replay_config.json").read_text())

    gt_path = scenario_dir / "ground_truth.json"
    ground_truth = json.loads(gt_path.read_text()) if gt_path.is_file() else None
    instr_path = scenario_dir / "instructions.md"
    instructions = instr_path.read_text() if instr_path.is_file() else None

    return Scenario(
        scenario_id=entities.get("scenario_id", name),
        entities=entities,
        history_seed=history_seed,
        live_stream=live_stream,
        replay_config=replay_config,
        ground_truth=ground_truth,
        instructions=instructions,
        source_dir=scenario_dir,
        generated=bool(entities.get("generated", False)),
    )


def available_scenarios(dataset_root: Path | None = None) -> list[str]:
    """List loadable scenario names, provided dataset first then generated."""
    root = dataset_root or Path(__file__).resolve().parents[2] / "evaluation"
    names: list[str] = []
    for base in (root / "dataset", root / "scenarios"):
        if base.is_dir():
            names.extend(sorted(p.name for p in base.iterdir() if p.is_dir()))
    return names
