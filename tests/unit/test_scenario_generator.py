"""Phase 1: synthetic generator scenario round-trip validation."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.ingestion.dataset_loader import load_scenario
from src.ingestion.generator import ARCHETYPES, generate_all


def test_all_archetypes_generate_and_reload() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        written = generate_all(out / "scenarios")
        assert len(written) == len(ARCHETYPES)
        for name in ARCHETYPES:
            scenario = load_scenario(name, dataset_root=out)
            assert scenario.ground_truth is not None
            assert scenario.generated is True
            assert len(scenario.live_stream) > 0
            # every generated event validates (loader already validated on load)
            assert scenario.customer_id.startswith("CUST_9")


def test_generated_scenarios_have_checkpoints() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        generate_all(out / "scenarios")
        for name in ARCHETYPES:
            scenario = load_scenario(name, dataset_root=out)
            cps = scenario.ground_truth["checkpoints"]
            assert len(cps) >= 1
            for cp in cps:
                assert "expected_inferred_state" in cp and "expected_action" in cp
