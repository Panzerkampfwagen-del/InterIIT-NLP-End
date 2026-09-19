"""Synthetic scenario generator (Phase 1) - SIMULATION/TEST data only.

The provided evaluation dataset (ADR-06) covers medical_hardship,
new_child_life_event, and churn_risk. This generator supplements the archetypes
the dataset does not cover, per the Phase 1 prompt's archetype list:

- conflicting_signals   (usage says healthy, transactions say distress)
- ambiguous_signals     (weak, mixed evidence - correct outcome is no_action)

All generated scenarios are clearly marked ``generated: true`` in entities.json
and live under ``evaluation/scenarios/generated_*`` so they can never be
confused with the provided dataset (which stays under ``evaluation/dataset``).
Deterministic given ``seed``.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.ingestion.schemas.events import EventEnvelope


def _ev(
    event_id: str,
    event_time: datetime,
    customer_id: str,
    source_system: str,
    event_type: str,
    payload: dict,
    account_id: str | None = None,
) -> EventEnvelope:
    """Build one synthetic event (ingestion_time == event_time by default)."""
    return EventEnvelope.model_validate(
        {
            "event_id": event_id,
            "event_time": event_time,
            "ingestion_time": event_time,
            "customer_id": customer_id,
            "account_id": account_id,
            "source_system": source_system,
            "event_type": event_type,
            "schema_version": "1.0",
            "payload": payload,
        }
    )


def _entities(scenario_id: str, customer_id: str, name: str, tier: str) -> dict:
    return {
        "scenario_id": scenario_id,
        "generated": True,
        "customer_id": customer_id,
        "household_id": None,
        "profile": {
            "name": name,
            "age": 45,
            "occupation": "engineer",
            "customer_value_tier": tier,
            "tenure_months": 60,
            "marital_status": "married",
            "dependents": 1,
        },
        "accounts": [
            {"account_id": "ACC_CHK_001", "type": "checking", "opened_date": "2021-01-15"},
            {"account_id": "ACC_SAV_001", "type": "savings", "opened_date": "2021-01-15"},
            {"account_id": "ACC_CC_001", "type": "credit_card", "opened_date": "2021-06-01"},
        ],
    }


def _replay_config(scenario_id: str, start: datetime, end: datetime) -> dict:
    return {
        "scenario_id": scenario_id,
        "simulated_start": start.isoformat().replace("+00:00", "Z"),
        "simulated_end": end.isoformat().replace("+00:00", "Z"),
        "replay_speed_seconds_per_simulated_day": 2,
    }


def _write_scenario(out_dir: Path, scenario_id: str, entities: dict, events: list[EventEnvelope], replay_config: dict, ground_truth: dict) -> Path:
    scenario_dir = out_dir / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "entities.json").write_text(json.dumps(entities, indent=2))
    with (scenario_dir / "live_stream.jsonl").open("w") as f:
        for e in sorted(events, key=lambda x: x.event_time):
            f.write(e.model_dump_json() + "\n")
    # empty history seed: generator scenarios are self-contained
    (scenario_dir / "history_seed.jsonl").write_text("")
    (scenario_dir / "replay_config.json").write_text(json.dumps(replay_config, indent=2))
    (scenario_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))
    (scenario_dir / "README.md").write_text(
        f"# {scenario_id}\n\nGENERATED synthetic scenario (src/ingestion/generator.py) - not part of the provided dataset.\n"
    )
    return scenario_dir


# --- archetype builders -----------------------------------------------------


def build_conflicting_signals(scenario_id: str = "generated_conflicting_signals") -> tuple[dict, list[EventEnvelope], dict, dict]:
    """Usage telemetry looks healthy while transactions scream distress."""
    cid = "CUST_90001"
    start = datetime(2026, 5, 1, tzinfo=UTC)
    events: list[EventEnvelope] = []
    n = 0

    def nxt(t: datetime) -> datetime:
        nonlocal n
        n += 1
        return t

    # 20 days of healthy-looking app engagement (high login frequency)
    for day in range(20):
        t = start + timedelta(days=day, hours=9)
        events.append(_ev(f"GEVT_{n:06d}", nxt(t), cid, "web_app_events", "login", {"feature_or_page": "dashboard", "device_type": "mobile"}, "ACC_CHK_001"))

    # salary credits then a sharp pattern shift: repeated declines + large withdrawals
    for day in range(10):
        t = start + timedelta(days=day, hours=8)
        events.append(_ev(f"GEVT_{n:06d}", nxt(t), cid, "core_banking_ledger", "deposit", {"amount": 4200, "balance_after": 30000 + day * 100, "transaction_type": "salary_credit"}, "ACC_CHK_001"))

    # distress window: 4 declines in 24h + big withdrawals
    distress_start = start + timedelta(days=21)
    for hour in (10, 12, 14, 16):
        events.append(_ev(f"GEVT_{n:06d}", nxt(distress_start + timedelta(hours=hour)), cid, "card_payments", "decline", {"merchant_name": "Grocer", "mcc_category": "grocery", "amount": 95, "currency": "USD", "is_international": False, "card_present": True, "decline_reason": "insufficient_funds"}, "ACC_CC_001"))
    for day in range(4):
        t = distress_start + timedelta(days=day, hours=11)
        events.append(_ev(f"GEVT_{n:06d}", nxt(t), cid, "core_banking_ledger", "withdrawal", {"amount": 3500, "balance_after": 20000 - day * 3000, "transaction_type": "atm_withdrawal"}, "ACC_CHK_001"))

    gt = {
        "scenario_id": scenario_id,
        "generated": True,
        "true_narrative": "Healthy engagement masks a transaction-pattern breakdown: repeated declines and accelerating withdrawals after day 21 indicate financial distress.",
        "signal_events": [e.event_id for e in events if e.event_type in ("decline", "withdrawal")],
        "red_herring_events": [],
        "checkpoints": [
            {"as_of_time": (start + timedelta(days=20)).isoformat().replace("+00:00", "Z"), "expected_inferred_state": "financial_distress_general", "expected_confidence_band": "low", "expected_action": "no_action", "notes": "Distress window has not started; healthy engagement only."},
            {"as_of_time": (distress_start + timedelta(days=4)).isoformat().replace("+00:00", "Z"), "expected_inferred_state": "financial_distress_general", "expected_confidence_band": "medium", "expected_action": "support_intervention", "expected_hitl_status": "auto_approved", "notes": "4 declines in 24h plus accelerating withdrawals - support intervention without escalation."},
        ],
    }
    return _entities(scenario_id, cid, "Conflict Carla", "mid"), events, _replay_config(scenario_id, start, distress_start + timedelta(days=5)), gt


def build_ambiguous(scenario_id: str = "generated_ambiguous") -> tuple[dict, list[EventEnvelope], dict, dict]:
    """Weak mixed evidence; the correct outcome is an explicit no_action."""
    cid = "CUST_90002"
    start = datetime(2026, 5, 1, tzinfo=UTC)
    events: list[EventEnvelope] = []
    n = 0
    for day in range(25):
        t = start + timedelta(days=day, hours=8)
        events.append(_ev(f"GEVT_{n:06d}", t + timedelta(minutes=n), cid, "core_banking_ledger", "deposit", {"amount": 3800, "balance_after": 25000, "transaction_type": "salary_credit"}, "ACC_CHK_001"))
        events.append(_ev(f"GEVT_{n:06d}", t + timedelta(hours=4), cid, "card_payments", "purchase", {"merchant_name": "Pharmacy", "mcc_category": "health", "amount": 40, "currency": "USD", "is_international": False, "card_present": True}, "ACC_CC_001"))
    # one moderate isolated event: nothing corroborating
    events.append(_ev(f"GEVT_{n:06d}", start + timedelta(days=12, hours=13), cid, "core_banking_ledger", "withdrawal", {"amount": 1500, "balance_after": 23500, "transaction_type": "atm_withdrawal"}, "ACC_CHK_001"))

    gt = {
        "scenario_id": scenario_id,
        "generated": True,
        "true_narrative": "Routine pharmacy spend and one moderate withdrawal - no corroborating pattern. Correct behaviour is no_action at every checkpoint.",
        "signal_events": [],
        "red_herring_events": [e.event_id for e in events if e.event_type == "withdrawal"],
        "checkpoints": [
            {"as_of_time": (start + timedelta(days=24)).isoformat().replace("+00:00", "Z"), "expected_inferred_state": "no_signal", "expected_confidence_band": "low", "expected_action": "no_action", "notes": "No corroborated pattern; explicit no_action."},
        ],
    }
    return _entities(scenario_id, cid, "Ambiguous Andy", "mid"), events, _replay_config(scenario_id, start, start + timedelta(days=25)), gt


ARCHETYPES = {
    "conflicting_signals": build_conflicting_signals,
    "ambiguous": build_ambiguous,
}


def generate_all(out_dir: Path | None = None) -> list[Path]:
    """Generate every supplementary archetype under evaluation/scenarios/."""
    out = out_dir or Path(__file__).resolve().parents[2] / "evaluation" / "scenarios"
    written: list[Path] = []
    for name, builder in ARCHETYPES.items():
        entities, events, replay_config, gt = builder()
        written.append(_write_scenario(out, name, entities, events, replay_config, gt))
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in generate_all():
        print("generated:", p)
