"""Phase 5: signal agent detection tests (no DB, no live LLM).

The KYC zero-LLM test asserts the design contract: the KYC agent never calls
the LLM. The fallback test asserts the strict fallback contract: LLM failure
keeps the deterministic summary.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agents.kyc_agent import KycAgent
from src.agents.llm import FakeLLMClient
from src.agents.support_agent import SupportAgent
from src.agents.transaction_agent import TransactionAgent
from src.agents.usage_agent import UsageAgent
from src.ingestion.schemas.events import EventEnvelope
from src.state.models import StateSnapshot

T0 = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)


def _ev(i, et, source, event_type, payload=None, cid="C1"):
    return EventEnvelope.model_validate(
        {
            "event_id": f"EVT_{i:06d}",
            "event_time": et,
            "ingestion_time": et,
            "customer_id": cid,
            "account_id": "ACC_CHK_001",
            "source_system": source,
            "event_type": event_type,
            "schema_version": "1.0",
            "payload": payload or {},
        }
    )


def _state(recent_refs=None, baseline=None) -> StateSnapshot:
    return StateSnapshot(
        customer_id="C1",
        recent_events=recent_refs or [],
        historical_baseline=baseline or {},
    )


# --- transaction agent -------------------------------------------------------

def test_decline_burst_detected() -> None:
    agent = TransactionAgent(board=None, llm=None)
    refs = [
        {"event_id": f"EVT_{i:06d}", "source_system": "card_payments", "event_type": "decline", "event_time": (T0 - timedelta(minutes=30 * i)).isoformat()}
        for i in range(1, 4)
    ]
    event = _ev(10, T0, "card_payments", "decline", {"amount": 95, "decline_reason": "insufficient_funds"})
    flags = agent.detect(event, _state(recent_refs=refs))
    types = [f["type"] for f in flags]
    assert "decline_burst" in types
    burst = next(f for f in flags if f["type"] == "decline_burst")
    assert len(burst["evidence"]) == 4  # 3 refs + triggering event


def test_large_amount_flagged() -> None:
    agent = TransactionAgent(board=None, llm=None)
    event = _ev(1, T0, "ach_wire", "outbound_transfer", {"amount": 12000})
    flags = agent.detect(event, _state())
    assert any(f["type"] == "large_amount" for f in flags)


def test_normal_transactions_no_flags() -> None:
    agent = TransactionAgent(board=None, llm=None)
    event = _ev(1, T0, "card_payments", "purchase", {"amount": 20})
    assert agent.detect(event, _state()) == []


def test_accelerated_withdrawals_vs_baseline() -> None:
    agent = TransactionAgent(board=None, llm=None)
    baseline = {"large_withdrawal_count_7d": {"mean": 0.2, "std": 0.1}}
    refs = [
        {"event_id": f"EVT_{i:06d}", "source_system": "core_banking_ledger", "event_type": "withdrawal", "event_time": (T0 - timedelta(days=i)).isoformat()}
        for i in range(1, 3)
    ]
    event = _ev(9, T0, "core_banking_ledger", "withdrawal", {"amount": 3500})
    flags = agent.detect(event, _state(recent_refs=refs, baseline=baseline))
    assert any(f["type"] == "accelerated_withdrawals" for f in flags)


# --- usage agent --------------------------------------------------------------

def test_engagement_drop_detected() -> None:
    agent = UsageAgent(board=None, llm=None)
    baseline = {"login_count_7d": {"mean": 20.0, "std": 3.0}}
    event = _ev(1, T0, "web_app_events", "login", {"device_type": "mobile"})
    flags = agent.detect(event, _state(recent_refs=[], baseline=baseline))
    assert any(f["type"] == "engagement_drop" for f in flags)


def test_search_spike_detected() -> None:
    agent = UsageAgent(board=None, llm=None)
    refs = [
        {"event_id": f"EVT_{i:06d}", "source_system": "web_app_events", "event_type": "search_query", "event_time": (T0 - timedelta(days=i)).isoformat()}
        for i in range(1, 3)
    ]
    event = _ev(5, T0, "web_app_events", "search_query", {"search_text": "medical hardship plan"})
    flags = agent.detect(event, _state(recent_refs=refs))
    assert any(f["type"] == "search_spike" for f in flags)


def test_usage_agent_ignores_other_sources() -> None:
    agent = UsageAgent(board=None, llm=None)
    event = _ev(1, T0, "card_payments", "purchase", {"amount": 20})
    assert agent.detect(event, _state()) == []


# --- support agent ------------------------------------------------------------

def test_negative_sentiment_lexicon() -> None:
    agent = SupportAgent(board=None, llm=None)
    event = _ev(1, T0, "support_logs", "ticket_created", {
        "channel": "chat", "category": "payment_arrangements",
        "raw_text": "I've been in the hospital and my income dropped. Can I set up a payment plan?",
        "resolution_status": "open",
    })
    flags = agent.detect(event, _state())
    assert any(f["type"] == "negative_sentiment" for f in flags)


def test_repeated_contact_detected() -> None:
    agent = SupportAgent(board=None, llm=None)
    refs = [
        {"event_id": "EVT_000001", "source_system": "support_logs", "event_type": "ticket_created", "event_time": (T0 - timedelta(days=5)).isoformat()},
    ]
    event = _ev(2, T0, "support_logs", "ticket_created", {
        "channel": "call", "category": "complaint",
        "raw_text": "This is unacceptable, the fee dispute was denied again.",
    })
    flags = agent.detect(event, _state(recent_refs=refs))
    types = [f["type"] for f in flags]
    assert "negative_sentiment" in types and "repeated_contact" in types


def test_neutral_ticket_no_flags() -> None:
    agent = SupportAgent(board=None, llm=None)
    event = _ev(1, T0, "support_logs", "ticket_created", {
        "channel": "email", "category": "general_question",
        "raw_text": "What are your branch hours on Saturday?",
    })
    assert agent.detect(event, _state()) == []


# --- kyc agent: deterministic, confidence 1.0, ZERO LLM -----------------------

def test_kyc_agent_is_fully_deterministic() -> None:
    llm = FakeLLMClient()  # would record any LLM call
    agent = KycAgent(board=None, llm=llm)
    event = _ev(1, T0, "loan_kyc", "dependents_change", {"event_subtype": "dependents_change", "old_value": 1, "new_value": 2})
    flags = agent.detect(event, _state())
    types = [f["type"] for f in flags]
    assert "household_change" in types
    for f in flags:
        assert f["confidence"] == 1.0  # deterministic facts


def test_kyc_agent_never_calls_llm() -> None:
    """Zero-LLM design contract: narrate() must not touch the LLM."""
    llm = FakeLLMClient()
    agent = KycAgent(board=None, llm=llm)
    event = _ev(1, T0, "loan_kyc", "address_change", {"event_subtype": "address_change", "old_value": "12 Elm St", "new_value": "999 Foreign Ave"})
    flags = agent.detect(event, _state())
    summary = agent.narrate(event, flags)
    assert llm.calls == []  # ZERO LLM calls
    assert "address_change_recent" in summary


def test_kyc_watchlist_hit() -> None:
    agent = KycAgent(board=None, llm=None)
    event = _ev(1, T0, "loan_kyc", "kyc_update", {"event_subtype": "watchlist_hit"})
    flags = agent.detect(event, _state())
    assert any(f["type"] == "watchlist_hit" and f["confidence"] == 1.0 for f in flags)


# --- LLM fallback contract ----------------------------------------------------

def test_llm_failure_keeps_deterministic_summary() -> None:
    """Strict fallback contract: LLM unavailable => deterministic summary flows."""
    agent = TransactionAgent(board=None, llm=FakeLLMClient(fail=True))
    event = _ev(1, T0, "ach_wire", "outbound_transfer", {"amount": 12000})
    flags = agent.detect(event, _state())
    summary = agent.narrate(event, flags)
    assert "large_amount" in summary  # deterministic summary retained


def test_llm_enhances_summary_when_available() -> None:
    agent = TransactionAgent(board=None, llm=FakeLLMClient(responses=[{"summary": "A large outbound transfer was detected."}]))
    event = _ev(1, T0, "ach_wire", "outbound_transfer", {"amount": 12000})
    flags = agent.detect(event, _state())
    summary = agent.narrate(event, flags)
    assert summary == "A large outbound transfer was detected."


def test_no_flags_no_llm_call() -> None:
    llm = FakeLLMClient()
    agent = UsageAgent(board=None, llm=llm)
    event = _ev(1, T0, "web_app_events", "login", {})
    flags = agent.detect(event, _state(baseline={"login_count_7d": {"mean": 20.0, "std": 3.0}}))
    # flags may exist (engagement drop) - assert narrate with no flags skips LLM
    flags = []
    summary = agent.narrate(event, flags)
    assert llm.calls == []
    assert "no notable signals" in summary
