"""Phase 1: event schema validation tests."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.ingestion.schemas.events import SourceSystem, parse_event


def _base(**overrides) -> dict:
    data = {
        "event_id": "EVT_000001",
        "event_time": datetime(2026, 2, 1, 8, 0, tzinfo=UTC),
        "ingestion_time": datetime(2026, 2, 1, 8, 0, tzinfo=UTC),
        "customer_id": "CUST_00042",
        "account_id": "ACC_CHK_001",
        "source_system": "card_payments",
        "event_type": "purchase",
        "schema_version": "1.0",
        "payload": {
            "merchant_name": "Starbucks",
            "mcc_category": "dining",
            "amount": 20,
            "currency": "USD",
            "is_international": False,
            "card_present": False,
        },
    }
    data.update(overrides)
    return data


def test_valid_card_payment_parses() -> None:
    event = parse_event(_base())
    assert event.source == "card_payments"
    assert event.event_id == "EVT_000001"
    assert event.payload["amount"] == 20


def test_unknown_event_type_for_source_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_event(_base(event_type="deposit"))  # deposit is not a card_payments type


def test_unknown_source_system_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_event(_base(source_system="carrier_pigeon"))


def test_missing_required_payload_field_rejected() -> None:
    payload = {"merchant_name": "X"}  # amount missing for card_payments
    with pytest.raises(ValidationError):
        parse_event(_base(payload=payload))


def test_decline_with_reason_ok() -> None:
    event = parse_event(
        _base(
            event_type="decline",
            payload={
                "merchant_name": "Grocer",
                "mcc_category": "grocery",
                "amount": 95,
                "currency": "USD",
                "is_international": False,
                "card_present": True,
                "decline_reason": "insufficient_funds",
            },
        )
    )
    assert event.payload["decline_reason"] == "insufficient_funds"


def test_kyc_change_events_validate() -> None:
    event = parse_event(
        _base(
            source_system="loan_kyc",
            event_type="dependents_change",
            account_id=None,
            payload={"event_subtype": "dependents_change", "old_value": 1, "new_value": 2},
        )
    )
    assert event.source == SourceSystem.LOAN_KYC


def test_support_ticket_with_null_account_ok() -> None:
    event = parse_event(
        _base(
            source_system="support_logs",
            event_type="ticket_created",
            account_id=None,
            payload={
                "channel": "chat",
                "category": "payment_arrangements",
                "raw_text": "I need a payment plan",
                "resolution_status": "open",
            },
        )
    )
    assert event.account_id is None


def test_late_ingestion_detection() -> None:
    et = datetime(2026, 2, 1, 8, 0, tzinfo=UTC)
    event = parse_event(_base(ingestion_time=et + timedelta(minutes=10)))
    assert event.is_late() is True
    on_time = parse_event(_base())
    assert on_time.is_late() is False


def test_negative_amount_rejected() -> None:
    payload = {"merchant_name": "X", "mcc_category": "y", "amount": -5, "currency": "USD"}
    with pytest.raises(ValidationError):
        parse_event(_base(payload=payload))


def test_envelope_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        parse_event(_base(surprise_field=True))
