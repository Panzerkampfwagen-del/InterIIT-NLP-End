"""Event contracts for all signal families (Phase 1).

Envelope + payload shapes mirror the Customer 360 evaluation dataset schema
(``evaluation/dataset/README_dataset_schema.md``) exactly, so a dataset event
validates without transformation. Extra payload fields are preserved (the
schema is additive), but the fields each source system is documented to carry
are validated when present.

NOTE (per docs/reference/04 Phase 1): the envelope carries the raw source-system
``customer_id`` - identity resolution (Phase 2) annotates and validates identity;
this schema deliberately does not assume a resolved identity.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceSystem(str, Enum):
    """The nine signal sources in the dataset schema."""

    CARD_PAYMENTS = "card_payments"
    INSTANT_PAYMENTS = "instant_payments"
    ACH_WIRE = "ach_wire"
    CORE_BANKING_LEDGER = "core_banking_ledger"
    TRADING_BROKERAGE = "trading_brokerage"
    LOAN_KYC = "loan_kyc"
    WEB_APP_EVENTS = "web_app_events"
    SUPPORT_LOGS = "support_logs"
    SOCIAL_SIGNAL_CONSENTED = "social_signal_consented"


# Documented event types per source system (dataset README).
EVENT_TYPES: dict[str, frozenset[str]] = {
    SourceSystem.CARD_PAYMENTS: frozenset({"purchase", "refund", "decline"}),
    SourceSystem.INSTANT_PAYMENTS: frozenset({"inbound_transfer", "outbound_transfer"}),
    SourceSystem.ACH_WIRE: frozenset({"inbound_transfer", "outbound_transfer"}),
    SourceSystem.CORE_BANKING_LEDGER: frozenset(
        {"deposit", "withdrawal", "standing_instruction", "fee", "interest_credit"}
    ),
    SourceSystem.TRADING_BROKERAGE: frozenset({"buy", "sell", "dividend", "deposit_to_brokerage"}),
    SourceSystem.LOAN_KYC: frozenset(
        {
            "loan_application",
            "loan_disbursed",
            "kyc_update",
            "address_change",
            "marital_status_change",
            "dependents_change",
        }
    ),
    SourceSystem.WEB_APP_EVENTS: frozenset(
        {"login", "search_query", "feature_used", "session_duration"}
    ),
    SourceSystem.SUPPORT_LOGS: frozenset({"ticket_created", "ticket_resolved", "call_transcript"}),
    SourceSystem.SOCIAL_SIGNAL_CONSENTED: frozenset({"life_event_mention"}),
}

# Documented payload fields per source system (dataset README).
PAYLOAD_FIELDS: dict[str, frozenset[str]] = {
    SourceSystem.CARD_PAYMENTS: frozenset(
        {
            "merchant_name",
            "mcc_category",
            "amount",
            "currency",
            "is_international",
            "card_present",
            "decline_reason",
        }
    ),
    SourceSystem.INSTANT_PAYMENTS: frozenset(
        {
            "direction",
            "amount",
            "currency",
            "counterparty_name",
            "counterparty_country",
            "transfer_type",
            "status",
        }
    ),
    SourceSystem.ACH_WIRE: frozenset(
        {
            "direction",
            "amount",
            "currency",
            "counterparty_name",
            "counterparty_country",
            "transfer_type",
            "status",
        }
    ),
    SourceSystem.CORE_BANKING_LEDGER: frozenset(
        {"amount", "balance_after", "transaction_type"}
    ),
    SourceSystem.TRADING_BROKERAGE: frozenset(
        {"instrument_type", "risk_category", "amount", "portfolio_value_after"}
    ),
    SourceSystem.LOAN_KYC: frozenset({"event_subtype", "old_value", "new_value"}),
    SourceSystem.WEB_APP_EVENTS: frozenset(
        {"feature_or_page", "search_text", "device_type", "session_length_sec"}
    ),
    SourceSystem.SUPPORT_LOGS: frozenset(
        {"channel", "category", "raw_text", "resolution_status"}
    ),
    SourceSystem.SOCIAL_SIGNAL_CONSENTED: frozenset({"platform", "raw_text", "consent_flag"}),
}


class EventEnvelope(BaseModel):
    """The common event envelope shared by history_seed and live_stream lines."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    event_time: datetime
    ingestion_time: datetime | None = None
    customer_id: str = Field(min_length=1)
    account_id: str | None = None
    source_system: SourceSystem
    event_type: str = Field(min_length=1)
    schema_version: str = Field(default="1.0")
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def _event_type_known(cls, v: str, info) -> str:
        source = (info.data or {}).get("source_system")
        if source is not None:
            allowed = EVENT_TYPES.get(source.value if isinstance(source, SourceSystem) else source, frozenset())
            if allowed and v not in allowed:
                raise ValueError(
                    f"event_type {v!r} is not documented for source_system {source!r}; "
                    f"allowed: {sorted(allowed)}"
                )
        return v

    @property
    def source(self) -> str:
        return self.source_system.value

    def is_late(self) -> bool:
        """True when ingestion_time is after event_time (arrived late)."""
        return self.ingestion_time is not None and self.ingestion_time > self.event_time


class CardPaymentPayload(BaseModel):
    """card_payments: purchase | refund | decline."""

    model_config = ConfigDict(extra="allow")

    merchant_name: str | None = None
    mcc_category: str | None = None
    amount: float = Field(ge=0)
    currency: str = "USD"
    is_international: bool = False
    card_present: bool = False
    decline_reason: str | None = None


class TransferPayload(BaseModel):
    """instant_payments / ach_wire: inbound_transfer | outbound_transfer."""

    model_config = ConfigDict(extra="allow")

    direction: str
    amount: float = Field(ge=0)
    currency: str = "USD"
    counterparty_name: str | None = None
    counterparty_country: str | None = None
    transfer_type: str | None = None
    status: str | None = None


class LedgerPayload(BaseModel):
    """core_banking_ledger: deposit | withdrawal | standing_instruction | fee | interest_credit."""

    model_config = ConfigDict(extra="allow")

    amount: float
    balance_after: float | None = None
    transaction_type: str | None = None


class BrokeragePayload(BaseModel):
    """trading_brokerage: buy | sell | dividend | deposit_to_brokerage."""

    model_config = ConfigDict(extra="allow")

    instrument_type: str | None = None
    risk_category: str | None = None
    amount: float | None = None
    portfolio_value_after: float | None = None


class KycPayload(BaseModel):
    """loan_kyc: loan_application | loan_disbursed | kyc_update | address_change | marital_status_change | dependents_change."""

    model_config = ConfigDict(extra="allow")

    event_subtype: str | None = None
    old_value: str | float | int | None = None
    new_value: str | float | int | None = None


class WebAppPayload(BaseModel):
    """web_app_events: login | search_query | feature_used | session_duration."""

    model_config = ConfigDict(extra="allow")

    feature_or_page: str | None = None
    search_text: str | None = None
    device_type: str | None = None
    session_length_sec: int | None = Field(default=None, ge=0)


class SupportPayload(BaseModel):
    """support_logs: ticket_created | ticket_resolved | call_transcript."""

    model_config = ConfigDict(extra="allow")

    channel: str | None = None
    category: str | None = None
    raw_text: str | None = None
    resolution_status: str | None = None


class SocialPayload(BaseModel):
    """social_signal_consented: life_event_mention (requires consent_flag)."""

    model_config = ConfigDict(extra="allow")

    platform: str | None = None
    raw_text: str | None = None
    consent_flag: bool


PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    SourceSystem.CARD_PAYMENTS: CardPaymentPayload,
    SourceSystem.INSTANT_PAYMENTS: TransferPayload,
    SourceSystem.ACH_WIRE: TransferPayload,
    SourceSystem.CORE_BANKING_LEDGER: LedgerPayload,
    SourceSystem.TRADING_BROKERAGE: BrokeragePayload,
    SourceSystem.LOAN_KYC: KycPayload,
    SourceSystem.WEB_APP_EVENTS: WebAppPayload,
    SourceSystem.SUPPORT_LOGS: SupportPayload,
    SourceSystem.SOCIAL_SIGNAL_CONSENTED: SocialPayload,
}


def validate_payload(event: EventEnvelope) -> None:
    """Validate the payload against the documented shape for its source system.

    Raises ``pydantic.ValidationError`` when documented fields are present but
    malformed; unknown extras are preserved (additive schema).
    """
    model = PAYLOAD_MODELS.get(event.source_system)
    if model is None:
        return
    model.model_validate(event.payload)


def parse_event(raw: dict[str, Any]) -> EventEnvelope:
    """Parse and validate one raw event dict into an EventEnvelope."""
    event = EventEnvelope.model_validate(raw)
    validate_payload(event)
    return event
