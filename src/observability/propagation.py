"""Context propagation across the event-bus boundary (Phase 11).

The bus message wraps the envelope with W3C TraceContext (``traceparent``):
the producer injects its active span context at publish time; the consumer
extracts it so downstream hops become children of the producer span - one
connected trace tree from event to decision (traceability #25).

Message shape: ``{"envelope": <event>, "trace": <carrier>}`` - the envelope
schema stays clean; trace metadata rides outside it.
"""
from __future__ import annotations

import json
from typing import Any

from opentelemetry import propagate

from src.ingestion.schemas.events import EventEnvelope

TRACE_KEY = "trace"
ENVELOPE_KEY = "envelope"


def publish_with_trace(event: EventEnvelope) -> str:
    """Serialize one bus message with the active trace context injected."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    message = {ENVELOPE_KEY: event.model_dump(mode="json"), TRACE_KEY: carrier}
    return json.dumps(message)


def consume_with_trace(message: str | bytes) -> tuple[EventEnvelope, dict[str, Any]]:
    """Parse one bus message; extract trace context for the consumer span."""
    if isinstance(message, bytes):
        message = message.decode()
    data = json.loads(message)
    event = EventEnvelope.model_validate(data[ENVELOPE_KEY])
    context = propagate.extract(data.get(TRACE_KEY) or {})
    return event, context
