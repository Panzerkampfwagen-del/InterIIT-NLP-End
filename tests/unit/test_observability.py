"""Phase 11: observability - span tree completeness, zero PII, propagation."""
from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.ingestion.schemas.events import parse_event
from src.observability.otel_setup import c360_attributes, setup_otel
from src.observability.pii_scan import scan_spans
from src.observability.propagation import consume_with_trace, publish_with_trace


@pytest.fixture()
def otel_env():
    exporter = InMemorySpanExporter()
    provider = setup_otel(service_name="c360-test", exporter=exporter)
    tracer = provider.get_tracer("c360-test")  # per-provider tracer (global set-once)
    yield tracer, exporter
    provider.shutdown()


def test_one_connected_span_tree_per_decision(otel_env) -> None:
    """One connected span tree per decision: all hops share the root trace."""
    tracer, exporter = otel_env
    with tracer.start_as_current_span("decision_pipeline", attributes=c360_attributes("CUST_A", decision_id="dec_1")):
        with tracer.start_as_current_span("agent.transaction_agent", attributes=c360_attributes("CUST_A")):
            with tracer.start_as_current_span("tool.state_update", attributes=c360_attributes("CUST_A")):
                pass
        with tracer.start_as_current_span("agent.life_event_agent", attributes=c360_attributes("CUST_A")):
            with tracer.start_as_current_span("tool.retrieval", attributes=c360_attributes("CUST_A")):
                pass
    spans = exporter.get_finished_spans()
    assert len(spans) == 5
    # every span shares ONE root trace_id (connected tree)
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1
    # hierarchy: agent spans are children of the decision span
    root = next(s for s in spans if s.name == "decision_pipeline")
    agents = [s for s in spans if s.name.startswith("agent.")]
    assert all(s.parent.span_id == root.context.span_id for s in agents)
    tools = [s for s in spans if s.name.startswith("tool.")]
    assert all(s.parent is not None for s in tools)


def test_zero_raw_pii_in_spans(otel_env) -> None:
    """Zero raw-PII matches: c360.* attributes carry only pseudonymous ids."""
    tracer, exporter = otel_env
    with tracer.start_as_current_span("agent.support_agent", attributes=c360_attributes("CUST_00088", domain="support")):
        with tracer.start_as_current_span("tool.retrieval", attributes=c360_attributes("CUST_00088", top_k=5)):
            pass
    spans = exporter.get_finished_spans()
    violations = scan_spans(spans)
    assert violations == {}, f"raw PII found in spans: {violations}"


def test_pii_scanner_flags_violations_negative_control(otel_env) -> None:
    """Negative control: the scanner DOES flag raw PII when present."""
    tracer, exporter = otel_env
    with tracer.start_as_current_span("bad_span", attributes={"c360.contact": "jane.doe@example.com", "c360.phone": "555-123-4567"}):
        pass
    spans = exporter.get_finished_spans()
    violations = scan_spans(spans)
    assert "bad_span" in violations
    kinds = {k.split(":")[-1] for k, _ in violations["bad_span"]}
    assert "email" in kinds and "phone" in kinds


def test_context_propagation_across_bus_boundary(otel_env) -> None:
    """Producer span -> inject -> bus message -> consume -> extract -> child span
    (one connected trace across the event-bus boundary)."""
    tracer, exporter = otel_env
    event = parse_event(
        {
            "event_id": "EVT_000001",
            "event_time": "2026-05-01T09:00:00Z",
            "ingestion_time": "2026-05-01T09:00:00Z",
            "customer_id": "CUST_A",
            "account_id": None,
            "source_system": "support_logs",
            "event_type": "ticket_created",
            "schema_version": "1.0",
            "payload": {"channel": "chat", "raw_text": "hospital hardship"},
        }
    )
    # producer: active span -> inject trace context into the bus message
    with tracer.start_as_current_span("producer.publish", attributes=c360_attributes("CUST_A")):
        message = publish_with_trace(event)

    # consumer: extract -> child span continues the SAME trace
    event2, context = consume_with_trace(message)
    assert event2.event_id == "EVT_000001"
    with tracer.start_as_current_span("consumer.process", context=context, attributes=c360_attributes("CUST_A")):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    producer_span = next(s for s in spans if s.name == "producer.publish")
    consumer_span = next(s for s in spans if s.name == "consumer.process")
    # connected: same trace_id, consumer's parent is the producer span
    assert consumer_span.context.trace_id == producer_span.context.trace_id
    assert consumer_span.parent is not None
    assert consumer_span.parent.span_id == producer_span.context.span_id


def test_message_wraps_envelope_with_trace_carrier() -> None:
    """The bus message wraps the envelope; trace metadata rides outside it."""
    import json

    from src.ingestion.schemas.events import parse_event

    event = parse_event(
        {
            "event_id": "EVT_000002",
            "event_time": "2026-05-01T09:00:00Z",
            "ingestion_time": "2026-05-01T09:00:00Z",
            "customer_id": "CUST_A",
            "account_id": None,
            "source_system": "web_app_events",
            "event_type": "login",
            "schema_version": "1.0",
            "payload": {},
        }
    )
    message = publish_with_trace(event)
    data = json.loads(message)
    assert "envelope" in data and "trace" in data
    assert data["envelope"]["event_id"] == "EVT_000002"
    event2, _ = consume_with_trace(message)
    assert event2.model_dump(mode="json") == event.model_dump(mode="json")
