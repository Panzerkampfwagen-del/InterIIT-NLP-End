"""OpenTelemetry setup (Phase 11) - GenAI semantic conventions.

Per RQ-O1: vendor-neutral ``gen_ai.*`` span schema composes model calls, tool
calls, and retrieval steps into one hierarchical trace tree; 100% of traces
are captured (no sampling) at project scale. Custom ``c360.*`` attributes are
customer_id-scoped and PII-redacted (never raw names/emails/phones).

The default exporter ships OTLP/HTTP to the local collector (Phase 0 infra);
tests inject an InMemorySpanExporter (offline-deterministic).
"""
from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON


def setup_otel(
    service_name: str = "agentic-customer-360",
    exporter: SpanExporter | None = None,
) -> TracerProvider:
    """Configure the global tracer provider; 100% sampled (no sampling)."""
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource, sampler=ALWAYS_ON)
    if exporter is None:
        exporter = _default_exporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # The global provider can only be set ONCE per process; tests construct
    # their own provider and call provider.get_tracer directly.
    current = trace.get_tracer_provider()
    if type(current).__name__ in ("ProxyTracerProvider", "NoneType"):
        trace.set_tracer_provider(provider)
    return provider


def _default_exporter() -> SpanExporter:
    """OTLP/HTTP exporter to the local collector; console fallback offline."""
    from src.config import get_settings

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        s = get_settings()
        return OTLPSpanExporter(endpoint=s.otel_endpoint, timeout=5)
    except Exception:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()


def get_tracer(name: str = "c360") -> trace.Tracer:
    return trace.get_tracer(name)


def c360_attributes(customer_id: str, **extra: Any) -> dict[str, Any]:
    """Standard c360.* span attributes (customer_id-scoped, PII-redacted).

    Only pseudonymous ids go into attributes - never raw names, emails,
    or phone numbers (PII scanner enforces this).
    """
    attrs: dict[str, Any] = {"c360.customer_id": customer_id}
    for k, v in extra.items():
        if v is not None:
            attrs[f"c360.{k}"] = v
    return attrs
