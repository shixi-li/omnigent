"""
Tests for ``emit_routing_event`` in ``omnigent.runtime.telemetry``.

Each test installs a fresh TracerProvider with an InMemorySpanExporter
through the OTel public API, emits routing events, then asserts on the
exported spans / span events.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from omnigent.runtime import telemetry


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """
    Install a fresh TracerProvider with an in-memory exporter for one
    test and opt telemetry in. Restores the previous provider on
    teardown so OTel's set-once semantics do not leak into later tests.

    :param monkeypatch: Pytest monkeypatch fixture.
    :returns: The in-memory exporter collecting finished spans.
    """
    monkeypatch.setenv("OMNIGENT_TELEMETRY_ENABLED", "true")
    previous = otel_trace._TRACER_PROVIDER  # type: ignore[attr-defined]
    previous_done = otel_trace._TRACER_PROVIDER_SET_ONCE._done  # type: ignore[attr-defined]
    in_mem = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_mem))
    otel_trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    otel_trace._TRACER_PROVIDER_SET_ONCE._done = True  # type: ignore[attr-defined]
    try:
        yield in_mem
    finally:
        in_mem.clear()
        with contextlib.suppress(Exception):
            provider.shutdown()
        otel_trace._TRACER_PROVIDER = previous  # type: ignore[attr-defined]
        otel_trace._TRACER_PROVIDER_SET_ONCE._done = previous_done  # type: ignore[attr-defined]


def test_event_names_are_namespaced() -> None:
    for name in (
        telemetry.ROUTING_EVENT_DECISION,
        telemetry.ROUTING_EVENT_ENABLED,
        telemetry.ROUTING_EVENT_DISABLED_MID_SESSION,
        telemetry.ROUTING_EVENT_FORK_FROM_ROUTED_SESSION,
    ):
        assert name.startswith("omnigent.routing.")


def test_event_rides_the_active_span(exporter: InMemorySpanExporter) -> None:
    with telemetry.span("routing.test"):
        telemetry.emit_routing_event(
            telemetry.ROUTING_EVENT_DECISION,
            {"routing.model": "databricks-claude-opus-4-8", "routing.scope": "turn"},
        )
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "routing.test"
    events = list(spans[0].events)
    assert len(events) == 1
    assert events[0].name == telemetry.ROUTING_EVENT_DECISION
    attrs = dict(events[0].attributes or {})
    assert attrs["routing.model"] == "databricks-claude-opus-4-8"
    assert attrs["routing.scope"] == "turn"


def test_event_without_active_span_creates_one(exporter: InMemorySpanExporter) -> None:
    telemetry.emit_routing_event(telemetry.ROUTING_EVENT_ENABLED, {"routing.harness": "codex"})
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == telemetry.ROUTING_EVENT_ENABLED
    assert dict(spans[0].attributes or {})["routing.harness"] == "codex"
    assert [e.name for e in spans[0].events] == [telemetry.ROUTING_EVENT_ENABLED]


def test_none_attributes_dropped_and_values_coerced(
    exporter: InMemorySpanExporter,
) -> None:
    telemetry.emit_routing_event(
        telemetry.ROUTING_EVENT_FORK_FROM_ROUTED_SESSION,
        {
            "routing.raw_model": None,
            "routing.applied": True,
            "routing.attempt": 2,
            "routing.options": ["a", "b"],
        },
    )
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert "routing.raw_model" not in attrs
    assert attrs["routing.applied"] is True
    assert attrs["routing.attempt"] == 2
    assert attrs["routing.options"] == "['a', 'b']"


def test_no_attributes_is_allowed(exporter: InMemorySpanExporter) -> None:
    telemetry.emit_routing_event(telemetry.ROUTING_EVENT_DISABLED_MID_SESSION)
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == [telemetry.ROUTING_EVENT_DISABLED_MID_SESSION]


def test_emits_nothing_when_telemetry_disabled(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMNIGENT_TELEMETRY_ENABLED", "false")
    telemetry.emit_routing_event(telemetry.ROUTING_EVENT_DECISION, {"routing.scope": "turn"})
    assert exporter.get_finished_spans() == ()
