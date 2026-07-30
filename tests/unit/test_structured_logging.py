"""
Tests for the unified logging setup (src/trelix/core/logging_setup.py).

Captures real emitted output (via a StringIO handler swapped in after
setup_*_logging() configures the root logger) rather than mocking structlog
internals — proves the actual rendered output, not just that some processor
was called.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable

import pytest

from trelix.core.logging_setup import (
    setup_console_logging,
    setup_json_logging,
    uvicorn_log_config,
)


def _swap_root_handler_streams(buf: io.StringIO) -> None:
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream = buf


def _capture_one_line(setup_fn: Callable[[int], None], level: int = logging.INFO) -> str:
    """Run setup_fn(level), swap the root handler's stream for a capture
    buffer, emit one log line, and return exactly that line (stripped)."""
    setup_fn(level)
    buf = io.StringIO()
    _swap_root_handler_streams(buf)
    logging.getLogger("trelix.test").info("hello %s", "world")
    return buf.getvalue().strip()


class TestConsoleLogging:
    def test_produces_plain_text_not_json(self) -> None:
        """CLI mode must stay human-readable — a JSON-shaped line here
        would mean setup_console_logging regressed to the JSON renderer."""
        line = _capture_one_line(setup_console_logging)
        with pytest.raises(json.JSONDecodeError):
            json.loads(line)

    def test_includes_event_and_logger_name(self) -> None:
        line = _capture_one_line(setup_console_logging)
        assert "hello world" in line
        assert "trelix.test" in line

    def test_respects_level_threshold(self) -> None:
        """WARNING-level setup must suppress an INFO-level record."""
        setup_console_logging(logging.WARNING)
        buf = io.StringIO()
        _swap_root_handler_streams(buf)
        logging.getLogger("trelix.test").info("should not appear")
        assert buf.getvalue().strip() == ""


class TestJSONLogging:
    def test_produces_valid_json(self) -> None:
        line = _capture_one_line(setup_json_logging)
        json.loads(line)  # raises if not valid JSON

    def test_has_expected_keys(self) -> None:
        line = _capture_one_line(setup_json_logging)
        data = json.loads(line)
        assert data["event"] == "hello world"
        assert data["logger"] == "trelix.test"
        assert data["level"] == "info"
        assert "timestamp" in data

    def test_trace_context_absent_when_no_span_active(self) -> None:
        """With no OTel span active, trace_id/span_id must be absent — the
        default (is_valid=False) span context, not a missing import, is
        what actually gates this in production (opentelemetry is a base
        dependency of trelix[otel]-adjacent tooling like langsmith, so it
        may already be imported by the time this runs)."""
        pytest.importorskip("opentelemetry")
        from opentelemetry import trace

        # No active span in this test's context -> get_current_span() is a
        # no-op span whose context is_valid is False.
        assert not trace.get_current_span().get_span_context().is_valid

        line = _capture_one_line(setup_json_logging)
        data = json.loads(line)
        assert "trace_id" not in data
        assert "span_id" not in data

    def test_trace_context_present_when_span_active(self) -> None:
        """With a real OTel span active, trace_id/span_id must appear and be
        valid hex — mirrors otel_tracing.py's existing pattern."""
        pytest.importorskip("opentelemetry")
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer("test")

        setup_json_logging()
        buf = io.StringIO()
        _swap_root_handler_streams(buf)

        with tracer.start_as_current_span("test-span"):
            logging.getLogger("trelix.test").info("inside span")

        data = json.loads(buf.getvalue().strip())
        assert len(data["trace_id"]) == 32
        assert len(data["span_id"]) == 16
        int(data["trace_id"], 16)  # raises ValueError if not valid hex
        int(data["span_id"], 16)


class TestUvicornLogConfig:
    def test_returns_dict_with_required_uvicorn_keys(self) -> None:
        config = uvicorn_log_config()
        assert config["version"] == 1
        assert "uvicorn" in config["loggers"]
        assert "uvicorn.access" in config["loggers"]
        assert "uvicorn.error" in config["loggers"]

    def test_level_option_is_applied(self) -> None:
        config = uvicorn_log_config(level="debug")
        assert config["loggers"]["uvicorn"]["level"] == "DEBUG"
