"""
Unified logging setup — structlog's ProcessorFormatter over stdlib logging.

Before this module, trelix used `logging.basicConfig` with a plain-text
formatter, called inconsistently (`serve`/`stats`/`graph`/`review` never
called it at all), and with no correlation to the OTel spans that already
exist on every REST route (see retrieval/otel_tracing.py).

ProcessorFormatter is the key mechanism: it lets every one of the ~164
existing `logger.*` call sites across the codebase keep working completely
unchanged (they still just call `logging.getLogger("trelix.foo").info(...)`)
while a single shared processor chain renders every record — whether it
originated from stdlib logging or from a future structlog-native call site.
No call-site rewrite is required for this phase.

Two output modes, matching the CLI-vs-server split:
  - CLI (`setup_console_logging`): human-readable console output — used by
    `trelix index`/`search`/`ask`/etc via cli/main.py's `_setup_logging()`.
  - Server (`setup_json_logging`): JSON lines — used by `trelix serve`
    before `uvicorn.run(...)`, together with `uvicorn_log_config()` so
    uvicorn's own access-log lines become JSON too, not just the app's.

Trace correlation: `_inject_trace_context` adds `trace_id`/`span_id` to every
log entry emitted while an OTel span is active, gated by
`RetrievalConfig.otel_enabled` — mirrors otel_tracing.py's own zero-cost-
when-disabled pattern, and never imports `opentelemetry` when disabled.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_LIBS_TO_QUIET = ("httpx", "httpcore", "openai", "sentence_transformers", "transformers")


def _quiet_noisy_libraries() -> None:
    for lib in _LIBS_TO_QUIET:
        logging.getLogger(lib).setLevel(logging.WARNING)


def _inject_trace_context(
    _logger: Any, _method_name: str, event_dict: structlog.typing.EventDict
) -> structlog.typing.EventDict:
    """Add trace_id/span_id to event_dict when an OTel span is active.

    Never imports opentelemetry unless a caller has already done so
    elsewhere in the process (otel is only ever imported when
    TRELIX_OTEL_ENABLED=true triggers otel_tracing.py's own lazy import) —
    checking sys.modules first keeps this a true no-op otherwise.
    """
    if "opentelemetry" not in sys.modules:
        return event_dict
    try:
        from opentelemetry import trace

        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            event_dict["trace_id"] = format(span_context.trace_id, "032x")
            event_dict["span_id"] = format(span_context.span_id, "016x")
    except Exception:  # noqa: BLE001 — logging setup must never crash the caller
        pass
    return event_dict


_SHARED_PROCESSORS: list[structlog.typing.Processor] = [
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
    _inject_trace_context,
]


def _configure_stdlib_root(level: int, renderer: structlog.typing.Processor) -> None:
    """Attach a single ProcessorFormatter-backed handler to the root logger.

    Replaces any handlers a prior call installed (idempotent — safe to call
    once per CLI command invocation, which is the existing _setup_logging()
    contract).
    """
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def setup_console_logging(level: int = logging.WARNING) -> None:
    """CLI mode — human-readable console output. Called from
    cli/main.py's _setup_logging() at the start of every CLI command.

    colors=False: this codebase's actual color output comes from Rich
    (cli/main.py's Console/Panel/Table), never from structlog's own
    ConsoleRenderer — the ~164 stdlib logger.* call sites this renders are
    incidental, mostly-invisible-at-default-level plumbing, not the CLI's
    visible UI. Leaving colors on its Windows default (auto-True whenever
    colorama is importable) makes every command call colorama.init() on
    construction, which monkeypatches sys.stdout/stderr into
    AnsiToWin32/StreamWrapper objects — confirmed to collide with Rich's
    own legacy_windows_render path on the same stream, crashing with
    `OSError: [Errno 22] Invalid argument` the next time Rich writes
    (reproduced on the Windows binary-build CI job). Rich already handles
    its own Windows color/encoding story independently; structlog's is
    redundant and actively harmful here.
    """
    _configure_stdlib_root(level, structlog.dev.ConsoleRenderer(colors=False))
    _quiet_noisy_libraries()


def setup_json_logging(level: int = logging.INFO) -> None:
    """Server mode — JSON lines. Called from the `serve` command before
    uvicorn.run(...), so every trelix.* logger call emits parseable JSON."""
    _configure_stdlib_root(level, structlog.processors.JSONRenderer())
    _quiet_noisy_libraries()


def uvicorn_log_config(level: str = "info") -> dict[str, Any]:
    """A uvicorn `log_config=` dict that renders uvicorn's own access/error
    logs as JSON via the same ProcessorFormatter chain — otherwise
    `trelix serve`'s output is half-JSON (the app's own logger) and
    half-uvicorn's-default-colorized-text (uvicorn.access/uvicorn.error),
    defeating the point of JSON server logs."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "trelix_json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "foreign_pre_chain": _SHARED_PROCESSORS,
                "processors": [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "trelix_json",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": level.upper(), "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": level.upper(), "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": level.upper(), "propagate": False},
        },
    }
