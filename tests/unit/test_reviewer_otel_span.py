"""One `trelix.review` span per hunk, carrying `trelix.review.hunk_status` (roadmap C-8, PR 4).

`DiffReviewer.review()` opens a `trelix.review` span around every hunk it attempts
(`_review_one`) and, once `_classify_reply()` has decided the hunk's status, sets
`trelix.review.hunk_status` on it. The status is known only after `complete()` has returned, so it
cannot live on the `chat` span (ended by then): it lives on this per-hunk parent, under which the
hunk's `trelix.retrieve` span, its legs and its `chat` span(s) nest by context propagation alone.

Tracing is switched on per reviewer instance (`reviewer._config.retrieval.otel_enabled = True`)
because the unit suite scrubs `TRELIX_OTEL_ENABLED`. The two fixtures are copies of
tests/unit/test_otel_tracing.py's (they attach to the incumbent, one-shot global TracerProvider),
which is why this file is in scripts/mutation.py::DESELECTED_FILES. The enabled-path tests skip
without `pip install trelix[otel]`; the flag-off test runs everywhere.

Mutations each group was checked against (every one fails a test below):
- set `hunk_status` before classification -> three `reviewed` instead of reviewed/truncated/refused;
- drop the `set_attribute` call -> KeyError on the attribute;
- let the handled exception escape the `with`, or `record_exception(exc)` -> status ERROR and the
  exception text on the span;
- open a span per remaining hunk after the placeholder -> two spans instead of one;
- swallow `LLMNotConfiguredError` inside `_review_one` -> two calls and `llm_available` True;
- wrap only the `set_attribute` call, after `_review_hunk` returned -> the chat span has no parent;
- `from opentelemetry import trace` at the top of reviewer.py -> the flag-off review imports it;
- in `set_attribute`: drop the `trelix.{stage}.` prefix -> bare key; drop the `None` guard -> a
  DEBUG record; drop the try/except -> AttributeError on a broken span.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.unit.otel_llm_fakes import UPSTREAM_MODE, span_text
from trelix.core.config import IndexConfig
from trelix.llm.client import ChatResponse
from trelix.review.diff_parser import DiffHunk
from trelix.review.reviewer import DiffReviewer

_ONE = '[{"line_start": 10, "line_end": 11, "severity": "WARN", "comment": "check this"}]'
_TWO_THEN_CUT = (
    '[{"line_start": 1, "line_end": 1, "severity": "INFO", "comment": "first"},'
    ' {"line_start": 2, "line_end": 2, "severity": "WARN", "comment": "second"},'
    ' {"line_start": 3, "line_end": 3, "severity": "ERR'
)


def _cfg(
    otel_enabled: bool,
    service_name: str = "trelix-test",
    otel_exporter_endpoint: str | None = None,
    otel_capture_content: bool | None = None,
) -> SimpleNamespace:
    """A RetrievalConfig stand-in. `otel_capture_content=None` leaves the attribute OFF the
    object, the shape of every config built before the flag existed."""
    fields: dict[str, Any] = {
        "otel_enabled": otel_enabled,
        "otel_service_name": service_name,
        "otel_exporter_endpoint": otel_exporter_endpoint,
    }
    if otel_capture_content is not None:
        fields = {**fields, "otel_capture_content": otel_capture_content}
    return SimpleNamespace(**fields)


def _hunk(path: str = "src/auth.py", start: int = 10) -> DiffHunk:
    return DiffHunk(
        file_path=path,
        old_start=start,
        new_start=start,
        old_lines=3,
        new_lines=4,
        added=["    return True"],
        removed=["    return self._check()"],
        context=["def login(self):"],
    )


def _reply(
    content: str,
    finish: str = "stop",
    raw: str | None = None,
    model: str = "gpt-test",
    output_tokens: int = 0,
) -> ChatResponse:
    return ChatResponse(
        content=content,
        model=model,
        finish_reason=finish,
        raw_finish_reason=raw,
        output_tokens=output_tokens,
    )


def _traced_reviewer(tmp_path: Path, *replies: object) -> tuple[DiffReviewer, MagicMock]:
    """A DiffReviewer with tracing on for this instance only (service name `trelix`, no endpoint,
    so no exporter is built and the fixture's provider receives the spans), a MagicMock retriever
    and a client that answers with *replies* in order."""
    reviewer = DiffReviewer(IndexConfig(repo_path=str(tmp_path)))
    reviewer._config.retrieval.otel_enabled = True
    reviewer._retriever = MagicMock()
    reviewer._retriever.retrieve.return_value = MagicMock(context_text="ctx", results=[])
    client = MagicMock()
    client.complete.side_effect = list(replies)
    reviewer._llm_client = client
    return reviewer, client


# ---------------------------------------------------------------------------
# Enabled path — real spans via InMemorySpanExporter
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _test_tracer_provider():
    """
    Get an InMemorySpanExporter fed by the process's global TracerProvider.

    ATTACHES to whatever provider is installed rather than assuming it can
    install its own, because the global TracerProvider is a ONE-SHOT process
    resource: `set_tracer_provider()` is guarded by a `Once`, so the second
    caller anywhere in the session gets a silent no-op plus a logged
    "Overriding of current TracerProvider is not allowed".

    test_otel_tracing.py used to be that second caller. tests/unit/test_structured_logging.py
    installs a bare `TracerProvider()` (no span processors) and never restores it, so
    under reverse collection order — where test_structured_logging sorts after
    test_otel_tracing and therefore runs first — this fixture's provider was
    discarded, every span went to a processor-less provider, and all five
    "enabled path" tests below failed on empty span lists (three on
    `len(spans) == 1` against `()`, two on a KeyError/`in spans` miss). Measured:
    `pytest tests/unit/test_otel_tracing.py tests/unit/test_otel_metrics.py
    tests/unit/test_structured_logging.py` was 38 passed forward, 6 failed
    reversed — these 5 plus one unrelated failure inside test_otel_metrics.py.

    Adding a SpanProcessor to the incumbent SDK provider is enough — the
    exporter sees every sampled span regardless of who owns the provider — and
    is order-independent in both directions. A non-SDK incumbent (e.g. a
    NoOpTracerProvider) has no `add_span_processor`, so that case fails loudly
    instead of silently collecting zero spans.

    Skips (rather than errors) when `opentelemetry-sdk` isn't installed —
    CI's default `pip install -e ".[local,dev]"` deliberately does NOT include
    the optional `otel` extra, since otel_enabled=False is the documented,
    tested-elsewhere default; these "enabled path" tests only run when a
    developer/CI job explicitly installs `trelix[otel]`.
    """
    pytest.importorskip("opentelemetry.sdk", reason="requires pip install trelix[otel]")

    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        # Nothing real installed yet (ProxyTracerProvider), so claim the slot.
        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "trelix-test"}))
        trace.set_tracer_provider(provider)
        installed = trace.get_tracer_provider()
        if installed is not provider:
            pytest.fail(
                "the global TracerProvider is held by a non-SDK "
                f"{type(installed).__name__}, which cannot export spans — some earlier "
                "test set it and did not restore it"
            )

    from tests.unit.otel_llm_fakes import exporting_spans

    # The provider is shared with the rest of the session: the processor is detached
    # again on teardown (shutting it down alone leaves it attached, see exporting_spans).
    with exporting_spans(provider) as exporter:
        yield exporter


@pytest.fixture()
def otel_test_exporter(_test_tracer_provider, monkeypatch: pytest.MonkeyPatch):
    """
    Yield the module-wide InMemorySpanExporter, cleared before each test.
    Also resets trelix's memoized TelemetryHandler so _get_handler() rebuilds
    it against the real (already-installed) TracerProvider on next use,
    since a freshly-imported handler observed the ProxyTracerProvider only
    once, at first import.

    The two upstream content variables are removed BEFORE the handler is reset:
    the handler reads them once, when it is built, so a developer shell with
    SPAN_ONLY exported would otherwise leak into every rebuilt handler. The
    content-gate state (`_env_otel_settings`, `_capture_inert_warned`) is reset
    alongside the handler so each test starts from "never warned".
    """
    import trelix.retrieval.otel_tracing as otel_tracing

    monkeypatch.delenv(UPSTREAM_MODE, raising=False)
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", raising=False)
    _test_tracer_provider.clear()
    prev_handler = otel_tracing._handler
    prev_service_name = otel_tracing._handler_service_name
    otel_tracing._handler = None
    otel_tracing._handler_service_name = None
    monkeypatch.setattr(otel_tracing, "_env_otel_settings", None)
    monkeypatch.setattr(otel_tracing, "_capture_inert_warned", False)
    try:
        yield _test_tracer_provider
    finally:
        otel_tracing._handler = prev_handler
        otel_tracing._handler_service_name = prev_service_name


class TestOneSpanPerHunk:
    def test_three_hunks_make_three_spans_carrying_their_status(
        self, tmp_path: Path, otel_test_exporter
    ) -> None:
        """MUTATIONS: set the attribute before classification -> three `reviewed`; drop
        `set_attribute` -> KeyError on the attribute."""
        from opentelemetry.trace import StatusCode

        reviewer, client = _traced_reviewer(
            tmp_path,
            _reply("[]"),
            _reply(_TWO_THEN_CUT, "length"),
            _reply(_TWO_THEN_CUT, "length"),
            _reply(_ONE, finish="refusal"),
        )

        reviewer.review([_hunk("a.py", 1), _hunk("b.py", 2), _hunk("c.py", 3)])

        spans = otel_test_exporter.get_finished_spans()
        assert [s.name for s in spans] == ["trelix.review", "trelix.review", "trelix.review"]
        assert [s.attributes["trelix.review.hunk_status"] for s in spans] == [
            "reviewed",
            "truncated",
            "refused",
        ]
        assert all(s.status.status_code is StatusCode.UNSET for s in spans)
        assert client.complete.call_count == 4
        assert reviewer.last_outcome.hunks_failed == 2

    def test_a_raising_call_is_an_error_span_that_never_sees_the_exception_text(
        self, tmp_path: Path, otel_test_exporter
    ) -> None:
        """MUTATIONS: let the generic exception escape the `with` -> status ERROR and the canary
        in the exception event; `span.record_exception(exc)` -> the canary is present."""
        from opentelemetry.trace import StatusCode

        reviewer, _ = _traced_reviewer(tmp_path, RuntimeError("401 credential canary rejected"))

        assert reviewer.review([_hunk()]) == []

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.name == "trelix.review"
        assert span.attributes["trelix.review.hunk_status"] == "error"
        assert span.status.status_code is StatusCode.UNSET
        assert "credential canary" not in span_text(span)
        assert reviewer.last_outcome.hunk_results[0].detail == "exception:RuntimeError"

    def test_an_unconfigured_backend_makes_one_error_span_for_two_hunks(
        self, tmp_path: Path, otel_test_exporter
    ) -> None:
        """The placeholder branch breaks the loop after the first hunk, so N hunks give ONE span
        while the outcome counts N errors; the exception crosses the span, so the SDK marks it
        ERROR (owner decision D11). MUTATIONS: a span per remaining hunk -> two spans; swallow
        `LLMNotConfiguredError` in `_review_one` -> `call_count == 2`, `llm_available is True`."""
        from opentelemetry.trace import StatusCode

        reviewer, client = _traced_reviewer(
            tmp_path,
            _reply('[{"line_start": 1, "comment": "looks like a finding"}]', model="none"),
        )

        assert reviewer.review([_hunk(), _hunk("src/b.py")]) == []

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.name == "trelix.review"
        assert span.attributes["trelix.review.hunk_status"] == "error"
        assert span.status.status_code is StatusCode.ERROR
        assert client.complete.call_count == 1
        assert reviewer.last_outcome.llm_available is False
        assert reviewer.last_outcome.hunks_failed == 2
        assert [r.detail for r in reviewer.last_outcome.hunk_results] == [
            "not_configured",
            "not_configured",
        ]


class TestNesting:
    def test_the_hunk_span_is_the_parent_of_work_done_during_the_call(
        self, tmp_path: Path, otel_test_exporter
    ) -> None:
        """A span opened while `complete()` runs (the chat span in production) nests under the
        hunk's span by context alone. MUTATION: wrap only the `set_attribute` call, after
        `_review_hunk` returned -> `spans[0].parent is None`."""

        def _complete(**kwargs: Any) -> ChatResponse:
            from opentelemetry import trace

            with trace.get_tracer("test").start_as_current_span("fake-chat"):
                pass
            return _reply("[]")

        reviewer, client = _traced_reviewer(tmp_path)
        client.complete.side_effect = _complete

        reviewer.review([_hunk()])

        spans = otel_test_exporter.get_finished_spans()
        assert [s.name for s in spans] == ["fake-chat", "trelix.review"]
        assert spans[0].parent.span_id == spans[1].context.span_id


class TestSetAttribute:
    def test_set_attribute_prefixes_the_stage_and_is_silent_without_a_span(
        self, otel_test_exporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`s` is always the context manager (`__enter__` returns self), never a span.
        MUTATIONS: drop the `trelix.{stage}.` prefix -> key `hunk_status`; drop the `None` guard
        -> `None.set_attribute` raises inside the `try` and a DEBUG record appears; drop the
        try/except -> AttributeError on the broken-span case."""
        from trelix.retrieval.otel_tracing import pipeline_stage_span

        cfg = _cfg(otel_enabled=True)
        with pipeline_stage_span(cfg, "review") as s:
            s.set_attribute("hunk_status", "refused")

        (span,) = otel_test_exporter.get_finished_spans()
        assert span.name == "trelix.review"
        assert dict(span.attributes) == {"trelix.review.hunk_status": "refused"}
        otel_test_exporter.clear()

        # Both no-span paths: tracing off, and constructed but never entered.
        with caplog.at_level(logging.DEBUG, logger="trelix.retrieval.otel"):
            with pipeline_stage_span(_cfg(otel_enabled=False), "review") as s:
                s.set_attribute("hunk_status", "x")
            pipeline_stage_span(cfg, "review").set_attribute("early", 1)
        assert otel_test_exporter.get_finished_spans() == ()
        assert caplog.records == []

        # A broken span object: the guard's try/except keeps the failure out of the review (this
        # one DOES log a DEBUG record by design, so it runs after the no-record assertion).
        with pipeline_stage_span(cfg, "review") as s:
            s._span = object()
            s.set_attribute("k", "v")


# ---------------------------------------------------------------------------
# Off path — no span, no opentelemetry import
# ---------------------------------------------------------------------------


class TestOffPath:
    def test_a_flag_off_review_imports_no_opentelemetry(self, tmp_path: Path) -> None:
        """With `otel_enabled` at the scrubbed default, a review through a FRESHLY imported
        reviewer module leaves no `opentelemetry` module behind.
        MUTATION: `from opentelemetry import trace` at the top of reviewer.py."""
        purged = {k: v for k, v in sys.modules.items() if k.startswith("opentelemetry")}
        reviewer_mod = sys.modules.pop("trelix.review.reviewer", None)
        otel_tracing_mod = sys.modules.pop("trelix.retrieval.otel_tracing", None)
        for k in purged:
            del sys.modules[k]
        try:
            import trelix.review.reviewer  # noqa: F401

            fresh = sys.modules["trelix.review.reviewer"].DiffReviewer(
                IndexConfig(repo_path=str(tmp_path))
            )
            fresh._retriever = MagicMock()
            fresh._retriever.retrieve.return_value = MagicMock(context_text="ctx", results=[])
            client = MagicMock()
            client.complete.side_effect = [_reply("[]")]
            fresh._llm_client = client

            assert fresh.review([_hunk()]) == []
            assert fresh.last_outcome.hunks_failed == 0
            assert not any(k.startswith("opentelemetry") for k in sys.modules)
        finally:
            sys.modules.update(purged)
            # Restoring sys.modules is NOT enough (see the same step in
            # test_otel_tracing.py::test_retriever_import_does_not_import_opentelemetry): the
            # fresh import above bound NEW module objects as the PACKAGE ATTRIBUTES
            # `trelix.review.reviewer` and `trelix.retrieval.otel_tracing`, so a later
            # `from trelix.retrieval import otel_tracing`, or a fixture that resets its
            # module-level handler, would otherwise operate on an orphan.
            for package_name, name, module in (
                ("trelix.review", "reviewer", reviewer_mod),
                ("trelix.retrieval", "otel_tracing", otel_tracing_mod),
            ):
                if module is None:
                    continue
                sys.modules[f"{package_name}.{name}"] = module
                package = sys.modules.get(package_name)
                if package is not None:
                    setattr(package, name, module)
