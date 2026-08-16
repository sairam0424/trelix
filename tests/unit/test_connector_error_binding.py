"""
Regression tests for ArtifactSource.sync()'s failure reporting.

sync() used to swallow per-artefact write failures with a bare `except
Exception:` — no `as exc`, no log call. A sync that reported "errors: 340"
gave the operator a count and nothing else: not the exception type, not the
message, not which artefact failed. These tests pin the two properties that
make such a run diagnosable AND readable: every failure is identified
somewhere in the log, and 340 failures do not become 340 WARNING lines.
"""

from __future__ import annotations

import logging

from trelix.core.models import Artifact
from trelix.indexing.connectors.base import _MAX_FAILURE_DETAIL_LOGS, ArtifactSource

_LOGGER_NAME = "trelix.indexing.connectors.base"


def _artifacts(n: int) -> list[Artifact]:
    return [
        Artifact(source_ref=f"ticket:{i}", artifact_kind="ticket", title=f"T{i}", body="")
        for i in range(n)
    ]


class _FakeSource(ArtifactSource):
    def __init__(self, artifacts: list[Artifact]) -> None:
        self._artifacts = artifacts

    def validate_config(self) -> None:
        return None

    def fetch(self) -> list[Artifact]:
        return self._artifacts


class _FailingWriter:
    """Raises a caller-chosen exception per artefact; None means "write OK"."""

    def __init__(self, error_for) -> None:  # type: ignore[no-untyped-def]
        self._error_for = error_for
        self.written: list[Artifact] = []

    def upsert_artifact(self, artifact: Artifact) -> int:
        exc = self._error_for(artifact)
        if exc is not None:
            raise exc
        self.written.append(artifact)
        return len(self.written)


class _FailingLinker:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def link_one(self, source_ref: str) -> int:
        raise self._exc


class TestSyncWriteFailureIsDiagnosable:
    def test_write_failure_logs_source_ref_type_and_message(self, caplog) -> None:  # type: ignore[no-untyped-def]
        source = _FakeSource(_artifacts(1))
        writer = _FailingWriter(
            lambda a: ValueError("UNIQUE constraint failed: artifacts.source_ref")
        )

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            result = source.sync(writer)  # type: ignore[arg-type]

        assert result.errors == 1
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "ticket:0" in text
        assert "ValueError" in text
        assert "UNIQUE constraint failed: artifacts.source_ref" in text

    def test_bulk_failures_do_not_emit_one_warning_each(self, caplog) -> None:  # type: ignore[no-untyped-def]
        source = _FakeSource(_artifacts(340))
        writer = _FailingWriter(lambda a: ValueError("boom"))

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            result = source.sync(writer)  # type: ignore[arg-type]

        assert result.errors == 340
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        # First N in full + exactly one aggregate line — never 340 WARNINGs.
        assert len(warnings) == _MAX_FAILURE_DETAIL_LOGS + 1
        # Every failure still identified somewhere, at DEBUG for the tail.
        assert sum(1 for r in caplog.records if "ticket:339" in r.getMessage()) == 1

    def test_summary_breaks_failures_down_by_exception_type(self, caplog) -> None:  # type: ignore[no-untyped-def]
        def error_for(artifact: Artifact) -> Exception | None:
            index = int(artifact.source_ref.split(":")[1])
            if index % 2 == 0:
                return ValueError("bad row")
            return RuntimeError("db locked")

        source = _FakeSource(_artifacts(10))
        writer = _FailingWriter(error_for)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            result = source.sync(writer)  # type: ignore[arg-type]

        assert result.errors == 10
        summary = [r.getMessage() for r in caplog.records if "10 of 10" in r.getMessage()]
        assert len(summary) == 1
        assert "ValueError x5" in summary[0]
        assert "RuntimeError x5" in summary[0]

    def test_successful_sync_logs_no_warning(self, caplog) -> None:  # type: ignore[no-untyped-def]
        source = _FakeSource(_artifacts(3))
        writer = _FailingWriter(lambda a: None)

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            result = source.sync(writer)  # type: ignore[arg-type]

        assert result.errors == 0
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


class TestSyncLinkFailureIsDiagnosable:
    def test_link_failure_is_reported_and_still_not_counted_as_error(self, caplog) -> None:  # type: ignore[no-untyped-def]
        source = _FakeSource(_artifacts(2))
        writer = _FailingWriter(lambda a: None)
        linker = _FailingLinker(KeyError("no such artifact"))

        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            result = source.sync(writer, linker=linker)  # type: ignore[arg-type]

        # Linking is best-effort: it must not inflate `errors`...
        assert result.errors == 0
        assert result.artifacts_written == 2
        assert result.edges_linked == 0
        # ...but it must not be silent either.
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "ticket:0" in text
        assert "KeyError" in text
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "2 of 2" in warnings[0]
        assert "KeyError x2" in warnings[0]
