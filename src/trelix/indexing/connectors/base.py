"""
ArtifactSource — the base interface every source connector implements.

Deliberately NOT built on BaseParser (src/trelix/indexing/parser/base.py):
that ABC is welded to on-disk file semantics (parse(source: str, file_id:
int) -> ParseResult, keyed by Tree-sitter symbols with real
file_id/line-spans). A Jira ticket or TestRail test case has no source
text on disk and no line-span — forcing it through BaseParser would mean
fabricating meaningless values for fields that don't apply. ArtifactSource
is a parallel, independent interface for exactly this reason.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from trelix.core.models import Artifact

if TYPE_CHECKING:
    from trelix.indexing.artifact_linker import ArtifactLinker

logger = logging.getLogger("trelix.indexing.connectors.base")

# How many per-artefact failures get a full WARNING before sync() falls back to
# DEBUG for the rest. A broken credential or a locked database fails EVERY
# artefact, so the failure count equals the fetch count: a real Jira project
# run produced 340 write failures, and one WARNING each would bury the summary
# line (and everything else the operator was reading) under 340 near-identical
# lines. The first few carry all the diagnostic signal — after that the
# per-type summary is what tells you whether it's one fault or several.
_MAX_FAILURE_DETAIL_LOGS = 5


@dataclass
class ConnectorSyncResult:
    """Outcome of one sync() call."""

    artifacts_fetched: int
    artifacts_written: int
    errors: int
    edges_linked: int = 0


def _describe(failures: Counter[str]) -> str:
    """Render a failure tally as `ValueError x338, TimeoutError x2`, worst
    first. The breakdown is the point: it answers "is this one broken
    credential or two unrelated faults?" without reading 340 lines."""
    return ", ".join(f"{name} x{count}" for name, count in failures.most_common())


def _log_item_failure(
    stage: str, *, seen_before: int, source_ref: str, exc: Exception, warn_limit: int
) -> None:
    """Log one per-artefact failure, naming the artefact, the exception type
    and its message — the three facts an `errors: N` count omits.

    The first *warn_limit* failures go to WARNING with a traceback; the rest go
    to DEBUG, so every failing artefact is recoverable by anyone who lowers the
    level without flooding a default-level run (the CLI sets WARNING — note
    `trelix sync` currently has no --verbose flag, so the DEBUG tail is only
    reachable from `serve`/library callers).
    """
    if seen_before < warn_limit:
        logger.warning(
            "Connector sync: %s failed for artefact %s — %s: %s",
            stage,
            source_ref,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
    else:
        logger.debug(
            "Connector sync: %s failed for artefact %s — %s: %s",
            stage,
            source_ref,
            type(exc).__name__,
            exc,
        )


class ArtifactSource(ABC):
    """
    A read-only connector to an external system (Jira, TestRail, ...).

    Implementations own their own HTTP client, auth, and pagination.
    `sync()` is the only method most callers need — `fetch()` exists
    separately so tests can exercise pagination/fetch logic without a real
    Database.
    """

    @abstractmethod
    def validate_config(self) -> None:
        """
        Raise a ValueError with a clear message if required config (base
        URL, credentials) is missing or malformed. Called once before the
        first fetch() so a misconfigured connector fails fast with a
        useful message, not a confusing HTTP 401 several requests in.
        """
        ...

    @abstractmethod
    def fetch(self) -> list[Artifact]:
        """
        Fetch every artefact this connector is configured to pull (paginated
        internally — callers get the fully-materialized list). Real network
        failures should raise; callers decide how to handle that (the CLI
        command catches and reports, it does not swallow silently the way
        GitLinker's git-command failures do, since a connector failure
        means real content is missing, not "this repo just isn't a ticket
        source").
        """
        ...

    def sync(
        self,
        db_writer: ArtifactWriter,
        linker: ArtifactLinker | None = None,
    ) -> ConnectorSyncResult:
        """
        fetch() then persist every artefact via *db_writer* (kept as a thin
        protocol rather than a direct Database import, so connector unit
        tests can pass a mock without needing a real Database instance).

        When *linker* is supplied, each successfully-written artefact is
        immediately passed through ArtifactLinker.link_one() — a synced
        artefact is reachable from generic_edges/the code graph the moment
        this call returns, without a separate `trelix link-artifacts` pass.
        A linking failure for one artefact is non-fatal (best-effort,
        mirrors ArtifactLinker's own "never raise per-artifact" posture) and
        does not count against `errors` (which tracks write failures only).

        Both failure paths log. They used to be bare `except Exception:` with
        nothing bound and nothing emitted, so `errors: 340` was the operator's
        entire diagnostic surface — no exception type, no message, no failing
        source_ref, at any log level. Counting a failure you cannot name is
        indistinguishable from not noticing it.
        """
        self.validate_config()
        fetched = self.fetch()
        written = 0
        errors = 0
        edges_linked = 0
        write_failures: Counter[str] = Counter()
        link_failures: Counter[str] = Counter()
        for artifact in fetched:
            try:
                db_writer.upsert_artifact(artifact)
                written += 1
            except Exception as exc:
                errors += 1
                _log_item_failure(
                    "write",
                    seen_before=sum(write_failures.values()),
                    source_ref=artifact.source_ref,
                    exc=exc,
                    warn_limit=_MAX_FAILURE_DETAIL_LOGS,
                )
                write_failures[type(exc).__name__] += 1
                continue
            if linker is not None:
                try:
                    edges_linked += linker.link_one(artifact.source_ref)
                except Exception as exc:
                    # warn_limit=0: the artefact IS persisted, so a link failure
                    # is a degradation, not a lost write. Per-artefact detail
                    # stays at DEBUG (matching ArtifactLinker's own posture for
                    # its internal fallbacks); the summary below carries it to
                    # WARNING once, so a link outage is visible at one line, not
                    # one line per artefact.
                    _log_item_failure(
                        "link",
                        seen_before=sum(link_failures.values()),
                        source_ref=artifact.source_ref,
                        exc=exc,
                        warn_limit=0,
                    )
                    link_failures[type(exc).__name__] += 1

        # One aggregate line per failing stage, always at WARNING even when the
        # per-item detail was demoted to DEBUG: default CLI logging is not at
        # DEBUG, so this is the line that has to make a partial sync visible.
        if write_failures:
            logger.warning(
                "%s: failed to write %d of %d artefact(s) — %s",
                type(self).__name__,
                sum(write_failures.values()),
                len(fetched),
                _describe(write_failures),
            )
        if link_failures:
            # Non-fatal for the sync, but the consequence is silent and lasting:
            # these artefacts are persisted yet unreachable from the code graph
            # until someone runs `trelix link-artifacts`.
            logger.warning(
                "%s: wrote but could not link %d of %d artefact(s) — %s"
                " (run `trelix link-artifacts` to retry)",
                type(self).__name__,
                sum(link_failures.values()),
                written,
                _describe(link_failures),
            )
        return ConnectorSyncResult(
            artifacts_fetched=len(fetched),
            artifacts_written=written,
            errors=errors,
            edges_linked=edges_linked,
        )


class ArtifactWriter(Protocol):
    """Minimal protocol ArtifactSource.sync() needs — Database satisfies
    this structurally (via matching method signature) without either
    module importing the other."""

    def upsert_artifact(self, artifact: Artifact) -> int: ...
