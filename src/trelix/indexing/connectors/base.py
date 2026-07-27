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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from trelix.core.models import Artifact


@dataclass
class ConnectorSyncResult:
    """Outcome of one sync() call."""

    artifacts_fetched: int
    artifacts_written: int
    errors: int


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

    def sync(self, db_writer: ArtifactWriter) -> ConnectorSyncResult:
        """
        fetch() then persist every artefact via *db_writer* (kept as a thin
        protocol rather than a direct Database import, so connector unit
        tests can pass a mock without needing a real Database instance).
        """
        self.validate_config()
        fetched = self.fetch()
        written = 0
        errors = 0
        for artifact in fetched:
            try:
                db_writer.upsert_artifact(artifact)
                written += 1
            except Exception:
                errors += 1
        return ConnectorSyncResult(
            artifacts_fetched=len(fetched), artifacts_written=written, errors=errors
        )


class ArtifactWriter(Protocol):
    """Minimal protocol ArtifactSource.sync() needs — Database satisfies
    this structurally (via matching method signature) without either
    module importing the other."""

    def upsert_artifact(self, artifact: Artifact) -> int: ...
