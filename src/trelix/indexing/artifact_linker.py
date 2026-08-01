"""
Artifact linker — scans connector-fetched artifacts (Jira tickets, TestRail
cases, ...) for mentions of indexed symbols, feeding trelix's generic_edges
cross-source graph.

`trelix connector sync` (src/trelix/indexing/connectors/base.py's
ArtifactSource.sync()) only writes to the `artifacts` table — it never
touches generic_edges. GitLinker (git_linker.py) is the only existing writer
of generic_edges, and only for git-commit-message ticket references, not for
the artifact content connectors actually fetch. This module closes that gap:
an artifact sitting in the `artifacts` table has zero effect on search or
PageRank until it's been through ArtifactLinker.link().

Two-stage matching, in order:
  1. Regex reference-extraction (free, deterministic) — mirrors
     GitLinker._extract_ticket_ids()'s compile-a-regex -> findall -> dedupe
     shape, but matches against real symbol names/qualified_names instead of
     a ticket-ID pattern.
  2. Embedding-similarity fallback (opt-in, ArtifactLinkerConfig.
     embedding_fallback_enabled) — only for artifacts where stage 1 found
     zero matches. Lower-confidence matches get weight=0.5 (vs. a regex
     hit's weight=1.0) so they don't dominate PageRank mass — the exact
     failure mode the EasyLink research flags with pure-embedding matching.
"""

from __future__ import annotations

import logging
import re

from trelix.core.config import ArtifactLinkerConfig, IndexConfig
from trelix.core.models import Artifact, GenericEdge
from trelix.store.db import Database

logger = logging.getLogger("trelix.indexing.artifact_linker")

# Identifier-shaped tokens: mirrors typical function/class/variable naming
# (word chars, dots for qualified-name segments) — a superset of what any
# indexed symbol's name/qualified_name could look like.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{2,}")

# Bare (non-qualified) symbol names that collide with ordinary English/
# generic-programming vocabulary are excluded from the bare-name index —
# a ticket titled "the test suite failed to run and update the process"
# would otherwise match `run`/`test`/`process`/`update` purely on English
# overlap with zero relation to those symbols. Matching via the more
# specific qualified_name (e.g. "auth.login" instead of bare "login") is
# unaffected — this list only gates the bare-name key.
_COMMON_WORD_STOPLIST = frozenset(
    {
        "get",
        "set",
        "run",
        "test",
        "tests",
        "process",
        "update",
        "data",
        "file",
        "files",
        "main",
        "init",
        "new",
        "list",
        "map",
        "filter",
        "type",
        "value",
        "item",
        "items",
        "check",
        "load",
        "save",
        "start",
        "stop",
        "add",
        "remove",
        "delete",
        "create",
        "build",
        "parse",
        "format",
        "print",
        "log",
        "error",
        "result",
        "results",
        "config",
        "state",
        "index",
        "name",
        "id",
    }
)


class ArtifactLinker:
    """
    Scans artifacts in the `artifacts` table for symbol-name mentions and
    inserts one GenericEdge per (symbol, artifact) match found.

    Never raises on a per-artifact basis — an embedding-fallback failure
    (e.g. no embedder configured) degrades that one artifact to "no link
    found", matching GitLinker's "never raise" posture.
    """

    def __init__(
        self,
        db: Database,
        config: ArtifactLinkerConfig | None = None,
        index_config: IndexConfig | None = None,
    ) -> None:
        self._db = db
        self._config = config or ArtifactLinkerConfig()
        # Needed only for the embedding fallback (make_embedder/
        # make_vector_store both key off IndexConfig) — regex matching never
        # touches this, so callers that only want regex linking can omit it.
        self._index_config = index_config
        # Lazily built, memoized on this instance — see _get_name_index()'s
        # docstring for why this matters for link_one()'s call pattern.
        self._name_index: dict[str, int] | None = None

    def link(self) -> int:
        """
        Run the full scan-artifacts -> match-symbols -> insert pipeline over
        every artifact in the table. Returns the number of GenericEdges
        inserted (0 if there are no artifacts or no matches).
        """
        artifacts = self._db.get_all_artifacts()
        if not artifacts:
            return 0

        name_to_id = self._get_name_index()
        edges: list[GenericEdge] = []
        for artifact in artifacts:
            edges.extend(self._match_artifact(artifact, name_to_id))

        if edges:
            self._db.insert_generic_edges(edges)
        logger.info(
            "ArtifactLinker: linked %d symbol-artifact edges from %d artifacts",
            len(edges),
            len(artifacts),
        )
        return len(edges)

    def link_one(self, source_ref: str) -> int:
        """
        Link a single already-persisted artifact by source_ref — used by
        ArtifactSource.sync()'s auto-link post-step so a fresh sync doesn't
        need a full table re-scan.
        """
        artifact = self._db.get_artifact_by_source_ref(source_ref)
        if artifact is None:
            return 0

        name_to_id = self._get_name_index()
        edges = self._match_artifact(artifact, name_to_id)
        if edges:
            self._db.insert_generic_edges(edges)
        return len(edges)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_name_index(self) -> dict[str, int]:
        """Build the name index once per ArtifactLinker instance, then
        reuse it. ArtifactSource.sync() (connectors/base.py) constructs one
        ArtifactLinker and calls link_one() once per synced artifact in a
        loop — without this memoization, each of those calls would rebuild
        the full symbol-name index from scratch (_build_name_index()'s own
        docstring promises an O(symbols) cost "per call, not per
        artifact", but that promise only held for link()'s single internal
        call; link_one() called N times reintroduced exactly the O(artifacts
        * symbols) scan the docstring says this design avoids). A caller
        that mutates symbols between link_one() calls on the same instance
        (e.g. a concurrent re-index) should construct a fresh ArtifactLinker
        rather than reuse a stale one — this class was never safe against
        that anyway, since Database itself isn't safe for concurrent writes
        from multiple connections.
        """
        if self._name_index is None:
            self._name_index = self._build_name_index()
        return self._name_index

    def _build_name_index(self) -> dict[str, int]:
        """casefold(name)/casefold(qualified_name) -> symbol_id, built once
        per call (not per artifact) to avoid an O(artifacts * symbols)
        scan.

        Keys are casefolded so a title-cased mention ("Login is broken")
        matches a lowercase symbol name — real prose capitalizes the first
        word of a sentence/title regardless of the symbol's actual casing.

        Any candidate key that collides with _COMMON_WORD_STOPLIST (e.g. a
        function named `run` or `process`) is skipped — matching those
        against arbitrary English prose produces edges with no real
        relationship to the ticket. This applies to `qualified_name` too,
        not just bare `name`: many symbols (anything without an enclosing
        module/class prefix) have qualified_name == name, so gating only
        on the `name` field would let the exact same stop-worthy string
        back in through the "qualified" key. A genuinely-qualified name
        like "auth.run" is unaffected — only the literal stoplisted string
        itself is excluded, not names that merely contain it as a
        substring.
        """
        name_to_id: dict[str, int] = {}
        for symbol_id, name, qualified_name in self._db.get_all_symbol_names():
            for candidate in (name.casefold(), qualified_name.casefold()):
                if candidate not in _COMMON_WORD_STOPLIST:
                    name_to_id.setdefault(candidate, symbol_id)
        return name_to_id

    def _match_artifact(self, artifact: Artifact, name_to_id: dict[str, int]) -> list[GenericEdge]:
        matched_ids = self._regex_match(f"{artifact.title} {artifact.body}", name_to_id)
        if matched_ids:
            return [
                GenericEdge(
                    from_symbol_id=symbol_id,
                    source_ref=artifact.source_ref,
                    edge_kind="references_artifact",
                    weight=1.0,
                )
                for symbol_id in matched_ids
            ]
        if not self._config.embedding_fallback_enabled:
            return []
        return [
            GenericEdge(
                from_symbol_id=symbol_id,
                source_ref=artifact.source_ref,
                edge_kind="references_artifact",
                weight=0.5,
            )
            for symbol_id in self._embedding_match(artifact)
        ]

    def _regex_match(self, text: str, name_to_id: dict[str, int]) -> list[int]:
        """De-duplicated, order-preserving list of symbol_ids whose
        name/qualified_name appears as an identifier-shaped token in *text*.

        _IDENTIFIER_RE's character class includes '.' (needed to capture
        qualified-name segments like "auth.login"), so it also greedily
        swallows a trailing sentence-ending period ("Cannot access
        login." -> "login.") — stripped here before lookup rather than
        tightened in the regex itself, since a qualified name can
        legitimately end right before a period with no way to distinguish
        the two cases from the regex alone."""
        matched: dict[int, None] = {}
        for raw_token in _IDENTIFIER_RE.findall(text):
            token = raw_token.rstrip(".").casefold()
            symbol_id = name_to_id.get(token)
            if symbol_id is not None:
                matched.setdefault(symbol_id, None)
        return list(matched)

    def _embedding_match(self, artifact: Artifact) -> list[int]:
        """Embed the artifact's title+body and search the configured vector
        store for similar chunks, keeping hits above the configured
        similarity floor. Never raises — any failure (no embedder
        configured, dimension mismatch, ...) degrades to "no matches"."""
        if self._index_config is None:
            logger.debug("ArtifactLinker: no IndexConfig — skipping embedding fallback")
            return []
        try:
            from trelix.embedder.base import make_embedder
            from trelix.store.vector import make_vector_store

            embedder = make_embedder(self._index_config.embedder)
            vector_store = make_vector_store(
                config=self._index_config, dimension=embedder.dimension
            )

            text = f"{artifact.title} {artifact.body}"[:2000]
            query_embedding = embedder.embed_query(text)
            raw = vector_store.search(query_embedding, k=5)

            matched: dict[int, None] = {}
            for chunk_id, distance in raw:
                similarity = max(0.0, 1.0 - distance)
                if similarity < self._config.similarity_threshold:
                    continue
                row = self._db.get_chunk_with_context(chunk_id)
                if row is None:
                    continue
                symbol = row[1]
                if symbol.id is not None:
                    matched.setdefault(symbol.id, None)
            return list(matched)
        except Exception as exc:
            logger.debug("ArtifactLinker: embedding fallback failed (non-fatal): %s", exc)
            return []
