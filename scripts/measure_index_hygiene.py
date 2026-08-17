#!/usr/bin/env python3
"""Measure index hygiene: how much of the corpus — and of actual search results — is noise.

Motivation
----------
`FileWalker` only reads the repo-root `.gitignore` (see `_load_gitignore_spec`). Any
directory that a *nested* `.gitignore` excludes is still walked and indexed. In this
repository that meant `workspace-vscode/.vscode-test/` — a 2.6 GB VS Code test harness —
was indexed, and its minified bundles crowded real answers out of the top-10.

Corpus-level pollution is easy to eyeball with a SQL count. What that number does *not*
tell you is the part users feel: how many of the results they actually read are junk. So
this script reports both, and the second one is the one that matters.

Usage
-----
    python scripts/measure_index_hygiene.py <repo> [--provider local] [--k 10]
    python scripts/measure_index_hygiene.py . --json > before.json

Exit codes
----------
    0  measurement completed
    1  index missing, unreadable, or contains no files
    2  bad arguments, including a --provider outside EmbedderConfig's provider Literal
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast, get_args

# Path fragments that should never appear in a healthy index of this repo. Each is a
# directory a nested .gitignore (or an ignore-list gap) was supposed to exclude.
NOISE_PATTERNS: tuple[str, ...] = (
    "/.vscode-test/",
    "/node_modules/",
    "/node_modules.asar",  # VS Code ships an unpacked asar whose dir name is not
    # exactly "node_modules", so a name-equality ignore list misses it entirely.
    "/.venv/",
    "/site-packages/",
    "/scratch-pad/",
    "/.trelix/",
    "/htmlcov/",
    "/dist/",
    "/out/",
)

# Queries chosen to be answerable *only* from this repo's own source. If a query this
# on-topic returns junk, no realistic query is safe.
DEFAULT_QUERIES: tuple[str, ...] = (
    "how does the file walker filter ignored directories",
    "reciprocal rank fusion combining vector and keyword results",
    "how are chunk embeddings stored in sqlite",
    "call graph expansion during retrieval",
    "BM25 scoring implementation",
    "how does the query planner classify retrieval intent",
    "incremental indexing skip unchanged files by hash",
    "tamper evident audit log hash chain",
    "azure openai embedder configuration",
    "tree-sitter parser symbol extraction",
)

# A minified-bundle symbol: one or two characters, or a short consonant cluster with no
# vowels. These are never meaningful search results — they are webpack/esbuild locals.
_MINIFIED = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]?$|^[bcdfghjklmnpqrstvwxz]{2,3}$")


@dataclass
class CorpusStats:
    """Row counts describing how much of the stored corpus is noise."""

    files_total: int
    files_noise: int
    chunks_total: int
    chunks_noise: int
    tokens_total: int
    tokens_noise: int

    @property
    def file_noise_pct(self) -> float:
        return 100.0 * self.files_noise / self.files_total if self.files_total else 0.0

    @property
    def chunk_noise_pct(self) -> float:
        return 100.0 * self.chunks_noise / self.chunks_total if self.chunks_total else 0.0


@dataclass
class QueryResult:
    """Per-query noise contamination within the top-k results."""

    query: str
    k: int
    noise_hits: int
    minified_hits: int
    first_noise_rank: int | None
    top_paths: list[str]

    @property
    def precision_at_k(self) -> float:
        """Fraction of the top-k a user would consider legitimate."""
        return 100.0 * (self.k - self.noise_hits) / self.k if self.k else 0.0


def _is_noise(path: str) -> bool:
    # Normalise so a Windows-style index does not silently score as clean.
    p = path.replace("\\", "/")
    return any(frag in p for frag in NOISE_PATTERNS)


def _noise_sql_predicate(column: str) -> tuple[str, list[str]]:
    """Build an OR-joined LIKE predicate mirroring NOISE_PATTERNS.

    Kept in lockstep with `_is_noise` on purpose: the corpus number and the
    result-set number must agree on what "noise" means, or the two halves of this
    report would not be comparable.
    """
    clauses = " OR ".join(f"{column} LIKE ?" for _ in NOISE_PATTERNS)
    return f"({clauses})", [f"%{frag}%" for frag in NOISE_PATTERNS]


def measure_corpus(db_path: Path) -> CorpusStats:
    """Count noise in the stored corpus straight from SQLite.

    Deliberately avoids loading the sqlite-vec extension: every table touched here is
    ordinary SQL, and requiring vec0 would make the check fail on machines that can
    read the index but cannot load the extension.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        predicate, params = _noise_sql_predicate("f.path")

        files_total = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        if not files_total:
            raise ValueError(f"{db_path} contains no indexed files")

        files_noise = conn.execute(
            f"SELECT COUNT(*) FROM files f WHERE {predicate}", params
        ).fetchone()[0]

        chunks_total, tokens_total = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM chunks"
        ).fetchone()

        # chunks carry no file_id; they reach a file only through symbols.
        chunks_noise, tokens_noise = conn.execute(
            f"""SELECT COUNT(*), COALESCE(SUM(ch.token_count), 0)
                  FROM chunks ch
                  JOIN symbols s ON ch.symbol_id = s.id
                  JOIN files   f ON s.file_id    = f.id
                 WHERE {predicate}""",
            params,
        ).fetchone()

        return CorpusStats(
            files_total=files_total,
            files_noise=files_noise,
            chunks_total=chunks_total,
            chunks_noise=chunks_noise,
            tokens_total=tokens_total,
            tokens_noise=tokens_noise,
        )
    finally:
        conn.close()


def valid_providers() -> tuple[str, ...]:
    """The embedder providers `--provider` accepts, read off `EmbedderConfig` itself.

    Derived via `get_args` rather than restated here so the two can never drift: this is
    the same `Literal` that `create_embedder` (embedder/base.py:647) switches on. Only
    `trelix.core.config` is imported, whose deps (pydantic, pydantic-settings) are core,
    so calling this does not undo `--corpus-only`'s freedom from the retrieval extras.
    """
    from trelix.core.config import EmbedderConfig

    return cast(tuple[str, ...], get_args(EmbedderConfig.model_fields["provider"].annotation))


def measure_queries(
    repo: Path, queries: tuple[str, ...], provider: str, k: int
) -> list[QueryResult]:
    """Run each query through the real retrieval stack and score the top-k.

    Imports are local so that `--corpus-only` works on a machine that can read the
    index but has not installed the retrieval extras.
    """
    from trelix.core.config import IndexConfig
    from trelix.retrieval.retriever import Retriever

    config = IndexConfig(repo_path=str(repo))
    # `EmbedderConfig.provider` is a Literal, but EmbedderConfig sets no
    # validate_assignment, so pydantic does not re-validate on assignment — verified:
    # `c.embedder.provider = 'totally-bogus'` is accepted and reads back verbatim. The
    # only gate is main()'s argparse `choices=valid_providers()`, which rejects the
    # value before this function is ever entered; hence the cast rather than a check
    # here, which main()'s `except Exception` would swallow into a warning anyway.
    config.embedder.provider = cast(Any, provider)
    retriever = Retriever(config)

    results: list[QueryResult] = []
    for query in queries:
        # retrieve() returns a RetrievedContext; the ranked hits live on .results, and
        # each SearchResult reaches its path via .file (an IndexedFile), not a flat field.
        hits = retriever.retrieve(query).results[:k]
        paths = [h.file.path for h in hits]
        symbols = [h.symbol.name for h in hits]

        noise_ranks = [i + 1 for i, p in enumerate(paths) if _is_noise(p)]
        results.append(
            QueryResult(
                query=query,
                k=len(paths),
                noise_hits=len(noise_ranks),
                minified_hits=sum(
                    1 for p, s in zip(paths, symbols) if _is_noise(p) and _MINIFIED.match(s or "")
                ),
                first_noise_rank=noise_ranks[0] if noise_ranks else None,
                top_paths=paths,
            )
        )
    return results


def _render(corpus: CorpusStats, queries: list[QueryResult], k: int) -> None:
    print("=" * 72)
    print("CORPUS HYGIENE")
    print("=" * 72)
    print(
        f"  files   {corpus.files_noise:>7,} / {corpus.files_total:>7,} noise "
        f"({corpus.file_noise_pct:5.1f}%)"
    )
    print(
        f"  chunks  {corpus.chunks_noise:>7,} / {corpus.chunks_total:>7,} noise "
        f"({corpus.chunk_noise_pct:5.1f}%)"
    )
    print(f"  tokens  {corpus.tokens_noise:>7,} / {corpus.tokens_total:>7,} noise")

    if not queries:
        return

    print()
    print("=" * 72)
    print(f"RETRIEVAL HYGIENE (top-{k})")
    print("=" * 72)
    for q in queries:
        flag = "OK " if q.noise_hits == 0 else "BAD"
        first = f"first noise @ rank {q.first_noise_rank}" if q.first_noise_rank else "clean"
        print(f"  [{flag}] {q.noise_hits}/{q.k} noise  {first:<22} {q.query[:38]}")

    mean = sum(q.precision_at_k for q in queries) / len(queries)
    minified = sum(q.minified_hits for q in queries)
    print()
    print(f"  mean precision@{k}: {mean:.1f}%")
    print(f"  minified-bundle symbols surfaced: {minified}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", help="Path to the indexed repository")
    # Not `choices=valid_providers()`: that resolves at parser-construction time and so
    # would import trelix on every invocation, including `--corpus-only`, which is pure
    # sqlite3 and documented to run where trelix is not installed at all (verified: under
    # a trelix-free interpreter the choices version died with ModuleNotFoundError before
    # parsing). Validation happens in the --corpus-only-guarded block below instead.
    ap.add_argument(
        "--provider",
        default="local",
        help="Embedding provider for queries; must be one of EmbedderConfig's providers",
    )
    ap.add_argument("--k", type=int, default=10, help="Top-k window to score")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    ap.add_argument(
        "--corpus-only",
        action="store_true",
        help="Skip live queries; report only SQL-level corpus counts",
    )
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    db_path = repo / ".trelix" / "index.db"
    if not db_path.exists():
        print(f"error: no index at {db_path} — run `trelix index {repo}` first", file=sys.stderr)
        return 1

    try:
        corpus = measure_corpus(db_path)
    except (sqlite3.Error, ValueError) as exc:
        print(f"error: cannot measure {db_path}: {exc}", file=sys.stderr)
        return 1

    queries: list[QueryResult] = []
    if not args.corpus_only:
        # Deliberately OUTSIDE the try below. A typo'd provider used to reach
        # create_embedder, raise ValueError, get caught as a "warning" and leave the run
        # exiting 0 with `"queries": []` and the bad provider echoed back as fact —
        # a before/after diff would show the retrieval half simply missing, with no
        # failure to attribute it to. Checking here makes it exit 2 instead.
        allowed = valid_providers()
        if args.provider not in allowed:
            print(
                f"error: --provider {args.provider!r} is not one of: {', '.join(allowed)}",
                file=sys.stderr,
            )
            return 2
        try:
            queries = measure_queries(repo, DEFAULT_QUERIES, args.provider, args.k)
        except Exception as exc:  # noqa: BLE001 - surface any retrieval failure verbatim
            print(
                f"warning: query measurement failed ({exc}); reporting corpus only", file=sys.stderr
            )

    if args.json:
        payload: dict[str, Any] = {
            "repo": str(repo),
            "provider": args.provider,
            "k": args.k,
            "corpus": asdict(corpus),
            "corpus_file_noise_pct": round(corpus.file_noise_pct, 2),
            "corpus_chunk_noise_pct": round(corpus.chunk_noise_pct, 2),
            "queries": [asdict(q) for q in queries],
        }
        if queries:
            payload["mean_precision_at_k"] = round(
                sum(q.precision_at_k for q in queries) / len(queries), 2
            )
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        _render(corpus, queries, args.k)

    return 0


if __name__ == "__main__":
    sys.exit(main())
