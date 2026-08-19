"""
SQLite store: all structured metadata lives here.

Schema design:
- files       → one row per indexed file, with hash for incremental updates
- symbols     → one row per extracted symbol (function, class, method, etc.)
- calls       → directed call graph edges
- imports     → file-level import edges
- chunks      → embeddable text chunks (1:1 with symbols for now)
- symbols_fts → FTS5 virtual table for BM25 keyword search

FTS5 is built into SQLite — zero extra dependencies, fast, good enough for
repos up to millions of lines. (Stolen from ctags-based tools.)

Schema generation: stamped in `pragma user_version` (SCHEMA_VERSION below) and
checked on every open, so an older trelix refuses a newer index instead of
misreading it. See SCHEMA_VERSION for when a bump is warranted.

Referential integrity: `PRAGMA foreign_keys = ON` is in the DDL and verified to be in
effect (tests/unit/test_db_structural.py), so every ON DELETE CASCADE here is real. Three
tables predate that discipline and hold symbol-derived rows with no foreign key —
sub_chunks, def_use_edges, sparse_embeddings — so no cascade reaches them. All three
are cleaned by _purge_fkless_symbol_rows(), which every symbol-removal path calls;
add any new FK-less table there. taint_flows / artifacts / diff_chunks are also
FK-less but are keyed by file path, source_ref and pr_ref rather than by a row id,
so they hold no id-shaped orphans and are not this method's business.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trelix.indexing.multi_granularity import SubSymbolChunk
    from trelix.store.read_pool import ReadOnlyConnectionPool

from trelix.core.models import (
    Artifact,
    CallEdge,
    Chunk,
    GenericEdge,
    ImportEdge,
    IndexedFile,
    Language,
    Symbol,
    SymbolKind,
    TypeEdge,
)

if TYPE_CHECKING:
    from trelix.analysis.defuse import DefUseEdge
    from trelix.analysis.taint import TaintFlow

logger = logging.getLogger("trelix.store.db")

# Schema generation stamped into `pragma user_version`. Before this existed the
# live index read 0 across 30 user tables built by an unversioned linear chain of
# ~17 `CREATE TABLE IF NOT EXISTS` / `ALTER TABLE` blocks whose comments name
# v2.2/v2.3/v2.4/v2.8 with nothing queryable — so an older install opened a
# newer-written index in silence, with DimensionGuard (embedding width only) the
# sole thing standing between that and wrong answers.
#
# The guard is forward-looking by construction: an install that predates this
# constant does not read user_version at all, so it still cannot refuse anything.
# What version 1 buys is that every index written from here on is identifiable.
#
# When to bump: ONLY for a change an older reader would MISREAD — a column whose
# meaning changed, a table an older reader would read with different semantics, a
# dropped/renamed column. Purely additive changes (new table, new nullable column)
# stay backward-readable and must NOT bump, because a bump locks every older
# install out of the index. That is the contract docs/BACKWARDS_COMPATIBILITY.md
# already promises for `query_telemetry`, now enforceable rather than aspirational.
#
# Version 1 is the first stamped generation: it is schema-identical to what
# unstamped (user_version = 0) indexes already hold. 0 therefore means
# "written before versioning", NOT corrupt — see _guard_schema_version().
SCHEMA_VERSION = 1


class SchemaVersionError(Exception):
    """Raised when the index on disk was written by a newer trelix than this one.

    Refusing beats misreading: the alternative is an older reader applying its own
    idea of a column's meaning to rows written under different semantics, which
    surfaces as wrong search results rather than an error.
    """

    def __init__(self, db_path: Path, found: int, supported: int) -> None:
        self.found = found
        self.supported = supported
        super().__init__(
            f"Index at {db_path} has schema version {found}, but this trelix "
            f"only understands up to {supported}.\n\n"
            f"The index was written by a newer trelix. Reading it with this one "
            f"could silently return wrong results, so it is refused.\n\n"
            f"Fix, either one:\n"
            f"  pip install --upgrade trelix          # match the version that wrote it\n"
            f"  rm {db_path}* && trelix index <repo>  # rebuild with this version\n\n"
            f"(The rm glob also removes the -wal/-shm sidecars; `trelix index` has no "
            f"force flag, so removing the file is how a rebuild from scratch is asked for.)"
        )


DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    rel_path    TEXT    NOT NULL,
    language    TEXT    NOT NULL,
    hash        TEXT    NOT NULL,
    size_bytes  INTEGER NOT NULL,
    indexed_at  TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_files_rel_path ON files(rel_path);

CREATE TABLE IF NOT EXISTS symbols (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id         INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name            TEXT    NOT NULL,
    qualified_name  TEXT    NOT NULL,
    kind            TEXT    NOT NULL,
    line_start      INTEGER NOT NULL,
    line_end        INTEGER NOT NULL,
    signature       TEXT    NOT NULL DEFAULT '',
    docstring       TEXT,
    context_summary TEXT,
    decorators      TEXT    NOT NULL DEFAULT '[]',
    is_public       INTEGER NOT NULL DEFAULT 1,
    parent_id       INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    body            TEXT    NOT NULL DEFAULT '',
    content_hash    TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS type_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    from_symbol_id  INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    to_type_name    TEXT    NOT NULL,
    edge_kind       TEXT    NOT NULL,
    to_symbol_id    INTEGER REFERENCES symbols(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_type_edges_from ON type_edges(from_symbol_id);
CREATE INDEX IF NOT EXISTS idx_type_edges_to   ON type_edges(to_symbol_id);

CREATE INDEX IF NOT EXISTS idx_symbols_file_id   ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_symbols_name       ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_kind       ON symbols(kind);

CREATE TABLE IF NOT EXISTS calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    caller_id         INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    callee_name       TEXT    NOT NULL,
    callee_id         INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    line              INTEGER NOT NULL,
    callee_type_hint  TEXT                          -- receiver static type, e.g. "UserService"
);

CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_id);
CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee_id);

CREATE TABLE IF NOT EXISTS imports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id         INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    imported_from   TEXT    NOT NULL,
    imported_names  TEXT    NOT NULL DEFAULT '[]'  -- JSON array
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id   INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    chunk_text  TEXT    NOT NULL,
    token_count INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_symbol_id ON chunks(symbol_id);

CREATE TABLE IF NOT EXISTS sub_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_symbol_id INTEGER NOT NULL,
    granularity TEXT NOT NULL CHECK(granularity IN ('function','block','statement')),
    chunk_text TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sub_chunks_symbol ON sub_chunks(parent_symbol_id);

CREATE TABLE IF NOT EXISTS file_summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL UNIQUE REFERENCES files(id) ON DELETE CASCADE,
    summary     TEXT    NOT NULL,
    chunk_id    INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_file_summaries_file_id ON file_summaries(file_id);

CREATE TABLE IF NOT EXISTS diff_chunks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    pr_ref           TEXT    NOT NULL,
    hunk_header      TEXT    NOT NULL DEFAULT '',
    before_code      TEXT    NOT NULL DEFAULT '',
    after_code       TEXT    NOT NULL DEFAULT '',
    embedding        BLOB,
    chunk_char_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_diff_chunks_pr_ref ON diff_chunks(pr_ref);

-- FTS5 for BM25 keyword search over symbol content
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name,
    qualified_name,
    docstring,
    body,
    context_summary,
    content='symbols',
    content_rowid='id',
    tokenize='porter ascii'
);

-- Keep FTS index in sync with symbols table
-- NOTE: triggers reference the 5-column FTS schema; decorators are not FTS-indexed
-- (they appear in chunk_text so vector + BM25 search already covers them)
CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbols_fts(rowid, name, qualified_name, docstring, body, context_summary)
    VALUES (new.id, new.name, new.qualified_name, new.docstring, new.body, new.context_summary);
END;

CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbols_fts(
        symbols_fts, rowid, name, qualified_name, docstring, body, context_summary
    ) VALUES (
        'delete', old.id, old.name, old.qualified_name,
        old.docstring, old.body, old.context_summary
    );
END;

CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
    INSERT INTO symbols_fts(
        symbols_fts, rowid, name, qualified_name, docstring, body, context_summary
    ) VALUES (
        'delete', old.id, old.name, old.qualified_name,
        old.docstring, old.body, old.context_summary
    );
    INSERT INTO symbols_fts(rowid, name, qualified_name, docstring, body, context_summary)
    VALUES (
        new.id, new.name, new.qualified_name,
        new.docstring, new.body, new.context_summary
    );
END;
"""


def read_only_uri(db_path: Path | str) -> str:
    """Build the `mode=ro` SQLite URI for `db_path`, percent-encoding the path.

    A path is not a URI. Interpolating one raw — `f"file:{db_path}?mode=ro"` — hands the
    path's own metacharacters to SQLite's URI parser: `#` ends the filename as a fragment
    and `?` opens a query section of its own, so `mode=ro` is discarded and SQLite opens
    the *truncated* path in the default read-write mode, creating an empty database there
    if none exists. The result is the exact inverse of what the caller asked for: a handle
    that accepts writes and cannot see the index's tables. `trelix index --dry-run
    --prune` hit both halves — it left a stray database in the parent directory of any
    index whose absolute path contained `#`, from a command whose promise is that it
    changes nothing.

    `pathname2url` rather than `Path.as_uri()`: `as_uri()` raises on a relative path, and
    the documented constructor call is `Database(Path(".trelix/index.db"))`. SQLite accepts
    a relative URI filename, so encoding in place keeps both path kinds working.
    """
    return f"file:{urllib.request.pathname2url(str(db_path))}?mode=ro"


class Database:
    """
    Thin wrapper around sqlite3 with typed methods for each table.

    Usage:
        db = Database(Path(".trelix/index.db"))
        file_id = db.upsert_file(indexed_file)
        symbol_id = db.insert_symbol(symbol)
        db.close()
    """

    def __init__(self, db_path: Path, *, read_only: bool = False) -> None:
        """Open the index. `read_only=True` opens it without being able to write to it.

        The default path runs `init_schema()`, which is a WRITE — it applies DDL, runs
        migrations and stamps `user_version`. That is correct for every normal caller and
        wrong for exactly one: `trelix index --dry-run`, whose entire promise is that it
        changes nothing. Before this flag existed, `--dry-run --prune` opened the index
        normally and committed a `DELETE` plus a `PRAGMA user_version` write on any
        pre-3.1.2 index, while the cost-preview path twenty lines away deliberately used a
        `mode=ro` URI "because a cost preview must not be able to modify the index it
        prices". Same promise, two answers.

        Enforced by SQLite rather than by discipline: a `mode=ro` URI makes a stray write
        raise `sqlite3.OperationalError` instead of silently succeeding, so a future caller
        cannot quietly reintroduce the write. Schema init is skipped, which is why this must
        not be used for anything that needs a current schema — reads that touch a table a
        migration would have added will fail, loudly, which is the right outcome.
        """
        self._read_only = read_only
        self._db_path = db_path

        # Declared once, above the branch, because there are now two constructor paths.
        # Annotating inside either is not enough — mypy reads a branch-local annotation as
        # narrowing rather than as the attribute's declaration, so the other path becomes a
        # [no-redef] and enable_bm25_read_pool's assignment becomes an [assignment] error.
        # CI caught both: the read-only path was added without re-running mypy.
        self._bm25_read_pool: ReadOnlyConnectionPool | None = None
        # Guards concurrent access to the single shared writer connection
        # (self._conn) from hydration calls that bm25_search()'s retrieval-layer
        # wrapper makes after drawing symbol_ids from the (thread-safe) read
        # pool — sqlite3.Connection is not safe for concurrent statement
        # execution from multiple threads even with check_same_thread=False.
        self._conn_lock = threading.Lock()

        if read_only:
            self._conn = sqlite3.connect(read_only_uri(db_path), uri=True, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            return

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.init_schema()

    def enable_bm25_read_pool(self, pool_size: int) -> None:
        """Opt-in: open a ReadOnlyConnectionPool for bm25_search() to draw
        from instead of the single shared writer connection. No-op (and
        disables an existing pool) if pool_size <= 0.
        """
        if pool_size <= 0:
            self._bm25_read_pool = None
            return
        from trelix.store.read_pool import ReadOnlyConnectionPool

        self._bm25_read_pool = ReadOnlyConnectionPool(self._db_path, pool_size=pool_size)

    def init_schema(self) -> None:
        """Initialize or refresh the database schema and apply all migrations.

        Safe to call multiple times — uses IF NOT EXISTS guards throughout.

        Refuses (SchemaVersionError) an index stamped NEWER than SCHEMA_VERSION
        before touching it, so an older reader never half-migrates a schema it
        does not understand.
        """
        on_disk = self.schema_version()
        self._guard_schema_version(on_disk)
        self._apply_ddl()
        self._apply_migrations()
        if on_disk < SCHEMA_VERSION:
            self._upgrade_to_current(on_disk)

    def schema_version(self) -> int:
        """Schema generation stamped in `pragma user_version`.

        0 on any index written before versioning existed (which is every index
        in the wild at the time this shipped) and on a freshly-created file, since
        SQLite defaults user_version to 0 — the two are indistinguishable and both
        are handled by _upgrade_to_current().
        """
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def _guard_schema_version(self, on_disk: int) -> None:
        if on_disk > SCHEMA_VERSION:
            raise SchemaVersionError(self._db_path, found=on_disk, supported=SCHEMA_VERSION)

    def _upgrade_to_current(self, on_disk: int) -> None:
        """One-time work for an index stamped below SCHEMA_VERSION, then stamp it.

        Gated on the stamp rather than run on every open because the def-use reclaim
        is an anti-join over the largest table in the index. Measured on a copy of the
        live 192 MB index: 28.4 ms for the upgrading open (4,768 rows reclaimed),
        1.2 ms for every open after it.
        """
        logger.debug("Upgrading index schema %d -> %d", on_disk, SCHEMA_VERSION)
        self._reclaim_orphaned_def_use_edges()
        # PRAGMA takes no bound parameters, so the value is interpolated. It is a
        # module-level int constant, never user input — the parameterized-query
        # rule is about untrusted values and there is none here.
        self._conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
        self._conn.commit()

    def _reclaim_orphaned_def_use_edges(self) -> int:
        """Delete def_use_edges rows whose symbol is already gone. Returns the count.

        These are the rows leaked before _purge_def_use_edges() existed: 4,768 of
        117,814 (4.0%) on the live index. Nothing else would ever reclaim them —
        the table has no foreign key, so no cascade reaches it, and get_data_flows()
        is keyed by symbol_id so an orphan is unreachable as well as dead.
        """
        cursor = self._conn.execute(
            "DELETE FROM def_use_edges WHERE symbol_id NOT IN (SELECT id FROM symbols)"
        )
        self._conn.commit()
        reclaimed = cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
        if reclaimed:
            logger.info(
                "Reclaimed %d orphaned def_use_edges rows left by pre-%s indexing",
                reclaimed,
                SCHEMA_VERSION,
            )
        return reclaimed

    def _apply_ddl(self) -> None:
        self._conn.executescript(DDL)
        self._conn.commit()

    def _apply_migrations(self) -> None:
        """Incremental schema migrations — safe to run on existing DBs."""
        # Task 2 migration: add idx_files_rel_path for watch performance (Phase 1)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_files_rel_path ON files(rel_path)")
        self._conn.commit()

        import_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(imports)").fetchall()}
        if "imported_file_id" not in import_cols:
            self._conn.execute(
                "ALTER TABLE imports ADD COLUMN imported_file_id INTEGER REFERENCES files(id)"
            )
            self._conn.commit()

        sym_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(symbols)").fetchall()}
        if "decorators" not in sym_cols:
            self._conn.execute(
                "ALTER TABLE symbols ADD COLUMN decorators TEXT NOT NULL DEFAULT '[]'"
            )
            self._conn.commit()
        if "is_public" not in sym_cols:
            self._conn.execute(
                "ALTER TABLE symbols ADD COLUMN is_public INTEGER NOT NULL DEFAULT 1"
            )
            self._conn.commit()
        if "context_summary" not in sym_cols:
            self._conn.execute("ALTER TABLE symbols ADD COLUMN context_summary TEXT")
            self._conn.commit()
        if "content_hash" not in sym_cols:
            self._conn.execute(
                "ALTER TABLE symbols ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()

        # calls.callee_type_hint — added in U9 for qualified-name + type-hint resolution
        calls_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(calls)").fetchall()}
        if "callee_type_hint" not in calls_cols:
            self._conn.execute("ALTER TABLE calls ADD COLUMN callee_type_hint TEXT")
            self._conn.commit()

        # type_edges table is idempotent (CREATE TABLE IF NOT EXISTS in DDL)

        # Phase 2 migration: add file_summaries if not present
        # IF NOT EXISTS is safe on existing DBs — no-op when table already exists
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS file_summaries ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "file_id INTEGER NOT NULL UNIQUE REFERENCES files(id) ON DELETE CASCADE, "
            "summary TEXT NOT NULL, chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL, "
            "created_at TEXT DEFAULT (datetime('now')))"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_summaries_file_id ON file_summaries(file_id)"
        )
        self._conn.commit()

        # MGS3 migration: add sub_chunks table for multi-granularity indexing
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sub_chunks ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "parent_symbol_id INTEGER NOT NULL, "
            "granularity TEXT NOT NULL CHECK(granularity IN ('function','block','statement')), "
            "chunk_text TEXT NOT NULL, "
            "line_start INTEGER NOT NULL, "
            "line_end INTEGER NOT NULL, "
            "token_count INTEGER NOT NULL DEFAULT 0)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sub_chunks_symbol ON sub_chunks(parent_symbol_id)"
        )
        self._conn.commit()

        # Task 6 migration: add query_telemetry table for observability
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS query_telemetry ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT NOT NULL DEFAULT (datetime('now')), "
            "query TEXT NOT NULL, "
            "intent TEXT DEFAULT '', "
            "elapsed_ms REAL DEFAULT 0.0, "
            "result_count INTEGER DEFAULT 0, "
            "leg_sizes TEXT DEFAULT '{}', "
            "thumbs_up INTEGER DEFAULT NULL"
            ")"
        )
        self._conn.commit()

        # v2.4 migration: add expansion observability columns (idempotent)
        for col_def in [
            "expansion_used INTEGER DEFAULT NULL",
            "expansion_variants INTEGER DEFAULT NULL",
            "expansion_elapsed_ms REAL DEFAULT NULL",
        ]:
            col_name = col_def.split()[0]
            existing = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(query_telemetry)").fetchall()
            }
            if col_name not in existing:
                self._conn.execute(f"ALTER TABLE query_telemetry ADD COLUMN {col_def}")
        self._conn.commit()

        # v2.3 Plan E migration: index_metadata table for embedding dimension guard
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

        # v2.2 migration: def-use chains (data-flow analysis)
        # symbol_id deliberately carries NO `REFERENCES symbols(id)`: the constraint
        # cannot be ALTERed into an existing table, and every install already holds
        # rows here (117,814 on the live index) — some of them orphaned, which would
        # make the required table rebuild fail outright under `foreign_keys = ON`.
        # Cleanup is explicit instead; see _purge_def_use_edges(), which every
        # symbol-removal path calls, and _reclaim_orphaned_def_use_edges() for the
        # one-time sweep of what the FK-less years left behind.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS def_use_edges ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "symbol_id INTEGER NOT NULL, "
            "var_name TEXT NOT NULL, "
            "def_line INTEGER NOT NULL, "
            "use_line INTEGER NOT NULL, "
            "edge_type TEXT NOT NULL CHECK(edge_type IN ('def', 'use'))"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_def_use_symbol ON def_use_edges(symbol_id)"
        )
        self._conn.commit()

        # v2.2 migration: taint flows (semgrep integration)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS taint_flows ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "source_file TEXT NOT NULL, "
            "source_line INTEGER NOT NULL, "
            "sink_file TEXT NOT NULL, "
            "sink_line INTEGER NOT NULL, "
            "rule_id TEXT NOT NULL, "
            "severity TEXT NOT NULL DEFAULT 'INFO'"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_taint_severity ON taint_flows(severity)")
        self._conn.commit()

        # v2.2 migration: sparse_embeddings inverted index for SPLADE-Code
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sparse_embeddings (
                chunk_id INTEGER NOT NULL,
                token_id INTEGER NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (chunk_id, token_id)
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sparse_token ON sparse_embeddings(token_id)"
        )
        self._conn.commit()

        # Phase 2 Plan B migration: diff_chunks table for semantic diff embeddings
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS diff_chunks ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "pr_ref TEXT NOT NULL, "
            "hunk_header TEXT NOT NULL DEFAULT '', "
            "before_code TEXT NOT NULL DEFAULT '', "
            "after_code TEXT NOT NULL DEFAULT '', "
            "embedding BLOB, "
            "chunk_char_count INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_diff_chunks_pr_ref ON diff_chunks(pr_ref)"
        )
        self._conn.commit()

        # v2.8 migration: agent_sessions + agent_turns for persistent ReAct history
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_sessions ("
            "id TEXT PRIMARY KEY, "
            "created_at TEXT NOT NULL DEFAULT (datetime('now')), "
            "last_active_at TEXT NOT NULL DEFAULT (datetime('now')), "
            "query TEXT NOT NULL DEFAULT '', "
            "turn_count INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_sessions_last_active "
            "ON agent_sessions(last_active_at)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_turns ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE, "
            "turn_index INTEGER NOT NULL, "
            "thought TEXT NOT NULL DEFAULT '', "
            "action_type TEXT NOT NULL, "
            "action_arguments TEXT NOT NULL DEFAULT '{}', "
            "observation_content TEXT NOT NULL DEFAULT '', "
            "observation_source TEXT NOT NULL DEFAULT '', "
            "observation_success INTEGER NOT NULL DEFAULT 1, "
            "created_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_turns_session "
            "ON agent_turns(session_id, turn_index)"
        )
        self._conn.commit()

        # Cross-source edges migration: generic typed edges linking a code
        # symbol to a non-code artefact (ticket, test, doc, ...). Exclusive
        # arc — from_symbol_id is a real FK (the code side is always a real
        # symbol for every writer today, e.g. the git-log ticket linker), and
        # source_ref is the non-symbol side as a free-form "<type>:<ref>"
        # string (e.g. "ticket:PROJ-123"), since a generic non-code artefact
        # has no id in this DB to reference. Nullable on both column groups
        # so the schema doesn't have to change again if a future writer ever
        # needs an edge between two non-code artefacts.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS generic_edges ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "from_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE, "
            "source_ref TEXT, "
            "edge_kind TEXT NOT NULL, "
            "weight REAL NOT NULL DEFAULT 1.0"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_generic_edges_from ON generic_edges(from_symbol_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_generic_edges_source_ref ON generic_edges(source_ref)"
        )
        # Re-running a writer (e.g. `trelix link-tickets` on the same repo)
        # must not duplicate an edge it already inserted.
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_generic_edges_dedup "
            "ON generic_edges(from_symbol_id, source_ref, edge_kind)"
        )
        self._conn.commit()

        # Source-connector artifacts migration: non-code content fetched
        # from Jira/TestRail (tickets, test cases). Joined to code purely
        # via generic_edges.source_ref — see Artifact's docstring in
        # core/models.py for why this is a new table, not a Chunk.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS artifacts ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "source_ref TEXT NOT NULL UNIQUE, "
            "artifact_kind TEXT NOT NULL, "
            "title TEXT NOT NULL DEFAULT '', "
            "body TEXT NOT NULL DEFAULT '', "
            "url TEXT, "
            "metadata TEXT NOT NULL DEFAULT '{}', "
            "fetched_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(artifact_kind)"
        )
        self._conn.commit()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def invalidate_all_symbol_hashes(self) -> int:
        """Blank every stored symbol content hash so the next index re-embeds.

        Invalidating file hashes alone only forces a re-PARSE. `Indexer._insert_one`
        then diffs each parsed symbol against the stored one by
        `qualified_name + content_hash` and leaves matches completely untouched — "no
        delete, no re-insert, no re-embed" — so a re-index after clearing embeddings
        produced `chunks_embedded=0` and the vectors never came back.

        Both levels have to be invalidated for a reset to mean anything. Returns the
        number of rows affected.
        """
        cursor = self._conn.execute("UPDATE symbols SET content_hash = ''")
        self._conn.commit()
        return int(cursor.rowcount or 0)

    def invalidate_all_file_hashes(self) -> int:
        """Blank every stored file hash so the next index re-parses and re-embeds.

        `Indexer.index()` skips any file whose stored hash still matches the one on
        disk, and that check knows nothing about whether the file's VECTORS still exist.
        So after embeddings are discarded, an index run finds every hash unchanged and
        prints "Nothing to index — all files up to date" over an index that has chunks
        and no vectors.

        Returns the number of rows invalidated, so a caller can report it rather than
        assert it.
        """
        cursor = self._conn.execute("UPDATE files SET hash = ''")
        self._conn.commit()
        return int(cursor.rowcount or 0)

    def get_file_hash(self, rel_path: str) -> str | None:
        """Return stored hash for a file, or None if not indexed yet."""
        row = self._conn.execute(
            "SELECT hash FROM files WHERE rel_path = ?", (rel_path,)
        ).fetchone()
        return row["hash"] if row else None

    def get_symbol_ids_for_file(self, rel_path: str) -> list[int]:
        """Return all symbol IDs belonging to the given file path."""
        rows = self._conn.execute(
            """
            SELECT s.id FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE f.rel_path = ?
            """,
            (rel_path,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_symbol_ids_for_file_id(self, file_id: int) -> list[int]:
        """Return all symbol IDs belonging to the given file id.

        The file_id-keyed counterpart to `get_symbol_ids_for_file`. Callers that already
        hold a file_id — the import graph in particular, whose edges are file_id-based —
        should not have to round-trip through rel_path to use it.
        """
        rows = self._conn.execute("SELECT id FROM symbols WHERE file_id = ?", (file_id,)).fetchall()
        return [r[0] for r in rows]

    def upsert_file(self, file: IndexedFile) -> int:
        """Insert or update file record. Returns file id."""
        cursor = self._conn.execute(
            """
            INSERT INTO files (path, rel_path, language, hash, size_bytes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                hash       = excluded.hash,
                size_bytes = excluded.size_bytes,
                language   = excluded.language,
                indexed_at = datetime('now')
            RETURNING id
            """,
            (file.path, file.rel_path, file.language.value, file.hash, file.size_bytes),
        )
        row = cursor.fetchone()
        self._conn.commit()
        return row[0]  # type: ignore[no-any-return]

    def _sub_chunk_ids_for_symbols(self, symbol_ids: list[int]) -> list[int]:
        """Row ids of the sub_chunks belonging to the given symbols."""
        if not symbol_ids:
            return []
        placeholders = ",".join("?" for _ in symbol_ids)
        rows = self._conn.execute(
            f"SELECT id FROM sub_chunks WHERE parent_symbol_id IN ({placeholders})",
            tuple(symbol_ids),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def _purge_sub_chunks(self, symbol_ids: list[int], vector_store: object | None) -> None:
        """Delete sub_chunks rows for `symbol_ids`, and their vectors first.

        `sub_chunks.parent_symbol_id` carries NO foreign key — unlike `chunks.symbol_id`,
        which has ON DELETE CASCADE — so deleting a symbol left its sub-chunks behind
        forever. `PRAGMA foreign_key_list(sub_chunks)` returns []; there was also no
        `DELETE FROM sub_chunks` anywhere in src/, so rows accumulated on every re-index.

        The FK is not added retroactively because SQLite cannot ALTER TABLE a constraint
        in, and rebuilding a table that may already hold a user's rows is a bigger risk
        than doing the cleanup here.

        Vectors are deleted BEFORE the rows, because the row id is the only handle on its
        vector: losing the row first would leave a vector nothing can ever find. A
        foreign key could not have done this half regardless — cascade cannot reach a
        virtual table.
        """
        sub_chunk_ids = self._sub_chunk_ids_for_symbols(symbol_ids)
        if not sub_chunk_ids:
            return
        if vector_store is not None:
            vector_store.delete_sub_chunk_embeddings(sub_chunk_ids)  # type: ignore[attr-defined]
        placeholders = ",".join("?" for _ in sub_chunk_ids)
        self._conn.execute(
            f"DELETE FROM sub_chunks WHERE id IN ({placeholders})", tuple(sub_chunk_ids)
        )

    def _purge_def_use_edges(self, symbol_ids: list[int]) -> None:
        """Delete def_use_edges rows for `symbol_ids`. Must run BEFORE the symbols.

        `def_use_edges.symbol_id` is `INTEGER NOT NULL` with no REFERENCES, so it is
        the same defect as sub_chunks above — except this is the LARGEST table in the
        index (117,814 rows on the live 192 MB index vs 10,991 symbols, ~11 edges per
        symbol) and it had no DELETE anywhere in src/ at all. Every file deletion
        through delete_file_by_path() — the watcher's path, which runs whenever a file
        disappears from a watched repo — leaked the deleted symbols' edges, and both
        re-index paths leaked on every changed symbol. Measured on the live index
        before the fix: 4,768 of 117,814 rows (4.0%) already orphaned.

        An explicit deleter, not a retroactive FK. Adding `REFERENCES symbols(id) ON
        DELETE CASCADE` cannot be done by ALTER TABLE; it needs the 12-step rebuild,
        which on a 180+ MB index means copying 117k rows at open time for every
        existing install. Worse, the copy runs with `PRAGMA foreign_keys = ON` (it is
        on — see the DDL, verified), so the 4,768 pre-existing orphans would make
        `INSERT INTO new SELECT * FROM old` fail and the migration would abort the
        open. The deleter has none of that blast radius and covers the same paths.
        """
        if not symbol_ids:
            return
        placeholders = ",".join("?" for _ in symbol_ids)
        self._conn.execute(
            f"DELETE FROM def_use_edges WHERE symbol_id IN ({placeholders})", tuple(symbol_ids)
        )

    def _purge_sparse_embeddings(self, symbol_ids: list[int]) -> None:
        """Delete sparse_embeddings rows for the symbols' chunks. Must run BEFORE them.

        `sparse_embeddings.chunk_id` has the same FK-less shape one level further out:
        it points at `chunks.id`, and chunks die by cascade from symbols. SparseStore
        (store/sparse_store.py) only ever deletes a chunk_id it is about to re-upsert,
        so a chunk removed by cascade left its sparse rows behind permanently and
        unreachable — chunk ids are AUTOINCREMENT, never reused, so nothing can ever
        match them again.

        Latent rather than measured: the live index holds 0 sparse rows because SPLADE
        is flag-gated and was itself broken until recently (see the note in
        core/config.py). This closes it before the flag is worth turning on.
        """
        if not symbol_ids:
            return
        placeholders = ",".join("?" for _ in symbol_ids)
        self._conn.execute(
            "DELETE FROM sparse_embeddings WHERE chunk_id IN "
            f"(SELECT id FROM chunks WHERE symbol_id IN ({placeholders}))",
            tuple(symbol_ids),
        )

    def _purge_fkless_symbol_rows(self, symbol_ids: list[int], vector_store: object | None) -> None:
        """Every symbol-derived table that no ON DELETE CASCADE reaches.

        Called from all three symbol-removal paths (full re-index, partial
        content-hash-diffed re-index, watcher file delete). A new FK-less table keyed
        on symbols must be added here, not to the three call sites.
        """
        self._purge_sub_chunks(symbol_ids, vector_store)
        self._purge_def_use_edges(symbol_ids)
        self._purge_sparse_embeddings(symbol_ids)

    def delete_file_symbols(self, file_id: int, vector_store: object | None = None) -> None:
        """Remove all symbols (and cascaded data) for a file before re-indexing."""
        symbol_ids = [
            int(r[0])
            for r in self._conn.execute(
                "SELECT id FROM symbols WHERE file_id = ?", (file_id,)
            ).fetchall()
        ]
        self._purge_fkless_symbol_rows(symbol_ids, vector_store)
        self._conn.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
        self._conn.execute("DELETE FROM imports WHERE file_id = ?", (file_id,))
        self._conn.commit()

    def get_symbol_hashes_for_file(self, file_id: int) -> dict[str, str]:
        """Return {qualified_name: content_hash} for every symbol currently
        stored under file_id. Used to diff newly-parsed symbols against
        what's already indexed, so unchanged symbols can skip re-embedding.
        """
        rows = self._conn.execute(
            "SELECT qualified_name, content_hash FROM symbols WHERE file_id = ?",
            (file_id,),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def delete_symbols_by_qualified_names(
        self,
        file_id: int,
        qualified_names: list[str],
        vector_store: object | None = None,
    ) -> None:
        """Remove only the named symbols (and cascaded data) for a file —
        a partial version of delete_file_symbols(), used when some symbols
        in the file are unchanged and must be preserved.
        """
        if not qualified_names:
            return
        placeholders = ",".join("?" for _ in qualified_names)
        # Resolved before the delete: sub_chunks, def_use_edges and sparse_embeddings
        # have no FK to symbols, so once the symbol rows are gone nothing links their
        # rows to anything.
        symbol_ids = [
            int(r[0])
            for r in self._conn.execute(
                f"SELECT id FROM symbols WHERE file_id = ? AND qualified_name IN ({placeholders})",
                (file_id, *qualified_names),
            ).fetchall()
        ]
        self._purge_fkless_symbol_rows(symbol_ids, vector_store)
        self._conn.execute(
            f"DELETE FROM symbols WHERE file_id = ? AND qualified_name IN ({placeholders})",
            (file_id, *qualified_names),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # FK-link repair after a partial (content-hash-diffed) symbol delete
    # ------------------------------------------------------------------
    # symbols.parent_id / calls.callee_id / type_edges.to_symbol_id are all
    # ON DELETE SET NULL — deleting a changed/removed symbol's old row (via
    # delete_symbols_by_qualified_names above) silently NULLs these on any
    # OTHER row that referenced it, including unchanged rows the current
    # pass never re-inserts. The three getters below MUST be called BEFORE
    # delete_symbols_by_qualified_names() so they see the link before the
    # cascade erases it; the three repoint_* methods are called afterwards
    # to re-point the link at the symbol's new id, once known.

    def get_children_with_stale_parent(self, old_parent_ids: list[int]) -> list[tuple[int, int]]:
        """Return (child_id, old_parent_id) for symbols whose parent_id
        currently matches one of old_parent_ids."""
        if not old_parent_ids:
            return []
        placeholders = ",".join("?" for _ in old_parent_ids)
        rows = self._conn.execute(
            f"SELECT id, parent_id FROM symbols WHERE parent_id IN ({placeholders})",
            old_parent_ids,
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def repoint_parent_ids(self, child_id_to_new_parent_id: dict[int, int]) -> None:
        """Re-point parent_id for children whose parent row was deleted and
        re-inserted with a new id during a partial re-index."""
        if not child_id_to_new_parent_id:
            return
        self._conn.executemany(
            "UPDATE symbols SET parent_id = ? WHERE id = ?",
            [(new_id, child_id) for child_id, new_id in child_id_to_new_parent_id.items()],
        )
        self._conn.commit()

    def get_calls_referencing_symbols(self, old_callee_ids: list[int]) -> list[tuple[int, int]]:
        """Return (call_id, old_callee_id) for calls rows whose callee_id
        currently matches one of old_callee_ids."""
        if not old_callee_ids:
            return []
        placeholders = ",".join("?" for _ in old_callee_ids)
        rows = self._conn.execute(
            f"SELECT id, callee_id FROM calls WHERE callee_id IN ({placeholders})",
            old_callee_ids,
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def repoint_call_callee_ids(self, call_id_to_new_callee_id: dict[int, int]) -> None:
        """Re-point callee_id for calls whose callee row was deleted and
        re-inserted with a new id during a partial re-index."""
        if not call_id_to_new_callee_id:
            return
        self._conn.executemany(
            "UPDATE calls SET callee_id = ? WHERE id = ?",
            [(new_id, call_id) for call_id, new_id in call_id_to_new_callee_id.items()],
        )
        self._conn.commit()

    def get_type_edges_referencing_symbols(
        self, old_target_ids: list[int]
    ) -> list[tuple[int, int]]:
        """Return (edge_id, old_target_id) for type_edges rows whose
        to_symbol_id currently matches one of old_target_ids."""
        if not old_target_ids:
            return []
        placeholders = ",".join("?" for _ in old_target_ids)
        rows = self._conn.execute(
            f"SELECT id, to_symbol_id FROM type_edges WHERE to_symbol_id IN ({placeholders})",
            old_target_ids,
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def repoint_type_edge_targets(self, edge_id_to_new_target_id: dict[int, int]) -> None:
        """Re-point to_symbol_id for type edges whose target row was deleted
        and re-inserted with a new id during a partial re-index."""
        if not edge_id_to_new_target_id:
            return
        self._conn.executemany(
            "UPDATE type_edges SET to_symbol_id = ? WHERE id = ?",
            [(new_id, edge_id) for edge_id, new_id in edge_id_to_new_target_id.items()],
        )
        self._conn.commit()

    def delete_file_by_path(
        self,
        abs_path: str,
        rel_path: str,
        vector_store: object | None = None,
    ) -> bool:
        """
        Fully delete a file's index data (file row + symbols + chunks + vectors).

        Cascades:
          - symbols ON DELETE CASCADE removes chunks, calls, type_edges
          - imports ON DELETE CASCADE removed via file_id FK
          - vector_store.delete_batch() cleans up embeddings if provided

        Args:
            abs_path:     Absolute filesystem path (used as primary lookup key).
            rel_path:     Repo-relative path (used as fallback lookup key).
            vector_store: Optional VectorStore — if provided, chunk vectors are
                          deleted before the DB rows are removed.

        Returns:
            True if a matching file row was found and deleted, False otherwise.
        """
        row = self._conn.execute(
            "SELECT id FROM files WHERE path = ? OR rel_path = ? LIMIT 1",
            (abs_path, rel_path),
        ).fetchone()

        if row is None:
            return False

        file_id: int = row[0]

        # Delete vectors before DB rows so we never have orphaned vectors.
        if vector_store is not None:
            # The file-summary vector is included, and it is NOT in
            # `get_chunk_ids_for_file()`. Summaries live in the same `chunk_embeddings`
            # table under the `chunk_id = -(file_id)` sentinel, so a delete driven only by
            # chunk ids never reached them — while the `file_summaries` row itself cascades
            # away with the `files` row, leaving a vector with neither a summary nor a file.
            #
            # Found by `verify-index.sh`'s "summary vectors == summaries" gate immediately
            # after the first real `--prune` removed one file: 482 vectors against 481
            # summaries, the orphan being `-475` for a `file_id` that no longer existed.
            # That gate was widened earlier in this release to catch exactly this class. The
            # leak long predates `--prune` — the watcher's delete path has taken it for as
            # long as file summaries have existed.
            #
            # One call rather than two, and deliberately NOT gated on `chunk_ids` being
            # non-empty: a file can have a summary and no chunks (every zero-symbol file
            # does, and there were 11 of those before the line-window fallback), so a guard
            # on chunk ids would skip precisely the files whose only vector is the sentinel.
            vector_store.delete_batch(  # type: ignore[attr-defined]
                [*self.get_chunk_ids_for_file(file_id), -file_id]
            )

        # The FK-less symbol-derived tables too. `symbols` cascades from `files`, but
        # sub_chunks / def_use_edges / sparse_embeddings have no foreign key to
        # `symbols`, so deleting the file row here removed the symbols and left their
        # rows — and, for sub-chunks, their vectors — behind. This is the WATCHER's
        # path, so it ran every time a file was deleted from a watched repo; the
        # def_use_edges half of that leak is where the live index's 4,768 orphans
        # came from.
        symbol_ids = [
            int(r[0])
            for r in self._conn.execute(
                "SELECT id FROM symbols WHERE file_id = ?", (file_id,)
            ).fetchall()
        ]
        self._purge_fkless_symbol_rows(symbol_ids, vector_store)

        # ON DELETE CASCADE on symbols handles chunks / calls / type_edges
        # Explicit import delete handles the file_id FK
        self._conn.execute("DELETE FROM imports WHERE file_id = ?", (file_id,))
        self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        self._conn.commit()
        return True

    # ------------------------------------------------------------------
    # Symbols
    # ------------------------------------------------------------------

    def insert_symbol(self, symbol: Symbol) -> int:
        content_hash = hashlib.sha256((symbol.signature + symbol.body).encode("utf-8")).hexdigest()
        cursor = self._conn.execute(
            """
            INSERT INTO symbols
              (file_id, name, qualified_name, kind, line_start, line_end,
               signature, docstring, context_summary, decorators, is_public, parent_id, body,
               content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol.file_id,
                symbol.name,
                symbol.qualified_name,
                symbol.kind.value,
                symbol.line_start,
                symbol.line_end,
                symbol.signature,
                symbol.docstring,
                symbol.context_summary,
                json.dumps(symbol.decorators),
                int(symbol.is_public),
                symbol.parent_id,
                symbol.body,
                content_hash,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_symbol_by_name(self, name: str) -> list[Symbol]:
        rows = self._conn.execute(
            "SELECT * FROM symbols WHERE name = ? OR qualified_name = ?",
            (name, name),
        ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def get_all_symbol_names(self) -> list[tuple[int, str, str]]:
        """(id, name, qualified_name) for every symbol — bulk lookup for
        scanning free text (e.g. artifact bodies) for identifier mentions,
        where repeated single-name queries would be too slow."""
        rows = self._conn.execute("SELECT id, name, qualified_name FROM symbols").fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def get_symbols_for_file(self, file_id: int) -> list[Symbol]:
        rows = self._conn.execute("SELECT * FROM symbols WHERE file_id = ?", (file_id,)).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def insert_call_edges(self, edges: list[CallEdge]) -> None:
        self._conn.executemany(
            "INSERT INTO calls (caller_id, callee_name, callee_id, line, callee_type_hint)"
            " VALUES (?, ?, ?, ?, ?)",
            [(e.caller_id, e.callee_name, e.callee_id, e.line, e.callee_type_hint) for e in edges],
        )

    def get_callees(self, symbol_id: int) -> list[int]:
        """Return symbol ids that symbol_id calls (1 hop out)."""
        rows = self._conn.execute(
            "SELECT callee_id FROM calls WHERE caller_id = ? AND callee_id IS NOT NULL",
            (symbol_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_callers(self, symbol_id: int) -> list[int]:
        """Return symbol ids that call symbol_id (1 hop in)."""
        rows = self._conn.execute(
            "SELECT caller_id FROM calls WHERE callee_id = ?", (symbol_id,)
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def insert_imports(self, edges: list[ImportEdge]) -> None:
        self._conn.executemany(
            "INSERT INTO imports (file_id, imported_from, imported_names) VALUES (?, ?, ?)",
            [(e.file_id, e.imported_from, json.dumps(e.imported_names)) for e in edges],
        )

    def get_imports_for_file(self, file_id: int) -> list[ImportEdge]:
        rows = self._conn.execute("SELECT * FROM imports WHERE file_id = ?", (file_id,)).fetchall()
        return [
            ImportEdge(
                file_id=r["file_id"],
                imported_from=r["imported_from"],
                imported_names=json.loads(r["imported_names"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: Chunk) -> int:
        cursor = self._conn.execute(
            "INSERT INTO chunks (symbol_id, chunk_text, token_count) VALUES (?, ?, ?)",
            (chunk.symbol_id, chunk.chunk_text, chunk.token_count),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def insert_chunk_for_symbol(self, symbol_id: int, chunk_text: str, token_count: int) -> int:
        """Insert a chunk for a symbol — graph/test helper. No-op if symbol already has a chunk."""
        existing = self._conn.execute(
            "SELECT id FROM chunks WHERE symbol_id = ?", (symbol_id,)
        ).fetchone()
        if existing:
            return int(existing[0])
        cur = self._conn.execute(
            "INSERT INTO chunks (symbol_id, chunk_text, token_count) VALUES (?, ?, ?)",
            (symbol_id, chunk_text, token_count),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    # ------------------------------------------------------------------
    # Sub-symbol chunks (MGS3: multi-granularity sub-symbol indexing)
    # ------------------------------------------------------------------

    def insert_sub_chunks(self, chunks: list[SubSymbolChunk]) -> list[int]:
        """Insert sub-symbol chunks. Returns list of inserted IDs."""
        if not chunks:
            return []
        ids: list[int] = []
        for chunk in chunks:
            granularity_val = (
                chunk.granularity.value
                if hasattr(chunk.granularity, "value")
                else chunk.granularity
            )
            cur = self._conn.execute(
                "INSERT INTO sub_chunks "
                "(parent_symbol_id, granularity, chunk_text, line_start, line_end, token_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chunk.parent_symbol_id,
                    granularity_val,
                    chunk.chunk_text,
                    chunk.line_start,
                    chunk.line_end,
                    chunk.token_count,
                ),
            )
            ids.append(int(cur.lastrowid or 0))
        self._conn.commit()
        return ids

    def get_sub_chunks_for_symbol(
        self, symbol_id: int, granularity: str | None = None
    ) -> list[SubSymbolChunk]:
        """Return sub-chunks for a symbol, optionally filtered by granularity."""
        from trelix.indexing.multi_granularity import Granularity, SubSymbolChunk

        if granularity:
            rows = self._conn.execute(
                "SELECT id, parent_symbol_id, granularity, chunk_text, "
                "line_start, line_end, token_count "
                "FROM sub_chunks WHERE parent_symbol_id = ? AND granularity = ?",
                (symbol_id, granularity),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, parent_symbol_id, granularity, chunk_text, "
                "line_start, line_end, token_count "
                "FROM sub_chunks WHERE parent_symbol_id = ?",
                (symbol_id,),
            ).fetchall()
        return [
            SubSymbolChunk(
                id=int(row[0]),
                parent_symbol_id=int(row[1]),
                granularity=Granularity(row[2]),
                chunk_text=row[3],
                line_start=int(row[4]),
                line_end=int(row[5]),
                token_count=int(row[6]),
            )
            for row in rows
        ]

    def get_sub_chunk_by_id(self, sub_chunk_id: int) -> SubSymbolChunk | None:
        """Return a single sub-chunk by primary key, or None if not found."""
        from trelix.indexing.multi_granularity import Granularity, SubSymbolChunk

        row = self._conn.execute(
            "SELECT id, parent_symbol_id, granularity, chunk_text, "
            "line_start, line_end, token_count "
            "FROM sub_chunks WHERE id = ?",
            (sub_chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return SubSymbolChunk(
            id=int(row[0]),
            parent_symbol_id=int(row[1]),
            granularity=Granularity(row[2]),
            chunk_text=row[3],
            line_start=int(row[4]),
            line_end=int(row[5]),
            token_count=int(row[6]),
        )

    # ------------------------------------------------------------------
    # File summaries (Phase 2: RAPTOR-style multi-granularity indexing)
    # ------------------------------------------------------------------

    def upsert_file_summary(self, file_id: int, summary: str, chunk_id: int | None = None) -> int:
        """Insert or replace a file-level summary. Returns the row id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO file_summaries (file_id, summary, chunk_id) VALUES (?, ?, ?) "
                "ON CONFLICT(file_id) DO UPDATE SET summary=excluded.summary, "
                "chunk_id=excluded.chunk_id, created_at=datetime('now')",
                (file_id, summary, chunk_id),
            )
        return cur.lastrowid or 0

    def get_file_summary(self, file_id: int) -> str | None:
        """Return the stored summary for a file, or None if not found."""
        row = self._conn.execute(
            "SELECT summary FROM file_summaries WHERE file_id = ?", (file_id,)
        ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # BM25 search (FTS5)
    # ------------------------------------------------------------------

    def bm25_search(
        self,
        query: str,
        limit: int = 20,
        path_filter: str | None = None,
        declaration_boost_weight: float = 1.0,
    ) -> list[tuple[int, float]]:
        """
        Full-text search over symbols using SQLite FTS5 BM25.
        Returns list of (symbol_id, rank) sorted by relevance.
        Lower rank = more relevant in SQLite FTS5 (it's negative BM25).

        `path_filter`, when set, restricts results to symbols whose file's
        rel_path starts with that prefix — pushed into the SQL join (same
        `f.rel_path LIKE ?` pattern as grep_search.py) rather than fetched
        and discarded, since FTS5 can take the join cheaply.

        `declaration_boost_weight` (default 1.0 — a no-op) is applied to the
        `name`/`qualified_name` FTS5 columns via an explicit bm25() call;
        `docstring`/`body`/`context_summary` stay at weight 1.0. Must call
        `bm25(symbols_fts, ...)` using the real table name even in the
        `path_filter` branch's aliased join — `bm25(f, ...)` raises
        "no such column: f".

        Draws from the read-only connection pool when enable_bm25_read_pool()
        has been called with pool_size > 0 — otherwise uses the single
        shared writer connection exactly as before.
        """
        w = declaration_boost_weight
        if path_filter:
            sql = """
                SELECT f.rowid, bm25(symbols_fts, ?, ?, 1.0, 1.0, 1.0) AS rank
                FROM symbols_fts f
                JOIN symbols s ON f.rowid = s.id
                JOIN files fi ON s.file_id = fi.id
                WHERE symbols_fts MATCH ?
                  AND fi.rel_path LIKE ?
                ORDER BY rank
                LIMIT ?
                """
            params: tuple[object, ...] = (w, w, query, f"{path_filter}%", limit)
        else:
            sql = """
                SELECT rowid, bm25(symbols_fts, ?, ?, 1.0, 1.0, 1.0) AS rank
                FROM symbols_fts
                WHERE symbols_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """
            params = (w, w, query, limit)
        if self._bm25_read_pool is not None:
            with self._bm25_read_pool.acquire() as conn:
                rows = conn.execute(sql, params).fetchall()
        else:
            with self._conn_lock:
                rows = self._conn.execute(sql, params).fetchall()
        return [(r[0], r[1]) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_symbol(self, row: sqlite3.Row) -> Symbol:
        return Symbol(
            id=row["id"],
            file_id=row["file_id"],
            name=row["name"],
            qualified_name=row["qualified_name"],
            kind=SymbolKind(row["kind"]),
            line_start=row["line_start"],
            line_end=row["line_end"],
            signature=row["signature"],
            docstring=row["docstring"],
            context_summary=row["context_summary"],
            decorators=json.loads(row["decorators"] or "[]"),
            is_public=bool(row["is_public"]),
            parent_id=row["parent_id"],
            body=row["body"],
        )

    # ------------------------------------------------------------------
    # Incremental cleanup
    # ------------------------------------------------------------------

    def get_chunk_ids_for_file(self, file_id: int) -> list[int]:
        """Return all chunk ids for a file. Called before re-indexing to clean the vector store."""
        rows = self._conn.execute(
            """
            SELECT c.id FROM chunks c
            JOIN symbols s ON c.symbol_id = s.id
            WHERE s.file_id = ?
            """,
            (file_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_chunk_ids_for_symbols(self, file_id: int, qualified_names: list[str]) -> list[int]:
        """Return chunk ids for a subset of a file's symbols (by qualified_name).

        Used by the incremental re-index path to clean up only the chunks
        belonging to symbols that actually changed, leaving unchanged
        symbols' chunks/embeddings untouched.
        """
        if not qualified_names:
            return []
        placeholders = ",".join("?" for _ in qualified_names)
        rows = self._conn.execute(
            f"""
            SELECT c.id FROM chunks c
            JOIN symbols s ON c.symbol_id = s.id
            WHERE s.file_id = ? AND s.qualified_name IN ({placeholders})
            """,
            (file_id, *qualified_names),
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Cross-file call resolution (second pass after all files indexed)
    # ------------------------------------------------------------------

    def resolve_cross_file_calls(self) -> int:
        """
        Update callee_id for all unresolved call edges using a 4-priority cascade.
        Called once after all files are indexed.

        Resolution priority (highest → lowest):

          1. Qualified-name exact match — callee_name equals a symbol's qualified_name.
             Most precise: resolves "UserService.login" directly.

          2. Type-hint + name match — callee_type_hint is set AND a symbol exists with
             the given name whose qualified_name starts with "<callee_type_hint>.".
             Resolves method calls like user_service.login() when user_service: UserService
             is annotated in the calling function's parameter list.

          3. Name-only if unique — callee_name matches exactly one symbol across the whole
             index.  Safe when there is no ambiguity.

          4. Leave NULL — callee_name is ambiguous (multiple candidates) or unknown.
             A wrong edge is worse than no edge.

        Returns total number of newly resolved edges across all passes.
        """
        total_resolved = 0

        # ── Pass 1: qualified_name exact match ───────────────────────────────
        cursor = self._conn.execute(
            """
            UPDATE calls
            SET callee_id = (
                SELECT id FROM symbols
                WHERE qualified_name = calls.callee_name
                LIMIT 1
            )
            WHERE callee_id IS NULL
              AND EXISTS (
                SELECT 1 FROM symbols WHERE qualified_name = calls.callee_name
              )
            """
        )
        total_resolved += cursor.rowcount

        # ── Pass 2: type-hint assisted name match ────────────────────────────
        # Only fires when callee_type_hint is non-NULL and a symbol with the
        # right name is found whose qualified_name starts with "<hint>.".
        cursor = self._conn.execute(
            """
            UPDATE calls
            SET callee_id = (
                SELECT id FROM symbols
                WHERE name = calls.callee_name
                  AND qualified_name LIKE (calls.callee_type_hint || '.%')
                LIMIT 1
            )
            WHERE callee_id IS NULL
              AND callee_type_hint IS NOT NULL
              AND EXISTS (
                SELECT 1 FROM symbols
                WHERE name = calls.callee_name
                  AND qualified_name LIKE (calls.callee_type_hint || '.%')
              )
            """
        )
        total_resolved += cursor.rowcount

        # ── Pass 3: name-only if unique (no ambiguity) ───────────────────────
        cursor = self._conn.execute(
            """
            UPDATE calls
            SET callee_id = (
                SELECT id FROM symbols WHERE name = calls.callee_name
            )
            WHERE callee_id IS NULL
              AND (
                SELECT COUNT(*) FROM symbols WHERE name = calls.callee_name
              ) = 1
            """
        )
        total_resolved += cursor.rowcount

        # Pass 4: leave NULL — ambiguous callee_name, better no edge than wrong one.

        self._conn.commit()
        return total_resolved

    # ------------------------------------------------------------------
    # Import file resolution (second pass after all files indexed)
    # ------------------------------------------------------------------

    def resolve_import_file_ids(self) -> int:
        """
        Resolve imported_from module paths to actual file_ids.
        Works for Python (dotted paths), JS/TS (relative), Go, Java, Rust.
        Populates imports.imported_file_id for all resolvable internal imports.
        Returns number of newly resolved imports.
        """
        all_files: dict[int, str] = {
            r[0]: r[1] for r in self._conn.execute("SELECT id, rel_path FROM files").fetchall()
        }

        # Build path lookup: normalized path (no ext, no common prefix) → file_id
        # Stores multiple variants per file for fuzzy matching
        path_lookup: dict[str, int] = {}
        for fid, rel in all_files.items():
            no_ext = rel
            stripped_ext = ""
            for ext in (".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".go", ".rs", ".java"):
                if no_ext.endswith(ext):
                    no_ext = no_ext[: -len(ext)]
                    stripped_ext = ext
                    break
            # "a directory IS a module" filename conventions — each gated on the
            # extension that owns it. Applying them language-blind made a Python
            # "pkg/mod.py" register under the key "pkg", so it masqueraded as its
            # own package directory and absorbed every "from . import x" and
            # "from ..pkg import y" aimed at pkg/__init__.py.
            if stripped_ext == ".rs":
                no_ext = no_ext.removesuffix("/mod")  # Rust: store/db/mod.rs → store/db
            if stripped_ext in (".ts", ".tsx", ".js", ".jsx", ".mjs"):
                # JS: components/Button/index.ts → components/Button
                no_ext = no_ext.removesuffix("/index")

            # Full path without extension
            path_lookup[no_ext] = fid

            # Without common source root prefixes (src/, lib/, app/)
            for prefix in ("src/", "lib/", "app/"):
                if no_ext.startswith(prefix):
                    path_lookup[no_ext[len(prefix) :]] = fid
                    break

            # Python's own "a directory IS a module" convention: pkg/__init__.py
            # *is* the module "pkg", so both "import a.pkg" and a relative
            # ".pkg" name the directory. Registered under the directory key with
            # setdefault so a real module file that already claimed the key keeps
            # it, and deliberately NOT under the bare last component — that would
            # be the directory name, and _resolve_module_to_file's Go/Java branch
            # matches a single trailing segment, which would mint cross-language
            # edges into unrelated __init__.py files.
            package_key = no_ext.removesuffix("/__init__")
            if package_key != no_ext:
                path_lookup.setdefault(package_key, fid)
                for prefix in ("src/", "lib/", "app/"):
                    if package_key.startswith(prefix):
                        path_lookup.setdefault(package_key[len(prefix) :], fid)
                        break

            # Last component only — used as fallback for single-segment matches
            last = no_ext.split("/")[-1]
            if last and last not in path_lookup:
                path_lookup[last] = fid

        rows = self._conn.execute(
            "SELECT id, file_id, imported_from FROM imports WHERE imported_file_id IS NULL"
        ).fetchall()

        resolved = 0
        for import_id, importer_file_id, imported_from in rows:
            if not imported_from:
                continue
            target_fid = self._resolve_module_to_file(
                imported_from, importer_file_id, all_files, path_lookup
            )
            if target_fid is not None and target_fid != importer_file_id:
                self._conn.execute(
                    "UPDATE imports SET imported_file_id = ? WHERE id = ?",
                    (target_fid, import_id),
                )
                resolved += 1

        self._conn.commit()
        return resolved

    def _resolve_module_to_file(
        self,
        module_path: str,
        importer_file_id: int,
        all_files: dict[int, str],
        path_lookup: dict[str, int],
    ) -> int | None:
        """
        Resolve a single import path to a file_id using language-aware heuristics.
        Handles: Python dotted paths, Python dot-separated relatives (".x", "..x.y"),
        JS/TS slash-separated relatives, Go/Java slash paths, Rust :: paths.
        """
        # --- Relative imports (JS/TS, Python in-package) ---
        if module_path.startswith("."):
            importer_rel = all_files.get(importer_file_id, "")
            importer_dirs = importer_rel.split("/")[:-1]

            # Two conventions share the leading dot and they are NOT
            # interchangeable. Python separates with dots ("..store.db"); JS/TS,
            # Go, CSS and friends separate with slashes ("../store/db"). This
            # branch only ever did the slash split, so a Python specifier stayed
            # one indivisible component: ".cache" matched neither ".." nor "."
            # and was appended verbatim, yielding the nonexistent candidate
            # "pkg/.cache". Measured on trelix's own index: slash-style resolved
            # 34/34, dot-style 0/40, every row naming its target correctly and
            # carrying imported_file_id = NULL.
            #
            # The dot COUNT also means different things, so a single shared
            # algorithm would break whichever convention it was not written for:
            #     ".x"  Python = this package (up 0)   "./x"  JS = this dir
            #     "..x" Python = parent pkg   (up 1)   "../x" JS = parent dir
            # i.e. Python ascends (dots - 1); a slash path ascends once per "..".
            # A specifier with no "/" at all cannot be a slash path, and the
            # degenerate "." / ".." agree under both readings (this dir / parent
            # dir), so the presence of "/" is a safe discriminator.
            if "/" not in module_path:
                leading_dots = len(module_path) - len(module_path.lstrip("."))
                ascend = leading_dots - 1
                if ascend > len(importer_dirs):
                    return None  # climbs past the repo root — nothing to point at
                base = importer_dirs[: len(importer_dirs) - ascend]
                tail = [p for p in module_path[leading_dots:].split(".") if p]
                candidate = "/".join(base + tail)
                # No suffix fallback on purpose: a relative import names an exact
                # module, so a near-miss here means the target is outside the
                # index. A wrong edge is worse than a missing one.
                return path_lookup.get(candidate)

            joined = "/".join(importer_dirs) + "/" + module_path if importer_dirs else module_path
            parts: list[str] = []
            for component in joined.split("/"):
                if component == "..":
                    if parts:
                        parts.pop()
                elif component and component != ".":
                    parts.append(component)
            candidate = "/".join(parts)
            if candidate in path_lookup:
                return path_lookup[candidate]
            # Strip trailing extension from the import itself (e.g. "./utils.js")
            for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".py"):
                if candidate.endswith(ext):
                    stripped = candidate[: -len(ext)]
                    if stripped in path_lookup:
                        return path_lookup[stripped]
            return None

        # --- Skip well-known external / stdlib prefixes ---
        _external = (
            "java.",
            "javax.",
            "android.",
            "kotlin.",  # Java/Kotlin stdlib
            "std::",
            "core::",
            "alloc::",  # Rust std
            "react",
            "vue",
            "@angular",
            "@types/",  # JS/TS frameworks
            "lodash",
            "axios",
            "express",
            "next",
            "nuxt",
        )
        if any(
            module_path == x
            or module_path.startswith(x + ".")
            or module_path.startswith(x + "/")
            or module_path.startswith(x + "::")
            for x in _external
        ):
            return None

        # Skip single-word Go stdlib packages (fmt, os, io, strings, etc.)
        if "/" not in module_path and "." not in module_path and "::" not in module_path:
            return None

        # --- Python: "trelix.store.db" → "trelix/store/db" ---
        if "." in module_path and "/" not in module_path and "::" not in module_path:
            candidate = module_path.replace(".", "/")
            if candidate in path_lookup:
                return path_lookup[candidate]
            parts_list = candidate.split("/")
            for n in (2, 1):
                if len(parts_list) >= n:
                    partial = "/".join(parts_list[-n:])
                    if partial in path_lookup:
                        return path_lookup[partial]
            return None

        # --- Rust use paths: "crate::store::db" → "store/db" ---
        if "::" in module_path:
            candidate = module_path
            for prefix in ("crate::", "super::", "self::"):
                if candidate.startswith(prefix):
                    candidate = candidate[len(prefix) :]
            candidate = candidate.replace("::", "/")
            if candidate in path_lookup:
                return path_lookup[candidate]
            parts_list = candidate.split("/")
            for n in (2, 1):
                if len(parts_list) >= n:
                    partial = "/".join(parts_list[-n:])
                    if partial in path_lookup:
                        return path_lookup[partial]
            return None

        # --- Go / Java: slash-delimited paths ---
        if "/" in module_path:
            parts_list = module_path.split("/")
            for n in (3, 2, 1):
                if len(parts_list) >= n:
                    partial = "/".join(parts_list[-n:])
                    if partial in path_lookup:
                        return path_lookup[partial]
            return None

        return None

    def get_file_imports_resolved(self, file_id: int) -> list[int]:
        """Return file_ids that this file imports (resolved internal imports only)."""
        rows = self._conn.execute(
            "SELECT imported_file_id FROM imports"
            " WHERE file_id = ? AND imported_file_id IS NOT NULL",
            (file_id,),
        ).fetchall()
        return list({r[0] for r in rows})

    def get_files_importing(self, file_id: int) -> list[int]:
        """Return file_ids that import this file (reverse direction)."""
        rows = self._conn.execute(
            "SELECT DISTINCT file_id FROM imports WHERE imported_file_id = ?",
            (file_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_file_by_rel_path_suffix(self, suffix: str) -> int | None:
        """
        Return the file_id whose rel_path ends with ``suffix``.
        Strips a leading slash from ``suffix`` before matching.
        Returns None if zero or multiple files match (ambiguous).
        """
        suffix = suffix.lstrip("/")
        rows = self._conn.execute(
            "SELECT id FROM files WHERE rel_path = ? OR rel_path LIKE ?",
            (suffix, f"%/{suffix}"),
        ).fetchall()
        if len(rows) == 1:
            return int(rows[0][0])
        return None

    # ------------------------------------------------------------------
    # Type edges
    # ------------------------------------------------------------------

    def insert_type_edges(self, edges: list[TypeEdge]) -> None:
        self._conn.executemany(
            "INSERT INTO type_edges"
            " (from_symbol_id, to_type_name, edge_kind, to_symbol_id)"
            " VALUES (?, ?, ?, ?)",
            [(e.from_symbol_id, e.to_type_name, e.edge_kind, e.to_symbol_id) for e in edges],
        )

    def get_type_parents(self, symbol_id: int) -> list[int]:
        """Return symbol ids this symbol inherits/implements (outgoing type edges)."""
        rows = self._conn.execute(
            "SELECT to_symbol_id FROM type_edges"
            " WHERE from_symbol_id = ? AND to_symbol_id IS NOT NULL",
            (symbol_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_type_children(self, symbol_id: int) -> list[int]:
        """Return symbol ids that inherit/implement this symbol (incoming type edges)."""
        rows = self._conn.execute(
            "SELECT from_symbol_id FROM type_edges WHERE to_symbol_id = ?",
            (symbol_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def resolve_cross_file_type_edges(self) -> int:
        """
        Resolve to_symbol_id for type edges where to_type_name matches a known symbol.
        Called once after all files indexed — mirrors resolve_cross_file_calls().
        Returns number of newly resolved edges.
        """
        cursor = self._conn.execute(
            """
            UPDATE type_edges
            SET to_symbol_id = (
                SELECT id FROM symbols
                WHERE name = type_edges.to_type_name
                  AND kind IN ('class', 'interface', 'struct', 'enum')
                LIMIT 1
            )
            WHERE to_symbol_id IS NULL
              AND EXISTS (
                SELECT 1 FROM symbols
                WHERE name = type_edges.to_type_name
                  AND kind IN ('class', 'interface', 'struct', 'enum')
              )
            """
        )
        self._conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Graph iteration helpers (used by CodeGraph)
    # ------------------------------------------------------------------

    def iter_all_symbols_with_files(
        self,
    ) -> list[tuple[Symbol, IndexedFile]]:
        """Return (Symbol, IndexedFile) for every symbol in the DB."""
        rows = self._conn.execute(
            """
            SELECT s.id, s.file_id, s.name, s.qualified_name, s.kind,
                   s.line_start, s.line_end, s.signature, s.docstring,
                   s.context_summary, s.decorators, s.is_public, s.parent_id, s.body,
                   f.id, f.path, f.rel_path, f.language, f.hash, f.size_bytes
            FROM symbols s
            JOIN files f ON f.id = s.file_id
            """
        ).fetchall()
        result: list[tuple[Symbol, IndexedFile]] = []
        for row in rows:
            sym = Symbol(
                id=row[0],
                file_id=row[1],
                name=row[2],
                qualified_name=row[3],
                kind=SymbolKind(row[4]),
                line_start=row[5],
                line_end=row[6],
                signature=row[7] or "",
                docstring=row[8],
                context_summary=row[9],
                decorators=json.loads(row[10] or "[]"),
                is_public=bool(row[11]),
                parent_id=row[12],
                body=row[13] or "",
            )
            fi = IndexedFile(
                id=row[14],
                path=row[15],
                rel_path=row[16],
                language=Language(row[17]),
                hash=row[18],
                size_bytes=row[19],
            )
            result.append((sym, fi))
        return result

    def iter_resolved_calls(self) -> list[tuple[int, int]]:
        """Return (caller_id, callee_id) for all resolved call edges."""
        rows = self._conn.execute(
            "SELECT caller_id, callee_id FROM calls WHERE callee_id IS NOT NULL"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def iter_resolved_imports(self) -> list[tuple[int, int]]:
        """Return (file_id, imported_file_id) for all resolved import edges."""
        rows = self._conn.execute(
            "SELECT file_id, imported_file_id FROM imports WHERE imported_file_id IS NOT NULL"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def iter_resolved_type_edges(self) -> list[tuple[int, str, int]]:
        """Return (from_symbol_id, edge_kind, to_symbol_id) for all resolved type edges."""
        rows = self._conn.execute(
            "SELECT from_symbol_id, edge_kind, to_symbol_id FROM type_edges"
            " WHERE to_symbol_id IS NOT NULL"
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    # ------------------------------------------------------------------
    # Generic (cross-source) edges
    # ------------------------------------------------------------------

    def insert_generic_edges(self, edges: list[GenericEdge]) -> None:
        """Re-inserting an edge already present (same from_symbol_id,
        source_ref, edge_kind) is a silent no-op — see idx_generic_edges_dedup,
        so a writer can be re-run against the same repo without duplicating
        rows."""
        self._conn.executemany(
            "INSERT OR IGNORE INTO generic_edges"
            " (from_symbol_id, source_ref, edge_kind, weight)"
            " VALUES (?, ?, ?, ?)",
            [(e.from_symbol_id, e.source_ref, e.edge_kind, e.weight) for e in edges],
        )
        self._conn.commit()

    def get_generic_edge_targets(self, symbol_id: int) -> list[str]:
        """Return source_ref strings for every generic edge originating from
        this symbol (e.g. ["ticket:PROJ-123", "ticket:PROJ-456"])."""
        rows = self._conn.execute(
            "SELECT source_ref FROM generic_edges WHERE from_symbol_id = ?",
            (symbol_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def iter_resolved_generic_edges(self) -> list[tuple[int, str, str, float]]:
        """Return (from_symbol_id, edge_kind, source_ref, weight) for every
        generic edge — mirrors iter_resolved_type_edges(). "Resolved" here
        just means from_symbol_id is set (always true for every current
        writer); no further resolution pass exists for the source_ref side,
        unlike type_edges' to_symbol_id, since source_ref never refers to a
        row in this DB. weight is surfaced so callers (CodeGraph._build())
        can carry a lower-confidence match's reduced influence into
        PageRank rather than the column being persisted but silently
        ignored on every read path."""
        rows = self._conn.execute(
            "SELECT from_symbol_id, edge_kind, source_ref, weight FROM generic_edges"
            " WHERE from_symbol_id IS NOT NULL"
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    # ------------------------------------------------------------------
    # Artifacts (source-connector content — Jira tickets, TestRail cases)
    # ------------------------------------------------------------------

    def upsert_artifact(self, artifact: Artifact) -> int:
        """Insert or update by source_ref (UNIQUE) — re-syncing a connector
        overwrites the previous fetch rather than duplicating rows."""
        cursor = self._conn.execute(
            """
            INSERT INTO artifacts (source_ref, artifact_kind, title, body, url, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_ref) DO UPDATE SET
                artifact_kind = excluded.artifact_kind,
                title = excluded.title,
                body = excluded.body,
                url = excluded.url,
                metadata = excluded.metadata,
                fetched_at = datetime('now')
            """,
            (
                artifact.source_ref,
                artifact.artifact_kind,
                artifact.title,
                artifact.body,
                artifact.url,
                json.dumps(artifact.metadata),
            ),
        )
        self._conn.commit()
        if cursor.lastrowid:
            return int(cursor.lastrowid)
        row = self._conn.execute(
            "SELECT id FROM artifacts WHERE source_ref = ?", (artifact.source_ref,)
        ).fetchone()
        return int(row[0])

    def get_artifact_by_source_ref(self, source_ref: str) -> Artifact | None:
        row = self._conn.execute(
            "SELECT id, source_ref, artifact_kind, title, body, url, metadata"
            " FROM artifacts WHERE source_ref = ?",
            (source_ref,),
        ).fetchone()
        if row is None:
            return None
        return Artifact(
            id=row[0],
            source_ref=row[1],
            artifact_kind=row[2],
            title=row[3],
            body=row[4],
            url=row[5],
            metadata=json.loads(row[6]),
        )

    def get_all_artifacts(self) -> list[Artifact]:
        """Every artifact fetched by any connector — used by ArtifactLinker
        to scan for symbol references across the whole table, since
        get_artifact_by_source_ref() only looks up one at a time."""
        rows = self._conn.execute(
            "SELECT id, source_ref, artifact_kind, title, body, url, metadata FROM artifacts"
        ).fetchall()
        return [
            Artifact(
                id=row[0],
                source_ref=row[1],
                artifact_kind=row[2],
                title=row[3],
                body=row[4],
                url=row[5],
                metadata=json.loads(row[6]),
            )
            for row in rows
        ]

    def get_file_by_id(self, file_id: int) -> IndexedFile | None:
        """Fetch a file record by primary key."""
        row = self._conn.execute(
            "SELECT id, path, rel_path, language, hash, size_bytes FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        if row is None:
            return None
        return IndexedFile(
            id=row[0],
            path=row[1],
            rel_path=row[2],
            language=Language(row[3]),
            hash=row[4],
            size_bytes=row[5],
        )

    # ------------------------------------------------------------------
    # Angular selector resolution (second pass after all files indexed)
    # ------------------------------------------------------------------

    _SELECTOR_RE = re.compile(r"selector\s*:\s*['\"]([^'\"]+)['\"]")

    def resolve_angular_selectors(self) -> int:
        """
        Create type_edges linking Angular @Component/@Directive TypeScript class symbols
        to their HTML template custom-element symbols.

        After indexing, if a TypeScript class has a decorator containing
        selector: 'app-xyz' and there exists an HTML symbol named 'app-xyz'
        (extracted by HtmlParser as a custom element), we insert a type_edge:
            from_symbol_id = TypeScript class id
            to_type_name   = 'app-xyz'
            edge_kind      = 'angular_selector'
            to_symbol_id   = HTML custom-element symbol id

        Because expand_with_type_edges() checks both directions (get_type_parents
        and get_type_children), this single edge enables cross-language expansion:
          - TS component found → pulls in HTML template element (shows bindings)
          - HTML element found → pulls in TS component class (shows implementation)

        Only kebab-case selectors (containing '-') are matched — Angular component
        and directive selectors always use kebab-case per the style guide, so this
        safely skips attribute selectors like [myDirective].

        Returns number of edges created.
        """
        rows = self._conn.execute(
            """
            SELECT s.id, s.decorators
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE s.kind = 'class'
              AND s.decorators != '[]'
              AND f.language IN ('typescript', 'tsx', 'javascript')
            """
        ).fetchall()

        edges_created = 0
        for row in rows:
            symbol_id = row[0]
            decorators = json.loads(row[1] or "[]")

            for dec in decorators:
                m = self._SELECTOR_RE.search(dec)
                if not m:
                    continue
                selector = m.group(1).strip()
                # Only kebab-case selectors (must contain hyphen)
                if not selector or "-" not in selector:
                    continue

                # Find HTML symbols with this name in HTML files
                html_rows = self._conn.execute(
                    """
                    SELECT s.id FROM symbols s
                    JOIN files f ON s.file_id = f.id
                    WHERE s.name = ? AND f.language = 'html'
                    """,
                    (selector,),
                ).fetchall()

                for html_row in html_rows:
                    html_symbol_id = html_row[0]
                    # Avoid duplicate edges
                    exists = self._conn.execute(
                        """
                        SELECT 1 FROM type_edges
                        WHERE from_symbol_id = ? AND to_symbol_id = ?
                          AND edge_kind = 'angular_selector'
                        """,
                        (symbol_id, html_symbol_id),
                    ).fetchone()
                    if not exists:
                        self._conn.execute(
                            """
                            INSERT INTO type_edges
                              (from_symbol_id, to_type_name, edge_kind, to_symbol_id)
                            VALUES (?, ?, 'angular_selector', ?)
                            """,
                            (symbol_id, selector, html_symbol_id),
                        )
                        edges_created += 1

        if edges_created:
            self._conn.commit()
        return edges_created

    def get_top_symbols_for_file(self, file_id: int, limit: int = 5) -> list[int]:
        """Return top symbol ids for a file, prioritising classes then functions."""
        rows = self._conn.execute(
            """
            SELECT id FROM symbols
            WHERE file_id = ?
            ORDER BY CASE kind
                WHEN 'class'     THEN 1
                WHEN 'function'  THEN 2
                WHEN 'method'    THEN 3
                ELSE 4
            END, line_start
            LIMIT ?
            """,
            (file_id, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def find_file_by_path_fragment(self, fragment: str) -> list[int]:
        """
        Return file IDs whose rel_path contains `fragment`.
        Ordered shortest-path-first so the most specific match wins.
        Used by file_overview and config_lookup retrieval paths.
        """
        rows = self._conn.execute(
            "SELECT id FROM files WHERE rel_path LIKE ? ORDER BY LENGTH(rel_path)",
            (f"%{fragment}%",),
        ).fetchall()
        return [r[0] for r in rows]

    def get_all_symbols_for_file(self, file_id: int) -> list[int]:
        """
        Return ALL symbol IDs for a file ordered by kind then line number.
        Used by file_overview retrieval (no limit — context assembler manages budget).
        """
        rows = self._conn.execute(
            """
            SELECT id FROM symbols
            WHERE file_id = ?
            ORDER BY CASE kind
                WHEN 'module'    THEN 0
                WHEN 'class'     THEN 1
                WHEN 'interface' THEN 2
                WHEN 'function'  THEN 3
                WHEN 'method'    THEN 4
                ELSE 5
            END, line_start
            """,
            (file_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_module_and_readme_symbols(self, limit: int = 40) -> list[int]:
        """
        Return symbol IDs for a project-level overview query.

        Priority order:
          0 — README files (README.md, README.rst, README.txt)
          1 — Other markdown docs
          2 — Project manifests (package.json, pyproject.toml, Cargo.toml, go.mod)
          3 — Build/compiler configs (tsconfig.json, setup.py, setup.cfg)
          4 — Module-level code symbols (kind='module')

        This ensures "what does this project do?" gets the most descriptive
        files first regardless of whether a README exists.
        """
        rows = self._conn.execute(
            """
            SELECT s.id FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE
                -- Markdown docs
                f.language = 'markdown'
                -- Project manifests
                OR (f.rel_path LIKE '%package.json'
                    AND f.rel_path NOT LIKE '%node_modules%')
                OR f.rel_path LIKE '%pyproject.toml'
                OR f.rel_path LIKE '%Cargo.toml'
                OR (f.rel_path LIKE '%go.mod'
                    AND f.rel_path NOT LIKE '%vendor%')
                OR f.rel_path LIKE '%setup.py'
                OR f.rel_path LIKE '%setup.cfg'
                -- Compiler/build configs
                OR (f.rel_path LIKE '%tsconfig.json'
                    AND f.rel_path NOT LIKE '%node_modules%')
                -- Module-level code summaries
                OR s.kind = 'module'
            ORDER BY
                CASE
                    WHEN f.rel_path LIKE '%README%'    THEN 0
                    WHEN f.language = 'markdown'        THEN 1
                    WHEN f.rel_path LIKE '%package.json'   THEN 2
                    WHEN f.rel_path LIKE '%pyproject.toml' THEN 2
                    WHEN f.rel_path LIKE '%Cargo.toml'     THEN 2
                    WHEN f.rel_path LIKE '%go.mod'         THEN 2
                    WHEN f.rel_path LIKE '%setup.py'       THEN 3
                    WHEN f.rel_path LIKE '%tsconfig.json'  THEN 3
                    WHEN s.kind = 'module'                 THEN 4
                    ELSE 5
                END,
                LENGTH(f.rel_path)   -- shallower files (root-level) before nested
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Project context helpers (used by QueryPlanner)
    # ------------------------------------------------------------------

    def get_language_stats(self) -> list[tuple[str, int]]:
        """Return (language, file_count) pairs sorted by count descending."""
        rows = self._conn.execute(
            "SELECT language, COUNT(*) as cnt FROM files GROUP BY language ORDER BY cnt DESC"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_top_level_directories(self) -> list[str]:
        """
        Return meaningful top-level project directories from indexed file paths.

        For monorepos where most files share a common prefix (e.g. 'projects/'),
        drills one level deeper so 'projects/console/', 'projects/experience-studio/'
        are returned rather than just 'projects/'. Capped at 12 entries.
        """
        rows = self._conn.execute("SELECT DISTINCT rel_path FROM files").fetchall()
        # Collect first path component for all files
        first: set[str] = set()
        for (rel_path,) in rows:
            parts = rel_path.split("/")
            if len(parts) > 1:
                first.add(parts[0] + "/")
            else:
                first.add(rel_path)

        # Monorepo heuristic: if only 1-2 top-level dirs, drill one level deeper
        if len(first) <= 2:
            second: set[str] = set()
            for (rel_path,) in rows:
                parts = rel_path.split("/")
                if len(parts) > 2:
                    second.add(parts[0] + "/" + parts[1] + "/")
                elif len(parts) == 2:
                    second.add(parts[0] + "/")
            return sorted(second)[:12]

        return sorted(first)[:12]

    def get_files_by_import_path(self, pattern: str) -> list[int]:
        """
        Return source file_ids that have an import matching the given pattern.
        Used for blast_radius on path aliases like '@shared' that live in the
        imports table but not in symbol bodies.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT file_id FROM imports WHERE imported_from LIKE ?",
            (f"%{pattern}%",),
        ).fetchall()
        return [r[0] for r in rows]

    def count_files(self) -> int:
        """Return the total number of indexed files."""
        return int(self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])

    def count_symbols(self) -> int:
        """Return the total number of indexed symbols."""
        return int(self._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])

    def count_chunks(self) -> int:
        """Return the total number of indexed chunks."""
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # ------------------------------------------------------------------
    # Query telemetry (Task 6)
    # ------------------------------------------------------------------

    def insert_query_telemetry(
        self,
        query: str,
        intent: str,
        elapsed_ms: float,
        result_count: int,
        leg_sizes: dict[str, int] | None = None,
        *,
        expansion_used: bool | None = None,
        expansion_variants: int | None = None,
        expansion_elapsed_ms: float | None = None,
    ) -> int:
        """Insert one telemetry row. Returns row id."""
        import json

        cur = self._conn.execute(
            "INSERT INTO query_telemetry "
            "(query, intent, elapsed_ms, result_count, leg_sizes, "
            " expansion_used, expansion_variants, expansion_elapsed_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                query,
                intent,
                elapsed_ms,
                result_count,
                json.dumps(leg_sizes or {}),
                int(expansion_used) if expansion_used is not None else None,
                expansion_variants,
                expansion_elapsed_ms,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def get_recent_telemetry(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return most recent telemetry rows as list of dicts."""
        import json

        rows = self._conn.execute(
            "SELECT id, ts, query, intent, elapsed_ms, result_count, leg_sizes, thumbs_up "
            "FROM query_telemetry ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": int(row[0]),
                "ts": row[1],
                "query": row[2],
                "intent": row[3],
                "elapsed_ms": float(row[4]),
                "result_count": int(row[5]),
                "leg_sizes": json.loads(row[6] or "{}"),
                "thumbs_up": row[7],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Def-use chains (v2.2 data-flow analysis)
    # ------------------------------------------------------------------

    def insert_def_use_edges(self, edges: list[DefUseEdge]) -> None:
        """Bulk-insert def-use edges for a symbol."""
        if not edges:
            return
        self._conn.executemany(
            "INSERT INTO def_use_edges (symbol_id, var_name, def_line, use_line, edge_type) "
            "VALUES (?, ?, ?, ?, ?)",
            [(e.symbol_id, e.var_name, e.def_line, e.use_line, e.edge_type) for e in edges],
        )
        self._conn.commit()

    def get_data_flows(self, symbol_id: int) -> list[DefUseEdge]:
        """Return all def-use edges for a symbol."""
        from trelix.analysis.defuse import DefUseEdge

        rows = self._conn.execute(
            "SELECT symbol_id, var_name, def_line, use_line, edge_type "
            "FROM def_use_edges WHERE symbol_id = ? ORDER BY def_line",
            (symbol_id,),
        ).fetchall()
        return [
            DefUseEdge(
                symbol_id=int(row[0]),
                var_name=row[1],
                def_line=int(row[2]),
                use_line=int(row[3]),
                edge_type=row[4],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Taint flows (v2.2 semgrep integration)
    # ------------------------------------------------------------------

    def insert_taint_flows(self, flows: list[TaintFlow]) -> None:
        """Bulk-insert taint flows from a semgrep scan."""
        if not flows:
            return
        self._conn.executemany(
            "INSERT INTO taint_flows "
            "(source_file, source_line, sink_file, sink_line, rule_id, severity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (f.source_file, f.source_line, f.sink_file, f.sink_line, f.rule_id, f.severity)
                for f in flows
            ],
        )
        self._conn.commit()

    def get_taint_flows(
        self,
        severity: str | None = None,
        limit: int = 50,
    ) -> list[TaintFlow]:
        """Return taint flows, optionally filtered by severity."""
        from trelix.analysis.taint import TaintFlow

        if severity:
            rows = self._conn.execute(
                "SELECT source_file, source_line, sink_file, sink_line, rule_id, severity "
                "FROM taint_flows WHERE severity = ? ORDER BY severity DESC LIMIT ?",
                (severity, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT source_file, source_line, sink_file, sink_line, rule_id, severity "
                "FROM taint_flows ORDER BY severity DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            TaintFlow(
                source_file=row[0],
                source_line=int(row[1]),
                sink_file=row[2],
                sink_line=int(row[3]),
                rule_id=row[4],
                severity=row[5],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Agent session persistence (v2.8: persistent ReAct loop memory)
    # ------------------------------------------------------------------

    def upsert_agent_session(self, session_id: str, query: str) -> None:
        """Create the session row if absent, else bump last_active_at + query.

        Called once at the start of AgentLoop.run() for a given session_id.
        """
        with self._conn_lock:
            self._conn.execute(
                "INSERT INTO agent_sessions (id, query) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "last_active_at = datetime('now'), query = excluded.query",
                (session_id, query),
            )
            self._conn.commit()

    def insert_agent_turn(
        self,
        session_id: str,
        thought: str,
        action_type: str,
        action_arguments: dict[str, Any],
        observation_content: str,
        observation_source: str,
        observation_success: bool,
    ) -> int:
        """Append one turn and bump the session's turn_count. Returns the
        assigned turn_index.

        turn_index is computed atomically as MAX(turn_index)+1 for this
        session inside the same locked operation — never derived by the
        caller from a row-count snapshot taken earlier, since a persistence
        gap (e.g. a dropped turn) would make that snapshot stale and collide
        with an existing turn_index. agent_turns has a UNIQUE(session_id,
        turn_index) index as defense-in-depth: a residual race between two
        separate Database connections (this lock only guards one connection)
        raises IntegrityError, which the caller catches and logs, rather than
        silently persisting a duplicate/colliding row.
        """
        with self._conn_lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM agent_turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            turn_index = int(row[0])
            self._conn.execute(
                "INSERT INTO agent_turns "
                "(session_id, turn_index, thought, action_type, action_arguments, "
                " observation_content, observation_source, observation_success) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    turn_index,
                    thought,
                    action_type,
                    json.dumps(action_arguments),
                    observation_content,
                    observation_source,
                    int(observation_success),
                ),
            )
            self._conn.execute(
                "UPDATE agent_sessions SET turn_count = turn_count + 1, "
                "last_active_at = datetime('now') WHERE id = ?",
                (session_id,),
            )
            self._conn.commit()
            return turn_index

    def get_agent_turns(self, session_id: str) -> list[dict[str, Any]]:
        """Return all turns for a session ordered by turn_index, as plain dicts."""
        rows = self._conn.execute(
            "SELECT turn_index, thought, action_type, action_arguments, "
            "observation_content, observation_source, observation_success "
            "FROM agent_turns WHERE session_id = ? ORDER BY turn_index",
            (session_id,),
        ).fetchall()
        return [
            {
                "turn_index": int(row[0]),
                "thought": row[1],
                "action_type": row[2],
                "action_arguments": json.loads(row[3]),
                "observation_content": row[4],
                "observation_source": row[5],
                "observation_success": bool(row[6]),
            }
            for row in rows
        ]

    def list_agent_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return most-recently-active sessions with metadata, newest first."""
        rows = self._conn.execute(
            "SELECT id, created_at, last_active_at, query, turn_count "
            "FROM agent_sessions ORDER BY last_active_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "session_id": row[0],
                "created_at": row[1],
                "last_active_at": row[2],
                "query": row[3],
                "turn_count": int(row[4]),
            }
            for row in rows
        ]

    def delete_agent_session(self, session_id: str) -> bool:
        """Delete a session and its turns (cascade). Returns True if it existed."""
        cur = self._conn.execute("SELECT 1 FROM agent_sessions WHERE id = ?", (session_id,))
        existed = cur.fetchone() is not None
        self._conn.execute("DELETE FROM agent_sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        return existed

    def evict_stale_agent_sessions(self, max_age_seconds: float) -> int:
        """Delete sessions whose last_active_at is older than max_age_seconds.

        Returns the number of sessions deleted.
        """
        cur = self._conn.execute(
            "DELETE FROM agent_sessions WHERE (unixepoch('now') - unixepoch(last_active_at)) > ?",
            (max_age_seconds,),
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        if self._bm25_read_pool is not None:
            self._bm25_read_pool.close_all()
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # index_metadata helpers (v2.3 Plan E: embedding dimension guard)
    # ------------------------------------------------------------------

    def get_index_metadata(self, key: str) -> str | None:
        """Return a raw `index_metadata` value, or None if the key is absent."""
        row = self._conn.execute(
            "SELECT value FROM index_metadata WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def set_index_metadata(self, key: str, value: str) -> None:
        """Upsert a raw `index_metadata` value."""
        self._conn.execute(
            "INSERT INTO index_metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def delete_index_metadata(self, key: str) -> None:
        """Remove a key from `index_metadata`. A missing key is not an error."""
        self._conn.execute("DELETE FROM index_metadata WHERE key = ?", (key,))
        self._conn.commit()

    def get_index_metadata_with_prefix(self, prefix: str) -> dict[str, str]:
        """Return every key starting with `prefix`, with the prefix stripped.

        LIKE with an explicit ESCAPE, because a prefix containing `_` — and the
        provenance keys do — would otherwise have it treated as a single-character
        wildcard and match keys it should not.
        """
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._conn.execute(
            "SELECT key, value FROM index_metadata WHERE key LIKE ? ESCAPE '\\'",
            (escaped + "%",),
        ).fetchall()
        return {str(k)[len(prefix) :]: str(v) for k, v in rows}

    # The three below predate the generic helpers and are kept as the named API the
    # dimension guard and its tests already call. They delegate rather than duplicate
    # the SQL: the guard's whole job is to be trustworthy about a mismatch, and two
    # copies of an upsert are two places for it to drift.

    def get_embedding_dimension(self) -> int | None:
        """Return stored embedding dimension, or None if not yet recorded."""
        value = self.get_index_metadata("embedding_dimension")
        return None if value is None else int(value)

    def set_embedding_dimension(self, dimension: int) -> None:
        """Store the embedding dimension used for this index."""
        self.set_index_metadata("embedding_dimension", str(dimension))

    def delete_embedding_dimension_key(self) -> None:
        """Delete the stored embedding_dimension key from index_metadata."""
        self.delete_index_metadata("embedding_dimension")

    def get_all_file_rel_paths(self) -> list[str]:
        """Every indexed file's repo-relative path.

        Used to find indexed files the walk no longer yields. Returned as a list rather
        than a generator because the caller diffs it against a set — streaming would buy
        nothing and invites the connection being reused mid-iteration.
        """
        return [str(r[0]) for r in self._conn.execute("SELECT rel_path FROM files")]

    def clear_all_embeddings(self, vector_store: object | None = None) -> None:
        """
        Discard every stored embedding.

        Pass the `vector_store`: `chunk_embeddings` is a sqlite-vec VIRTUAL table, and
        this class never loads the sqlite-vec extension on its own connection. Without
        it, `DELETE FROM chunk_embeddings` raises `OperationalError: no such module:
        vec0` — which the previous implementation caught and discarded, making the
        method a guaranteed no-op on every install. `trelix migrate-vectors --reset`
        printed "Embeddings and dimension metadata cleared" while clearing none of them.

        The `vector_store` parameter follows `delete_file_by_path`, which takes one for
        the same reason: only the store's own connection can reach its virtual table.

        Called with no vector_store, this raises rather than pretending — a caller that
        cannot clear embeddings needs to know.
        """
        if vector_store is None:
            raise ValueError(
                "clear_all_embeddings() requires the vector_store: chunk_embeddings is "
                "a sqlite-vec virtual table and Database's connection cannot reach it"
            )
        # recreate() rather than a row delete, because a vec0 table's vector width is
        # fixed by its CREATE statement — deleting rows leaves a table that still
        # rejects vectors of a new dimension, which is the situation this is called to
        # recover from.
        vector_store.recreate()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Hydration queries  (chunk_id / symbol_id → full objects)
    # Used by Retriever, BM25, graph expansion to build SearchResult objects
    # ------------------------------------------------------------------

    def get_chunk_with_context(self, chunk_id: int) -> tuple[Chunk, Symbol, IndexedFile] | None:
        """
        Single JOIN query: chunk → symbol → file.
        Returns (Chunk, Symbol, IndexedFile) or None if not found.
        This is the primary hydration path called by Retriever._hydrate_chunk().

        Locked: reachable concurrently from the same sub-query
        ThreadPoolExecutor as bm25_search()'s hydration calls whenever a
        strategy's legs include both 'vector' and 'bm25' — same shared
        self._conn hazard as get_symbol_with_file().
        """
        with self._conn_lock:
            row = self._conn.execute(
                """
                SELECT
                    c.id         AS c_id,
                    c.symbol_id  AS c_symbol_id,
                    c.chunk_text AS c_chunk_text,
                    c.token_count AS c_token_count,

                    s.id              AS s_id,
                    s.file_id         AS s_file_id,
                    s.name            AS s_name,
                    s.qualified_name  AS s_qualified_name,
                    s.kind            AS s_kind,
                    s.line_start      AS s_line_start,
                    s.line_end        AS s_line_end,
                    s.signature       AS s_signature,
                    s.docstring       AS s_docstring,
                    s.context_summary AS s_context_summary,
                    s.decorators      AS s_decorators,
                    s.is_public       AS s_is_public,
                    s.parent_id       AS s_parent_id,
                    s.body            AS s_body,

                    f.id         AS f_id,
                    f.path       AS f_path,
                    f.rel_path   AS f_rel_path,
                    f.language   AS f_language,
                    f.hash       AS f_hash,
                    f.size_bytes AS f_size_bytes
                FROM chunks c
                JOIN symbols s ON c.symbol_id = s.id
                JOIN files   f ON s.file_id   = f.id
                WHERE c.id = ?
                """,
                (chunk_id,),
            ).fetchone()

        if not row:
            return None

        return self._row_to_hydrated(row)

    def get_symbol_with_file(self, symbol_id: int) -> tuple[Symbol, IndexedFile] | None:
        """
        Load a symbol and its file in one query.
        Used by graph expansion and grep search hydration.

        Locked: self._conn is a single shared connection with no internal
        thread-safety for concurrent statement execution. bm25_search()'s
        retrieval-layer wrapper calls this after drawing symbol_ids from the
        (thread-safe) read pool, so this method IS reachable concurrently —
        confirmed by a CI flake (SymbolKind(None)/InterfaceError under
        concurrent load) before this lock was added.
        """
        with self._conn_lock:
            row = self._conn.execute(
                """
                SELECT
                    s.id              AS s_id,
                    s.file_id         AS s_file_id,
                    s.name            AS s_name,
                    s.qualified_name  AS s_qualified_name,
                    s.kind            AS s_kind,
                    s.line_start      AS s_line_start,
                    s.line_end        AS s_line_end,
                    s.signature       AS s_signature,
                    s.docstring       AS s_docstring,
                    s.context_summary AS s_context_summary,
                    s.decorators      AS s_decorators,
                    s.is_public       AS s_is_public,
                    s.parent_id       AS s_parent_id,
                    s.body            AS s_body,

                    f.id         AS f_id,
                    f.path       AS f_path,
                    f.rel_path   AS f_rel_path,
                    f.language   AS f_language,
                    f.hash       AS f_hash,
                    f.size_bytes AS f_size_bytes
                FROM symbols s
                JOIN files f ON s.file_id = f.id
                WHERE s.id = ?
                """,
                (symbol_id,),
            ).fetchone()

        if not row:
            return None

        symbol = Symbol(
            id=row["s_id"],
            file_id=row["s_file_id"],
            name=row["s_name"],
            qualified_name=row["s_qualified_name"],
            kind=SymbolKind(row["s_kind"]),
            line_start=row["s_line_start"],
            line_end=row["s_line_end"],
            signature=row["s_signature"],
            docstring=row["s_docstring"],
            context_summary=row["s_context_summary"],
            decorators=json.loads(row["s_decorators"] or "[]"),
            is_public=bool(row["s_is_public"]),
            parent_id=row["s_parent_id"],
            body=row["s_body"],
        )
        file = IndexedFile(
            id=row["f_id"],
            path=row["f_path"],
            rel_path=row["f_rel_path"],
            language=Language(row["f_language"]),
            hash=row["f_hash"],
            size_bytes=row["f_size_bytes"],
        )
        return symbol, file

    def get_first_chunk_for_symbol(self, symbol_id: int) -> Chunk | None:
        """
        Return the first (and usually only) chunk for a symbol.
        Used when we have a symbol_id from BM25/graph and need a Chunk for SearchResult.

        Locked for the same reason as get_symbol_with_file() — reachable
        concurrently via bm25_search()'s hydration path when the read pool
        is enabled.
        """
        with self._conn_lock:
            row = self._conn.execute(
                "SELECT * FROM chunks WHERE symbol_id = ? LIMIT 1",
                (symbol_id,),
            ).fetchone()
        if not row:
            return None
        return Chunk(
            id=row["id"],
            symbol_id=row["symbol_id"],
            chunk_text=row["chunk_text"],
            token_count=row["token_count"],
        )

    def get_chunk_by_id(self, chunk_id: int) -> Chunk | None:
        """
        Return a single chunk by id. Used by sparse_search() to hydrate
        SparseStore hits.

        Locked for the same reason as get_symbol_with_file() — sparse_search
        runs inside the same sub-query ThreadPoolExecutor as the bm25/grep
        legs, all sharing this connection.
        """
        with self._conn_lock:
            row = self._conn.execute(
                "SELECT id, symbol_id, chunk_text, token_count FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return Chunk(
            id=int(row[0]),
            symbol_id=int(row[1]),
            chunk_text=row[2],
            token_count=int(row[3]),
        )

    def _row_to_hydrated(self, row: sqlite3.Row) -> tuple[Chunk, Symbol, IndexedFile]:
        """Convert a JOIN row into (Chunk, Symbol, IndexedFile)."""
        chunk = Chunk(
            id=row["c_id"],
            symbol_id=row["c_symbol_id"],
            chunk_text=row["c_chunk_text"],
            token_count=row["c_token_count"],
        )
        symbol = Symbol(
            id=row["s_id"],
            file_id=row["s_file_id"],
            name=row["s_name"],
            qualified_name=row["s_qualified_name"],
            kind=SymbolKind(row["s_kind"]),
            line_start=row["s_line_start"],
            line_end=row["s_line_end"],
            signature=row["s_signature"],
            docstring=row["s_docstring"],
            context_summary=row["s_context_summary"],
            decorators=json.loads(row["s_decorators"] or "[]"),
            is_public=bool(row["s_is_public"]),
            parent_id=row["s_parent_id"],
            body=row["s_body"],
        )
        file = IndexedFile(
            id=row["f_id"],
            path=row["f_path"],
            rel_path=row["f_rel_path"],
            language=Language(row["f_language"]),
            hash=row["f_hash"],
            size_bytes=row["f_size_bytes"],
        )
        return chunk, symbol, file
