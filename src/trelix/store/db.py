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
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from trelix.core.models import (
    CallEdge,
    Chunk,
    ImportEdge,
    IndexedFile,
    Language,
    Symbol,
    SymbolKind,
    TypeEdge,
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
    decorators      TEXT    NOT NULL DEFAULT '[]',
    is_public       INTEGER NOT NULL DEFAULT 1,
    parent_id       INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    body            TEXT    NOT NULL DEFAULT ''
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
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    caller_id   INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    callee_name TEXT    NOT NULL,
    callee_id   INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    line        INTEGER NOT NULL
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

-- FTS5 for BM25 keyword search over symbol content
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name,
    qualified_name,
    docstring,
    body,
    content='symbols',
    content_rowid='id',
    tokenize='porter ascii'
);

-- Keep FTS index in sync with symbols table
-- NOTE: triggers reference the 4-column FTS schema; decorators are not FTS-indexed
-- (they appear in chunk_text so vector + BM25 search already covers them)
CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbols_fts(rowid, name, qualified_name, docstring, body)
    VALUES (new.id, new.name, new.qualified_name, new.docstring, new.body);
END;

CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, qualified_name, docstring, body)
    VALUES ('delete', old.id, old.name, old.qualified_name, old.docstring, old.body);
END;

CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, qualified_name, docstring, body)
    VALUES ('delete', old.id, old.name, old.qualified_name, old.docstring, old.body);
    INSERT INTO symbols_fts(rowid, name, qualified_name, docstring, body)
    VALUES (new.id, new.name, new.qualified_name, new.docstring, new.body);
END;
"""


class Database:
    """
    Thin wrapper around sqlite3 with typed methods for each table.

    Usage:
        db = Database(Path(".trelix/.codeindex/index.db"))
        file_id = db.upsert_file(indexed_file)
        symbol_id = db.insert_symbol(symbol)
        db.close()
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_ddl()
        self._apply_migrations()

    def _apply_ddl(self) -> None:
        self._conn.executescript(DDL)
        self._conn.commit()

    def _apply_migrations(self) -> None:
        """Incremental schema migrations — safe to run on existing DBs."""
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

        # type_edges table is idempotent (CREATE TABLE IF NOT EXISTS in DDL)

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

    def get_file_hash(self, rel_path: str) -> Optional[str]:
        """Return stored hash for a file, or None if not indexed yet."""
        row = self._conn.execute(
            "SELECT hash FROM files WHERE rel_path = ?", (rel_path,)
        ).fetchone()
        return row["hash"] if row else None

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
        return row[0]

    def delete_file_symbols(self, file_id: int) -> None:
        """Remove all symbols (and cascaded data) for a file before re-indexing."""
        self._conn.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
        self._conn.execute("DELETE FROM imports WHERE file_id = ?", (file_id,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Symbols
    # ------------------------------------------------------------------

    def insert_symbol(self, symbol: Symbol) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO symbols
              (file_id, name, qualified_name, kind, line_start, line_end,
               signature, docstring, decorators, is_public, parent_id, body)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol.file_id, symbol.name, symbol.qualified_name,
                symbol.kind.value, symbol.line_start, symbol.line_end,
                symbol.signature, symbol.docstring,
                json.dumps(symbol.decorators), int(symbol.is_public),
                symbol.parent_id, symbol.body,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_symbol_by_name(self, name: str) -> list[Symbol]:
        rows = self._conn.execute(
            "SELECT * FROM symbols WHERE name = ?", (name,)
        ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def get_symbols_for_file(self, file_id: int) -> list[Symbol]:
        rows = self._conn.execute(
            "SELECT * FROM symbols WHERE file_id = ?", (file_id,)
        ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def insert_call_edges(self, edges: list[CallEdge]) -> None:
        self._conn.executemany(
            "INSERT INTO calls (caller_id, callee_name, callee_id, line) VALUES (?, ?, ?, ?)",
            [(e.caller_id, e.callee_name, e.callee_id, e.line) for e in edges],
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
        rows = self._conn.execute(
            "SELECT * FROM imports WHERE file_id = ?", (file_id,)
        ).fetchall()
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

    # ------------------------------------------------------------------
    # BM25 search (FTS5)
    # ------------------------------------------------------------------

    def bm25_search(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        """
        Full-text search over symbols using SQLite FTS5 BM25.
        Returns list of (symbol_id, rank) sorted by relevance.
        Lower rank = more relevant in SQLite FTS5 (it's negative BM25).
        """
        rows = self._conn.execute(
            """
            SELECT rowid, rank
            FROM symbols_fts
            WHERE symbols_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
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

    # ------------------------------------------------------------------
    # Cross-file call resolution (second pass after all files indexed)
    # ------------------------------------------------------------------

    def resolve_cross_file_calls(self) -> int:
        """
        Update callee_id for all unresolved call edges where callee_name
        matches a known symbol in the DB. Called once after all files indexed.

        Handles cross-file calls where the callee file was indexed after the
        caller file, leaving callee_id = NULL at index time.
        Returns number of newly resolved edges.
        """
        cursor = self._conn.execute(
            """
            UPDATE calls
            SET callee_id = (
                SELECT id FROM symbols
                WHERE name = calls.callee_name
                LIMIT 1
            )
            WHERE callee_id IS NULL
              AND EXISTS (
                SELECT 1 FROM symbols WHERE name = calls.callee_name
              )
            """
        )
        self._conn.commit()
        return cursor.rowcount

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
            r[0]: r[1]
            for r in self._conn.execute("SELECT id, rel_path FROM files").fetchall()
        }

        # Build path lookup: normalized path (no ext, no common prefix) → file_id
        # Stores multiple variants per file for fuzzy matching
        path_lookup: dict[str, int] = {}
        for fid, rel in all_files.items():
            no_ext = rel
            for ext in (".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".go", ".rs", ".java"):
                if no_ext.endswith(ext):
                    no_ext = no_ext[: -len(ext)]
                    break
            # Handle Go package files and Rust module files
            no_ext = no_ext.removesuffix("/mod")    # Rust: store/db/mod.rs → store/db
            no_ext = no_ext.removesuffix("/index")  # JS: components/Button/index.ts → components/Button

            # Full path without extension
            path_lookup[no_ext] = fid

            # Without common source root prefixes (src/, lib/, app/)
            for prefix in ("src/", "lib/", "app/"):
                if no_ext.startswith(prefix):
                    path_lookup[no_ext[len(prefix):]] = fid
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
    ) -> "int | None":
        """
        Resolve a single import path to a file_id using language-aware heuristics.
        Handles: Python dotted paths, JS/TS relative, Go/Java slash paths, Rust :: paths.
        """
        # --- Relative imports (JS/TS, Python in-package) ---
        if module_path.startswith("."):
            importer_rel = all_files.get(importer_file_id, "")
            importer_dir = "/".join(importer_rel.split("/")[:-1])
            joined = (importer_dir + "/" + module_path) if importer_dir else module_path
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
            "java.", "javax.", "android.", "kotlin.",       # Java/Kotlin stdlib
            "std::", "core::", "alloc::",                   # Rust std
            "react", "vue", "@angular", "@types/",          # JS/TS frameworks
            "lodash", "axios", "express", "next", "nuxt",
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
                    candidate = candidate[len(prefix):]
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
            "SELECT imported_file_id FROM imports WHERE file_id = ? AND imported_file_id IS NOT NULL",
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

    # ------------------------------------------------------------------
    # Type edges
    # ------------------------------------------------------------------

    def insert_type_edges(self, edges: list[TypeEdge]) -> None:
        self._conn.executemany(
            "INSERT INTO type_edges (from_symbol_id, to_type_name, edge_kind, to_symbol_id) VALUES (?, ?, ?, ?)",
            [(e.from_symbol_id, e.to_type_name, e.edge_kind, e.to_symbol_id) for e in edges],
        )

    def get_type_parents(self, symbol_id: int) -> list[int]:
        """Return symbol ids this symbol inherits/implements (outgoing type edges)."""
        rows = self._conn.execute(
            "SELECT to_symbol_id FROM type_edges WHERE from_symbol_id = ? AND to_symbol_id IS NOT NULL",
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

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Hydration queries  (chunk_id / symbol_id → full objects)
    # Used by Retriever, BM25, graph expansion to build SearchResult objects
    # ------------------------------------------------------------------

    def get_chunk_with_context(
        self, chunk_id: int
    ) -> "tuple[Chunk, Symbol, IndexedFile] | None":
        """
        Single JOIN query: chunk → symbol → file.
        Returns (Chunk, Symbol, IndexedFile) or None if not found.
        This is the primary hydration path called by Retriever._hydrate_chunk().
        """
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

    def get_symbol_with_file(
        self, symbol_id: int
    ) -> "tuple[Symbol, IndexedFile] | None":
        """
        Load a symbol and its file in one query.
        Used by graph expansion and grep search hydration.
        """
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

    def get_first_chunk_for_symbol(self, symbol_id: int) -> "Chunk | None":
        """
        Return the first (and usually only) chunk for a symbol.
        Used when we have a symbol_id from BM25/graph and need a Chunk for SearchResult.
        """
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

    def _row_to_hydrated(
        self, row: sqlite3.Row
    ) -> "tuple[Chunk, Symbol, IndexedFile]":
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
