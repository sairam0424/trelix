"""
trelix-mcp MCP Resources — URI-addressable index data.

MCP Resources (spec: modelcontextprotocol.io/docs/concepts/resources) are
application-controlled passive data sources, distinguished from Tools (model-controlled
callable functions). Resources expose the trelix index as navigable URI-addressable content.

Resource types implemented:
  Direct:   trelix://index/stats               — aggregate index statistics (hint only;
                                                  no repo_path available in direct form)
  Template: trelix://repo/{repo_path}/manifest  — list of indexed files
  Template: trelix://repo/{repo_path}/symbols/{qualified_name} — symbol source

All handlers return JSON strings. Never raise — return {"error": "..."} on failure.
All log output goes to stderr via the module logger only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from trelix.core.config import IndexConfig
from trelix.store.db import Database

_log = logging.getLogger("trelix_mcp.resources")


def get_index_stats(repo_path: str) -> str:
    """Return aggregate statistics for the trelix index at *repo_path* as JSON.

    Args:
        repo_path: Absolute path to the repository root.

    Returns:
        JSON string with keys ``symbol_count``, ``file_count``, ``chunk_count``,
        ``repo_path``, or ``{"error": "..."}`` on any failure.
    """
    if not Path(repo_path).is_dir():
        return json.dumps({"error": f"repo_path is not a directory: {repo_path}"})
    try:
        config = IndexConfig(repo_path=repo_path)
        db = Database(config.db_path_absolute)
        row = db._conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM symbols) AS symbol_count,
                (SELECT COUNT(*) FROM files)   AS file_count,
                (SELECT COUNT(*) FROM chunks)  AS chunk_count
            """
        ).fetchone()
        if row is None:
            return json.dumps({"error": "Index not found. Run: trelix index <repo>"})
        return json.dumps(
            {
                "symbol_count": int(row[0]),
                "file_count": int(row[1]),
                "chunk_count": int(row[2]),
                "repo_path": repo_path,
            }
        )
    except Exception as exc:  # noqa: BLE001
        _log.debug("get_index_stats failed: %s", exc)
        return json.dumps({"error": str(exc), "hint": "Run: trelix index <repo>"})


def get_repo_manifest(repo_path: str) -> str:
    """Return the list of indexed files with language and symbol count as JSON.

    Args:
        repo_path: Absolute path to the repository root.

    Returns:
        JSON string with keys ``repo_path``, ``file_count``, ``files`` (list of
        ``{path, language, symbol_count}`` dicts), or ``{"error": "..."}`` on failure.
    """
    if not Path(repo_path).is_dir():
        return json.dumps({"error": f"repo_path is not a directory: {repo_path}"})
    try:
        config = IndexConfig(repo_path=repo_path)
        db = Database(config.db_path_absolute)
        rows = db._conn.execute(
            """
            SELECT f.rel_path, f.language, COUNT(s.id) AS symbol_count
            FROM files f
            LEFT JOIN symbols s ON s.file_id = f.id
            GROUP BY f.id
            ORDER BY f.rel_path
            LIMIT 500
            """
        ).fetchall()
        return json.dumps(
            {
                "repo_path": repo_path,
                "file_count": len(rows),
                "files": [
                    {
                        "path": row[0],
                        "language": row[1],
                        "symbol_count": int(row[2]),
                    }
                    for row in rows
                ],
            }
        )
    except Exception as exc:  # noqa: BLE001
        _log.debug("get_repo_manifest failed: %s", exc)
        return json.dumps({"error": str(exc)})


def get_symbol_source(repo_path: str, qualified_name: str) -> str:
    """Return full source body of a symbol by qualified name as JSON.

    Args:
        repo_path: Absolute path to the repository root.
        qualified_name: Fully-qualified symbol name, e.g. ``AuthService.login``.

    Returns:
        JSON string with keys ``qualified_name``, ``kind``, ``signature``, ``body``,
        or ``{"error": "..."}`` if the symbol is not found or any failure occurs.
    """
    if not Path(repo_path).is_dir():
        return json.dumps({"error": f"repo_path is not a directory: {repo_path}"})
    try:
        config = IndexConfig(repo_path=repo_path)
        db = Database(config.db_path_absolute)
        short_name = qualified_name.split(".")[-1]
        symbols = db.get_symbol_by_name(short_name)
        exact = [s for s in symbols if s.qualified_name == qualified_name]
        sym = (exact or symbols[:1] or [None])[0]
        if sym is None:
            return json.dumps({"error": f"Symbol '{qualified_name}' not found"})
        kind_val = sym.kind.value if hasattr(sym.kind, "value") else str(sym.kind)
        return json.dumps(
            {
                "qualified_name": sym.qualified_name,
                "kind": kind_val,
                "signature": sym.signature,
                "body": sym.body,
            }
        )
    except Exception as exc:  # noqa: BLE001
        _log.debug("get_symbol_source failed: %s", exc)
        return json.dumps({"error": str(exc)})
