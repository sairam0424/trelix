# trelix Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full Knowledge Graph layer to trelix that unifies the existing call/import/type edges into a traversable Code Property Graph, adds LLM-extracted semantic concept nodes, detects architectural communities, exposes graph-aware retrieval as a first-class search leg, and serves the graph via REST API and MCP tools — turning trelix into the first open-source hybrid vector+graph code intelligence system.

**Architecture:**  
A new `trelix/graph/` module wraps trelix's existing SQLite edge tables (calls, imports, type_edges) into a unified NetworkX MultiDiGraph (the `CodeGraph`), adds LLM-extracted semantic concept nodes inspired by the knowledge_graph repo's approach, runs Girvan-Newman community detection to cluster architectural modules, and exposes the graph as a 4th retrieval leg in the Retriever pipeline. The REST API gains `/graph` endpoints and the MCP server gains `graph_search` + `get_community` tools. Visualization is served as an interactive Pyvis HTML file.

**Tech Stack:**  
Python 3.11+, networkx ≥3.3.0 (already a dependency), pyvis ≥0.3.2 (new optional dep), existing trelix LLM client for concept extraction, SQLite for graph persistence, fastapi (existing serve extra) for graph endpoints.

## Global Constraints

- Python ≥ 3.11 — same as trelix core
- Version stays `1.1.0` during development; bump to `3.0.0` only at release cut
- All new code in `src/trelix/graph/` — no changes to existing module boundaries except additive
- All tests in `tests/unit/test_graph_*.py` — mirror source layout
- `networkx` is already in core deps — no new hard deps; `pyvis` goes in optional `[graph-viz]` extra
- New optional extra `[graph]` covers: `pyvis>=0.3.2`, `seaborn>=0.13.0`
- Branch: `feature/knowledge-graph` → PR → `develop`
- Every task ends with a passing `pytest tests/unit/test_graph_*.py -q` run
- `ruff check src/ tests/ && ruff format src/ tests/ && mypy src/trelix/graph/ --strict` must pass before PR
- No changes to `trelix/retrieval/graph.py` (existing call-graph expansion) — the new `trelix/graph/` module is additive and separate

---

## Phase 1 — Core Graph Model (Tasks 1–3)

### Task 1: `CodeGraph` — unified MultiDiGraph over existing edge tables

**Files:**
- Create: `src/trelix/graph/__init__.py`
- Create: `src/trelix/graph/code_graph.py`
- Create: `tests/unit/test_graph_code_graph.py`

**Interfaces:**
- Consumes: `trelix.store.db.Database`, `trelix.core.models.Symbol`, `trelix.core.models.IndexedFile`
- Produces:
  - `CodeGraph(db: Database)` — constructor builds the NX graph
  - `CodeGraph.nx: nx.MultiDiGraph` — the underlying NetworkX graph (read-only property)
  - `CodeGraph.node_count: int`
  - `CodeGraph.edge_count: int`
  - `CodeGraph.neighbors(symbol_id: int) -> list[int]`
  - `CodeGraph.shortest_path(src: int, dst: int) -> list[int] | None`
  - `CodeGraph.subgraph(symbol_ids: list[int]) -> nx.MultiDiGraph`

**Node schema** (each symbol_id is a node):
```python
{
    "type": "symbol",          # always "symbol" for now
    "name": str,               # symbol.name
    "qualified_name": str,     # symbol.qualified_name
    "kind": str,               # symbol.kind.value
    "file": str,               # file.rel_path
    "language": str,           # file.language.value
    "community": int | None,   # filled later by Task 3
}
```

**Edge schema** (directed, labeled):
```
CALLS        — caller_id → callee_id         (from calls table)
IMPORTS      — file_node → imported_file_node (from imports table, file-level)
EXTENDS      — child_id → parent_id          (edge_kind "extends")
IMPLEMENTS   — child_id → interface_id       (edge_kind "implements")
TRAIT_IMPL   — struct_id → trait_id          (edge_kind "trait_impl")
EMBEDDED     — struct_id → embedded_id       (edge_kind "embedded")
```

- [ ] **Step 1: Create `src/trelix/graph/__init__.py`**

```python
"""trelix Knowledge Graph — unified code property graph over indexed codebases."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_graph_code_graph.py`:

```python
"""Tests for CodeGraph — unified MultiDiGraph over trelix edge tables."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from trelix.core.config import IndexConfig
from trelix.core.models import (
    CallEdge,
    ImportEdge,
    IndexedFile,
    Language,
    Symbol,
    SymbolKind,
    TypeEdge,
)
from trelix.graph.code_graph import CodeGraph
from trelix.store.db import Database


def _make_db(tmp_path: Path) -> Database:
    db = Database(str(tmp_path / "index.db"))
    return db


def _insert_file(db: Database, rel_path: str, lang: Language = Language.PYTHON) -> int:
    f = IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=lang,
        hash="abc",
        size_bytes=100,
    )
    return db.upsert_file(f)


def _insert_symbol(
    db: Database,
    file_id: int,
    name: str,
    kind: SymbolKind = SymbolKind.FUNCTION,
    parent_id: int | None = None,
) -> int:
    s = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=name,
        kind=kind,
        line_start=1,
        line_end=10,
        signature=f"def {name}()",
        body=f"def {name}(): pass",
    )
    return db.insert_symbol(s)


class TestCodeGraphConstruction:
    def test_empty_db_builds_empty_graph(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        cg = CodeGraph(db)
        assert cg.node_count == 0
        assert cg.edge_count == 0

    def test_nodes_from_symbols(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "auth.py")
        _insert_symbol(db, fid, "login")
        _insert_symbol(db, fid, "logout")
        cg = CodeGraph(db)
        assert cg.node_count == 2

    def test_call_edge_added(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "auth.py")
        sid1 = _insert_symbol(db, fid, "login")
        sid2 = _insert_symbol(db, fid, "hash_password")
        db.insert_call_edges([
            CallEdge(caller_id=sid1, callee_name="hash_password", callee_id=sid2, line=5)
        ])
        cg = CodeGraph(db)
        # CALLS edge: login → hash_password
        assert cg.edge_count >= 1
        neighbors = cg.neighbors(sid1)
        assert sid2 in neighbors

    def test_type_edge_extends(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "models.py")
        sid1 = _insert_symbol(db, fid, "AdminUser", SymbolKind.CLASS)
        sid2 = _insert_symbol(db, fid, "User", SymbolKind.CLASS)
        db.insert_type_edges([
            TypeEdge(
                from_symbol_id=sid1,
                to_type_name="User",
                edge_kind="extends",
                to_symbol_id=sid2,
            )
        ])
        cg = CodeGraph(db)
        neighbors = cg.neighbors(sid1)
        assert sid2 in neighbors

    def test_node_attributes(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "auth.py")
        sid = _insert_symbol(db, fid, "login")
        cg = CodeGraph(db)
        attrs = cg.nx.nodes[sid]
        assert attrs["name"] == "login"
        assert attrs["kind"] == SymbolKind.FUNCTION.value
        assert attrs["file"] == "auth.py"
        assert attrs["community"] is None

    def test_shortest_path_connected(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "app.py")
        sid1 = _insert_symbol(db, fid, "handle_request")
        sid2 = _insert_symbol(db, fid, "authenticate")
        sid3 = _insert_symbol(db, fid, "hash_password")
        db.insert_call_edges([
            CallEdge(caller_id=sid1, callee_name="authenticate", callee_id=sid2, line=3),
            CallEdge(caller_id=sid2, callee_name="hash_password", callee_id=sid3, line=7),
        ])
        cg = CodeGraph(db)
        path = cg.shortest_path(sid1, sid3)
        assert path is not None
        assert path[0] == sid1
        assert path[-1] == sid3

    def test_shortest_path_disconnected(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "app.py")
        sid1 = _insert_symbol(db, fid, "fn_a")
        sid2 = _insert_symbol(db, fid, "fn_b")
        cg = CodeGraph(db)
        assert cg.shortest_path(sid1, sid2) is None

    def test_subgraph(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "app.py")
        sid1 = _insert_symbol(db, fid, "fn_a")
        sid2 = _insert_symbol(db, fid, "fn_b")
        sid3 = _insert_symbol(db, fid, "fn_c")
        cg = CodeGraph(db)
        sg = cg.subgraph([sid1, sid2])
        assert sid1 in sg.nodes
        assert sid2 in sg.nodes
        assert sid3 not in sg.nodes
```

- [ ] **Step 3: Run test to confirm it fails**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
.venv/bin/python -m pytest tests/unit/test_graph_code_graph.py -v --tb=short 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'trelix.graph'`

- [ ] **Step 4: Implement `src/trelix/graph/code_graph.py`**

```python
"""Unified Code Property Graph over trelix's SQLite edge tables."""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.code_graph")

_EDGE_KINDS_TO_LABEL: dict[str, str] = {
    "extends": "EXTENDS",
    "implements": "IMPLEMENTS",
    "trait_impl": "TRAIT_IMPL",
    "embedded": "EMBEDDED",
    "angular_selector": "ANGULAR_SELECTOR",
}


class CodeGraph:
    """
    Unified MultiDiGraph over trelix's SQLite edge tables.

    Nodes  = symbol IDs (int), with attrs: name, qualified_name, kind, file, language, community
    Edges  = directed, labeled: CALLS | IMPORTS | EXTENDS | IMPLEMENTS | TRAIT_IMPL | EMBEDDED
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._build()

    def _build(self) -> None:
        """Load all symbols as nodes and all edges from the DB."""
        # --- Nodes: all symbols ---
        for symbol, file in self._db.iter_all_symbols_with_files():
            self._g.add_node(
                symbol.id,
                type="symbol",
                name=symbol.name,
                qualified_name=symbol.qualified_name,
                kind=symbol.kind.value,
                file=file.rel_path,
                language=file.language.value,
                community=None,
            )

        # --- CALLS edges ---
        for caller_id, callee_id in self._db.iter_resolved_calls():
            if caller_id in self._g and callee_id in self._g:
                self._g.add_edge(caller_id, callee_id, label="CALLS")

        # --- IMPORTS edges (file-level → map to representative file-module node) ---
        for file_id, imported_file_id in self._db.iter_resolved_imports():
            # Add file nodes if not present as symbols (files may have no symbols)
            for nid in (file_id, imported_file_id):
                if nid not in self._g:
                    fi = self._db.get_file_by_id(nid)
                    if fi is not None:
                        self._g.add_node(
                            nid,
                            type="file",
                            name=fi.rel_path,
                            qualified_name=fi.rel_path,
                            kind="file",
                            file=fi.rel_path,
                            language=fi.language.value,
                            community=None,
                        )
            if file_id in self._g and imported_file_id in self._g:
                self._g.add_edge(file_id, imported_file_id, label="IMPORTS")

        # --- TYPE edges (EXTENDS / IMPLEMENTS / TRAIT_IMPL / EMBEDDED) ---
        for from_id, edge_kind, to_id in self._db.iter_resolved_type_edges():
            label = _EDGE_KINDS_TO_LABEL.get(edge_kind, "TYPE_REL")
            if from_id in self._g and to_id in self._g:
                self._g.add_edge(from_id, to_id, label=label)

        logger.debug(
            "CodeGraph built: %d nodes, %d edges",
            self._g.number_of_nodes(),
            self._g.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def nx(self) -> nx.MultiDiGraph:
        return self._g

    @property
    def node_count(self) -> int:
        return self._g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._g.number_of_edges()

    def neighbors(self, symbol_id: int) -> list[int]:
        """Return all adjacent node IDs (successors + predecessors)."""
        if symbol_id not in self._g:
            return []
        succs = list(self._g.successors(symbol_id))
        preds = list(self._g.predecessors(symbol_id))
        return list(dict.fromkeys(succs + preds))

    def shortest_path(self, src: int, dst: int) -> list[int] | None:
        """Return node-ID list of shortest undirected path, or None."""
        try:
            return list(nx.shortest_path(self._g.to_undirected(), src, dst))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def subgraph(self, symbol_ids: list[int]) -> nx.MultiDiGraph:
        """Return induced subgraph over the given node IDs."""
        return self._g.subgraph(symbol_ids).copy()

    def get_node_attrs(self, symbol_id: int) -> dict[str, Any]:
        """Return node attribute dict, or empty dict if not found."""
        return dict(self._g.nodes.get(symbol_id, {}))
```

- [ ] **Step 5: Add required DB iterator methods to `src/trelix/store/db.py`**

Open `src/trelix/store/db.py` and add these four methods (after `resolve_cross_file_type_edges`):

```python
def iter_all_symbols_with_files(
    self,
) -> list[tuple[Symbol, IndexedFile]]:
    """Yield (Symbol, IndexedFile) for every symbol in the DB."""
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
    result = []
    for row in rows:
        sym = Symbol(
            id=row[0], file_id=row[1], name=row[2], qualified_name=row[3],
            kind=SymbolKind(row[4]), line_start=row[5], line_end=row[6],
            signature=row[7] or "", docstring=row[8], context_summary=row[9],
            decorators=json.loads(row[10] or "[]"),
            is_public=bool(row[11]), parent_id=row[12], body=row[13] or "",
        )
        fi = IndexedFile(
            id=row[14], path=row[15], rel_path=row[16],
            language=Language(row[17]), hash=row[18], size_bytes=row[19],
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
        "SELECT from_symbol_id, edge_kind, to_symbol_id FROM type_edges WHERE to_symbol_id IS NOT NULL"
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]

def get_file_by_id(self, file_id: int) -> IndexedFile | None:
    """Fetch a file record by primary key."""
    row = self._conn.execute(
        "SELECT id, path, rel_path, language, hash, size_bytes FROM files WHERE id = ?",
        (file_id,),
    ).fetchone()
    if row is None:
        return None
    return IndexedFile(
        id=row[0], path=row[1], rel_path=row[2],
        language=Language(row[3]), hash=row[4], size_bytes=row[5],
    )
```

Also verify `import json` is at the top of `db.py` (it should already be there since decorators are stored as JSON).

- [ ] **Step 6: Run tests — they should pass**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
.venv/bin/python -m pytest tests/unit/test_graph_code_graph.py -v --tb=short
```

Expected: all 7 tests PASS.

- [ ] **Step 7: Lint and type-check**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
.venv/bin/ruff check src/trelix/graph/ tests/unit/test_graph_code_graph.py
.venv/bin/ruff format src/trelix/graph/ tests/unit/test_graph_code_graph.py
.venv/bin/python -m mypy src/trelix/graph/code_graph.py --strict --ignore-missing-imports
```

Fix any issues. Common fixes: `-> None` return types on `_build`, `list[...]` return type annotations.

- [ ] **Step 8: Commit**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
git checkout -b feature/knowledge-graph
git add src/trelix/graph/ src/trelix/store/db.py tests/unit/test_graph_code_graph.py
git commit -m "feat(graph): add CodeGraph — unified MultiDiGraph over call/import/type edges

- New trelix/graph/ module with CodeGraph(db) that loads symbols as nodes
- Edges: CALLS, IMPORTS, EXTENDS, IMPLEMENTS, TRAIT_IMPL, EMBEDDED
- Node attrs: name, qualified_name, kind, file, language, community
- Methods: neighbors(), shortest_path(), subgraph(), get_node_attrs()
- DB: add iter_all_symbols_with_files, iter_resolved_calls,
  iter_resolved_imports, iter_resolved_type_edges, get_file_by_id"
```

---

### Task 2: `GraphPersistence` — serialize/deserialize CodeGraph to SQLite

**Files:**
- Create: `src/trelix/graph/persistence.py`
- Modify: `src/trelix/graph/__init__.py` (add export)
- Create: `tests/unit/test_graph_persistence.py`

**Interfaces:**
- Consumes: `CodeGraph` from Task 1, `trelix.store.db.Database`
- Produces:
  - `save_graph_metadata(db: Database, cg: CodeGraph) -> None` — writes community assignments back to DB
  - `load_graph_metadata(db: Database, cg: CodeGraph) -> None` — reads community assignments from DB
  - SQL table: `graph_metadata (symbol_id INTEGER PRIMARY KEY, community INTEGER, centrality REAL, node_type TEXT)`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_graph_persistence.py`:

```python
"""Tests for graph persistence — save/load community assignments."""
from __future__ import annotations

from pathlib import Path

from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.code_graph import CodeGraph
from trelix.graph.persistence import load_graph_metadata, save_graph_metadata
from trelix.store.db import Database


def _make_db_with_symbol(tmp_path: Path) -> tuple[Database, int]:
    db = Database(str(tmp_path / "index.db"))
    f = IndexedFile(path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="x", size_bytes=10)
    fid = db.upsert_file(f)
    s = Symbol(
        file_id=fid, name="fn", qualified_name="fn", kind=SymbolKind.FUNCTION,
        line_start=1, line_end=5, signature="def fn()", body="def fn(): pass",
    )
    sid = db.insert_symbol(s)
    return db, sid


class TestGraphPersistence:
    def test_save_then_load_community(self, tmp_path: Path) -> None:
        db, sid = _make_db_with_symbol(tmp_path)
        cg = CodeGraph(db)
        # Manually set community
        cg.nx.nodes[sid]["community"] = 42
        save_graph_metadata(db, cg)

        # Fresh graph — community should be None before load
        cg2 = CodeGraph(db)
        assert cg2.nx.nodes[sid]["community"] is None

        # After load, community should be restored
        load_graph_metadata(db, cg2)
        assert cg2.nx.nodes[sid]["community"] == 42

    def test_save_idempotent(self, tmp_path: Path) -> None:
        db, sid = _make_db_with_symbol(tmp_path)
        cg = CodeGraph(db)
        cg.nx.nodes[sid]["community"] = 1
        save_graph_metadata(db, cg)
        cg.nx.nodes[sid]["community"] = 2
        save_graph_metadata(db, cg)  # Should overwrite, not error

        cg2 = CodeGraph(db)
        load_graph_metadata(db, cg2)
        assert cg2.nx.nodes[sid]["community"] == 2

    def test_missing_nodes_skipped_gracefully(self, tmp_path: Path) -> None:
        db, sid = _make_db_with_symbol(tmp_path)
        cg = CodeGraph(db)
        # Save with one community
        cg.nx.nodes[sid]["community"] = 5
        save_graph_metadata(db, cg)

        # New graph with no nodes — load should not crash
        import networkx as nx
        cg_empty = CodeGraph.__new__(CodeGraph)
        cg_empty._g = nx.MultiDiGraph()
        cg_empty._db = db
        load_graph_metadata(db, cg_empty)  # no crash
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_persistence.py -v --tb=short 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'trelix.graph.persistence'`

- [ ] **Step 3: Implement `src/trelix/graph/persistence.py`**

```python
"""Persist CodeGraph metadata (community, centrality) back to SQLite."""
from __future__ import annotations

import logging

import networkx as nx

from trelix.graph.code_graph import CodeGraph
from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.persistence")

_DDL = """
CREATE TABLE IF NOT EXISTS graph_metadata (
    symbol_id INTEGER PRIMARY KEY,
    community INTEGER,
    centrality REAL DEFAULT 0.0,
    node_type TEXT DEFAULT 'symbol'
);
"""


def _ensure_table(db: Database) -> None:
    db._conn.execute(_DDL)
    db._conn.commit()


def save_graph_metadata(db: Database, cg: CodeGraph) -> None:
    """Write community and centrality for all nodes into graph_metadata table."""
    _ensure_table(db)

    centrality: dict[int, float] = {}
    try:
        centrality = nx.degree_centrality(cg.nx)
    except Exception:
        pass

    rows = [
        (
            node_id,
            attrs.get("community"),
            centrality.get(node_id, 0.0),
            attrs.get("type", "symbol"),
        )
        for node_id, attrs in cg.nx.nodes(data=True)
    ]
    db._conn.executemany(
        """
        INSERT INTO graph_metadata (symbol_id, community, centrality, node_type)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(symbol_id) DO UPDATE SET
            community = excluded.community,
            centrality = excluded.centrality,
            node_type = excluded.node_type
        """,
        rows,
    )
    db._conn.commit()
    logger.debug("Saved graph metadata for %d nodes", len(rows))


def load_graph_metadata(db: Database, cg: CodeGraph) -> None:
    """Read community and centrality from graph_metadata and set node attrs."""
    _ensure_table(db)
    rows = db._conn.execute(
        "SELECT symbol_id, community, centrality FROM graph_metadata"
    ).fetchall()
    for symbol_id, community, centrality in rows:
        if symbol_id in cg.nx:
            cg.nx.nodes[symbol_id]["community"] = community
            cg.nx.nodes[symbol_id]["centrality"] = centrality
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_persistence.py -v --tb=short
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trelix/graph/persistence.py tests/unit/test_graph_persistence.py
git commit -m "feat(graph): add GraphPersistence — save/load community + centrality to SQLite

- graph_metadata table: symbol_id, community, centrality, node_type
- save_graph_metadata(): writes degree centrality + community to DB
- load_graph_metadata(): restores metadata onto existing CodeGraph"
```

---

### Task 3: `CommunityDetector` — Girvan-Newman architectural clustering

**Files:**
- Create: `src/trelix/graph/community.py`
- Create: `tests/unit/test_graph_community.py`

**Interfaces:**
- Consumes: `CodeGraph` from Task 1
- Produces:
  - `detect_communities(cg: CodeGraph, algorithm: str = "louvain") -> dict[int, int]`
    — returns `{symbol_id: community_id}` mapping
  - `assign_communities(cg: CodeGraph, communities: dict[int, int]) -> None`
    — sets `cg.nx.nodes[sid]["community"] = community_id` for all nodes
  - `get_community_summary(cg: CodeGraph) -> list[dict]`
    — returns list of `{community_id, size, top_files, top_symbols, label}`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_graph_community.py`:

```python
"""Tests for community detection on CodeGraph."""
from __future__ import annotations

from pathlib import Path

from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import assign_communities, detect_communities, get_community_summary
from trelix.store.db import Database


def _build_clustered_db(tmp_path: Path) -> tuple[Database, list[int]]:
    """Build a DB with two clearly separated clusters."""
    db = Database(str(tmp_path / "index.db"))

    def _file(name: str) -> int:
        f = IndexedFile(path=f"/r/{name}", rel_path=name, language=Language.PYTHON, hash="x", size_bytes=10)
        return db.upsert_file(f)

    def _sym(fid: int, name: str) -> int:
        s = Symbol(file_id=fid, name=name, qualified_name=name, kind=SymbolKind.FUNCTION,
                   line_start=1, line_end=5, signature=f"def {name}()", body="pass")
        return db.insert_symbol(s)

    # Cluster A: auth module (3 symbols, densely connected)
    fid_a = _file("auth.py")
    a1 = _sym(fid_a, "login")
    a2 = _sym(fid_a, "logout")
    a3 = _sym(fid_a, "hash_password")

    # Cluster B: db module (3 symbols, densely connected)
    fid_b = _file("db.py")
    b1 = _sym(fid_b, "query")
    b2 = _sym(fid_b, "insert")
    b3 = _sym(fid_b, "connect")

    # Dense intra-cluster edges
    db.insert_call_edges([
        CallEdge(caller_id=a1, callee_name="hash_password", callee_id=a3, line=2),
        CallEdge(caller_id=a2, callee_name="hash_password", callee_id=a3, line=3),
        CallEdge(caller_id=b1, callee_name="connect", callee_id=b3, line=2),
        CallEdge(caller_id=b2, callee_name="connect", callee_id=b3, line=3),
    ])

    return db, [a1, a2, a3, b1, b2, b3]


class TestCommunityDetection:
    def test_returns_mapping_for_all_nodes(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assert isinstance(mapping, dict)
        for sid in sids:
            assert sid in mapping

    def test_community_ids_are_ints(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        for cid in mapping.values():
            assert isinstance(cid, int)

    def test_assign_sets_node_attrs(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assign_communities(cg, mapping)
        for sid in sids:
            assert cg.nx.nodes[sid]["community"] is not None

    def test_community_summary_structure(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assign_communities(cg, mapping)
        summary = get_community_summary(cg)
        assert isinstance(summary, list)
        assert len(summary) >= 1
        for item in summary:
            assert "community_id" in item
            assert "size" in item
            assert "top_files" in item
            assert "top_symbols" in item

    def test_empty_graph(self, tmp_path: Path) -> None:
        db = Database(str(tmp_path / "index.db"))
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assert mapping == {}
        summary = get_community_summary(cg)
        assert summary == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_community.py -v --tb=short 2>&1 | head -15
```

- [ ] **Step 3: Implement `src/trelix/graph/community.py`**

```python
"""Community detection for CodeGraph — Louvain (fast) or Girvan-Newman (quality)."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import networkx as nx

from trelix.graph.code_graph import CodeGraph

logger = logging.getLogger("trelix.graph.community")


def detect_communities(
    cg: CodeGraph,
    algorithm: str = "louvain",
) -> dict[int, int]:
    """
    Detect communities and return {node_id: community_id}.

    algorithm:
        "louvain"       — fast, good quality, O(n log n). Preferred for >500 nodes.
        "girvan_newman" — betweenness-based, high quality, O(n³). Use for small graphs.
        "label_prop"    — very fast, approximate. Use for >10k nodes.
    """
    if cg.node_count == 0:
        return {}

    # Work on undirected version for community detection
    G_undirected = cg.nx.to_undirected()

    # Remove isolated nodes for community detection (they'd each be own community)
    G_connected = nx.Graph(
        (u, v) for u, v, _ in G_undirected.edges(data=True)
    )
    # Re-add isolated nodes to ensure they get community -1
    for node in G_undirected.nodes():
        if node not in G_connected:
            G_connected.add_node(node)

    mapping: dict[int, int] = {}

    try:
        if algorithm == "louvain":
            communities = nx.community.louvain_communities(G_connected, seed=42)
        elif algorithm == "girvan_newman":
            gen = nx.community.girvan_newman(G_connected)
            # Take 3 levels of splitting for reasonable granularity
            try:
                next(gen)
                next(gen)
                communities_tuple = next(gen)
                communities = [set(c) for c in communities_tuple]
            except StopIteration:
                communities = [set(G_connected.nodes())]
        elif algorithm == "label_prop":
            communities = list(nx.community.label_propagation_communities(G_connected))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm!r}")

        for community_id, members in enumerate(communities):
            for node_id in members:
                mapping[int(node_id)] = community_id

    except Exception as exc:
        logger.warning("Community detection failed (%s), assigning all to 0: %s", algorithm, exc)
        for node_id in cg.nx.nodes():
            mapping[int(node_id)] = 0

    return mapping


def assign_communities(cg: CodeGraph, communities: dict[int, int]) -> None:
    """Write community IDs back into CodeGraph node attributes."""
    for node_id, community_id in communities.items():
        if node_id in cg.nx:
            cg.nx.nodes[node_id]["community"] = community_id


def get_community_summary(cg: CodeGraph) -> list[dict[str, Any]]:
    """Return summary info per detected community."""
    if cg.node_count == 0:
        return []

    by_community: dict[int, list[int]] = defaultdict(list)
    for node_id, attrs in cg.nx.nodes(data=True):
        cid = attrs.get("community")
        if cid is not None:
            by_community[int(cid)].append(node_id)

    if not by_community:
        return []

    summaries = []
    for cid, members in sorted(by_community.items()):
        # Top files by member count
        file_counts: Counter[str] = Counter()
        symbol_names: list[str] = []
        for mid in members:
            attrs = cg.nx.nodes.get(mid, {})
            f = attrs.get("file", "")
            if f:
                file_counts[f] += 1
            name = attrs.get("qualified_name") or attrs.get("name", "")
            if name:
                symbol_names.append(name)

        summaries.append(
            {
                "community_id": cid,
                "size": len(members),
                "top_files": [f for f, _ in file_counts.most_common(5)],
                "top_symbols": symbol_names[:10],
                "label": f"community_{cid}",
            }
        )

    return summaries
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_community.py -v --tb=short
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trelix/graph/community.py tests/unit/test_graph_community.py
git commit -m "feat(graph): add CommunityDetector — Louvain/Girvan-Newman community detection

- detect_communities(cg, algorithm) → {node_id: community_id}
- Supports 'louvain' (default, fast), 'girvan_newman' (quality), 'label_prop' (huge graphs)
- assign_communities() sets node attrs in-place
- get_community_summary() returns top files/symbols per community"
```

---

## Phase 2 — Semantic Concept Layer (Tasks 4–5)

### Task 4: `ConceptExtractor` — LLM-powered semantic node enrichment

**Files:**
- Create: `src/trelix/graph/concepts.py`
- Create: `tests/unit/test_graph_concepts.py`

**Interfaces:**
- Consumes: `trelix.llm.factory.build_chat_client`, `trelix.core.config.LLMConfig`, `trelix.store.db.Database`
- Produces:
  - `SemanticConcept` dataclass: `{name: str, category: str, importance: int, source_symbol_ids: list[int]}`
  - `ConceptExtractor(llm_config: LLMConfig)` class
  - `ConceptExtractor.extract_from_symbols(symbols: list[Symbol], max_symbols: int = 20) -> list[SemanticConcept]`
  - `ConceptExtractor.extract_from_file_summary(summary: str, file_id: int) -> list[SemanticConcept]`
  - SQL table: `graph_concepts (id, name, category, importance, source_symbol_ids TEXT)`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_graph_concepts.py`:

```python
"""Tests for ConceptExtractor — LLM semantic concept extraction."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import LLMConfig
from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.concepts import ConceptExtractor, SemanticConcept, save_concepts, load_concepts
from trelix.store.db import Database


def _make_symbols() -> list[Symbol]:
    return [
        Symbol(
            id=1, file_id=1, name="authenticate_user",
            qualified_name="AuthService.authenticate_user",
            kind=SymbolKind.METHOD, line_start=10, line_end=30,
            signature="def authenticate_user(self, token: str) -> User",
            body="def authenticate_user(self, token: str) -> User:\n    ...",
        ),
        Symbol(
            id=2, file_id=1, name="refresh_token",
            qualified_name="AuthService.refresh_token",
            kind=SymbolKind.METHOD, line_start=35, line_end=55,
            signature="def refresh_token(self, token: str) -> str",
            body="def refresh_token(self, token: str) -> str:\n    ...",
        ),
    ]


class TestConceptExtractor:
    def test_extract_returns_list_of_semantic_concepts(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = (
            '[{"entity": "JWT authentication", "importance": 5, "category": "security"}, '
            '{"entity": "token refresh", "importance": 4, "category": "concept"}]'
        )
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_symbols(_make_symbols())

        assert isinstance(concepts, list)
        assert len(concepts) == 2
        assert all(isinstance(c, SemanticConcept) for c in concepts)
        assert concepts[0].name == "jwt authentication"  # lowercased
        assert concepts[0].importance == 5
        assert concepts[0].category == "security"

    def test_extract_tolerates_malformed_json(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = "not valid json at all"
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_symbols(_make_symbols())
        # Should return empty list, not crash
        assert concepts == []

    def test_extract_tolerates_llm_exception(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("LLM unavailable")
        cfg = LLMConfig()
        with patch("trelix.graph.concepts.build_chat_client", return_value=mock_client):
            extractor = ConceptExtractor(cfg)
            concepts = extractor.extract_from_symbols(_make_symbols())
        assert concepts == []

    def test_save_and_load_concepts(self, tmp_path: Path) -> None:
        db = Database(str(tmp_path / "index.db"))
        # Insert a dummy file and symbol so DB is valid
        from trelix.core.models import IndexedFile, Language
        fid = db.upsert_file(IndexedFile(path="/r/a.py", rel_path="a.py",
                                         language=Language.PYTHON, hash="x", size_bytes=10))
        concepts = [
            SemanticConcept(name="jwt auth", category="security", importance=5, source_symbol_ids=[1, 2]),
            SemanticConcept(name="token refresh", category="concept", importance=3, source_symbol_ids=[2]),
        ]
        save_concepts(db, concepts)
        loaded = load_concepts(db)
        assert len(loaded) == 2
        names = {c.name for c in loaded}
        assert "jwt auth" in names
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_concepts.py -v --tb=short 2>&1 | head -15
```

- [ ] **Step 3: Implement `src/trelix/graph/concepts.py`**

```python
"""LLM-powered semantic concept extraction for the knowledge graph."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from trelix.core.config import LLMConfig
from trelix.core.models import Symbol
from trelix.llm.factory import build_chat_client
from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.concepts")

_EXTRACT_SYS = """\
You are a code analysis assistant. Extract the key technical concepts from the
provided code symbols. Focus on architectural concepts, design patterns, and
domain logic — NOT individual variable names.

Return ONLY a valid JSON array (no markdown, no prose) with objects:
[{"entity": "concept name", "importance": 1-5, "category": "security|concept|pattern|architecture|domain|misc"}]

Rules:
- Normalize names to lowercase
- importance 5 = core architectural concept, 1 = trivial implementation detail
- Maximum 8 concepts per batch
- Return [] if no meaningful concepts found
"""

_DDL_CONCEPTS = """
CREATE TABLE IF NOT EXISTS graph_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'concept',
    importance INTEGER NOT NULL DEFAULT 3,
    source_symbol_ids TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_graph_concepts_name ON graph_concepts(name);
"""


@dataclass
class SemanticConcept:
    name: str
    category: str
    importance: int
    source_symbol_ids: list[int] = field(default_factory=list)


class ConceptExtractor:
    """Extract semantic concepts from code symbols using an LLM."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._client = build_chat_client(llm_config)

    def extract_from_symbols(
        self,
        symbols: list[Symbol],
        max_symbols: int = 20,
    ) -> list[SemanticConcept]:
        """Extract concepts from a batch of symbols. Returns [] on any failure."""
        if not symbols:
            return []

        # Build compact symbol representation
        batch = symbols[:max_symbols]
        code_context = "\n\n".join(
            f"# {s.qualified_name} ({s.kind.value})\n{s.signature}\n{s.body[:300]}"
            for s in batch
        )
        source_ids = [s.id for s in batch if s.id is not None]

        try:
            response = self._client.chat(
                system=_EXTRACT_SYS,
                user=f"Extract concepts from these code symbols:\n\n{code_context}",
            )
            parsed = json.loads(response)
            if not isinstance(parsed, list):
                return []
            result = []
            for item in parsed:
                if not isinstance(item, dict) or "entity" not in item:
                    continue
                result.append(
                    SemanticConcept(
                        name=str(item["entity"]).lower().strip(),
                        category=str(item.get("category", "concept")),
                        importance=int(item.get("importance", 3)),
                        source_symbol_ids=source_ids,
                    )
                )
            return result
        except Exception as exc:
            logger.debug("ConceptExtractor failed: %s", exc)
            return []

    def extract_from_file_summary(
        self,
        summary: str,
        file_id: int,
    ) -> list[SemanticConcept]:
        """Extract concepts from a RAPTOR-style file summary."""
        if not summary.strip():
            return []
        try:
            response = self._client.chat(
                system=_EXTRACT_SYS,
                user=f"Extract concepts from this file summary:\n\n{summary}",
            )
            parsed = json.loads(response)
            if not isinstance(parsed, list):
                return []
            return [
                SemanticConcept(
                    name=str(item["entity"]).lower().strip(),
                    category=str(item.get("category", "concept")),
                    importance=int(item.get("importance", 3)),
                    source_symbol_ids=[file_id],
                )
                for item in parsed
                if isinstance(item, dict) and "entity" in item
            ]
        except Exception as exc:
            logger.debug("ConceptExtractor.extract_from_file_summary failed: %s", exc)
            return []


def save_concepts(db: Database, concepts: list[SemanticConcept]) -> None:
    """Persist concepts to graph_concepts table (upsert by name)."""
    db._conn.executescript(_DDL_CONCEPTS)
    db._conn.executemany(
        """
        INSERT INTO graph_concepts (name, category, importance, source_symbol_ids)
        VALUES (?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        [
            (c.name, c.category, c.importance, json.dumps(c.source_symbol_ids))
            for c in concepts
        ],
    )
    db._conn.commit()


def load_concepts(db: Database) -> list[SemanticConcept]:
    """Load all persisted concepts from DB."""
    try:
        rows = db._conn.execute(
            "SELECT name, category, importance, source_symbol_ids FROM graph_concepts"
        ).fetchall()
        return [
            SemanticConcept(
                name=row[0],
                category=row[1],
                importance=row[2],
                source_symbol_ids=json.loads(row[3] or "[]"),
            )
            for row in rows
        ]
    except Exception:
        return []
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_concepts.py -v --tb=short
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trelix/graph/concepts.py tests/unit/test_graph_concepts.py
git commit -m "feat(graph): add ConceptExtractor — LLM semantic concept extraction

- SemanticConcept dataclass: name, category, importance, source_symbol_ids
- ConceptExtractor.extract_from_symbols() with LLM, crash-safe (returns [] on failure)
- extract_from_file_summary() for RAPTOR file-level summaries
- save_concepts() / load_concepts() via graph_concepts SQLite table"
```

---

### Task 5: `GraphBuilder` — orchestrate full graph construction pipeline

**Files:**
- Create: `src/trelix/graph/builder.py`
- Create: `tests/unit/test_graph_builder.py`
- Modify: `src/trelix/graph/__init__.py` — export `GraphBuilder`, `CodeGraph`, `detect_communities`

**Interfaces:**
- Consumes: all of Tasks 1–4
- Produces:
  - `GraphBuilder(config: IndexConfig)` class
  - `GraphBuilder.build(extract_concepts: bool = False) -> GraphBuildResult`
  - `GraphBuildResult` dataclass: `{code_graph: CodeGraph, community_count: int, node_count: int, edge_count: int, concept_count: int, elapsed_seconds: float}`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_graph_builder.py`:

```python
"""Tests for GraphBuilder — full graph construction pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import IndexConfig
from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.builder import GraphBuildResult, GraphBuilder
from trelix.store.db import Database


def _populated_repo(tmp_path: Path) -> Path:
    """Create a minimal indexed repo at tmp_path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".trelix").mkdir()
    db = Database(str(repo / ".trelix" / "index.db"))

    fid = db.upsert_file(IndexedFile(
        path=str(repo / "auth.py"), rel_path="auth.py",
        language=Language.PYTHON, hash="x", size_bytes=100,
    ))
    sid1 = db.insert_symbol(Symbol(
        file_id=fid, name="login", qualified_name="login",
        kind=SymbolKind.FUNCTION, line_start=1, line_end=10,
        signature="def login()", body="def login(): pass",
    ))
    sid2 = db.insert_symbol(Symbol(
        file_id=fid, name="hash_password", qualified_name="hash_password",
        kind=SymbolKind.FUNCTION, line_start=12, line_end=20,
        signature="def hash_password()", body="def hash_password(): pass",
    ))
    db.insert_call_edges([
        CallEdge(caller_id=sid1, callee_name="hash_password", callee_id=sid2, line=5)
    ])
    return repo


class TestGraphBuilder:
    def test_build_returns_result(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        builder = GraphBuilder(config)
        result = builder.build(extract_concepts=False)
        assert isinstance(result, GraphBuildResult)
        assert result.node_count >= 2
        assert result.edge_count >= 1
        assert result.community_count >= 1
        assert result.concept_count == 0  # no concept extraction

    def test_build_with_concepts_disabled_does_not_call_llm(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        builder = GraphBuilder(config)
        with patch("trelix.graph.builder.ConceptExtractor") as MockCE:
            result = builder.build(extract_concepts=False)
        MockCE.assert_not_called()
        assert result.concept_count == 0

    def test_build_assigns_communities(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        builder = GraphBuilder(config)
        result = builder.build(extract_concepts=False)
        # All nodes should have community set
        for _, attrs in result.code_graph.nx.nodes(data=True):
            assert attrs.get("community") is not None

    def test_elapsed_seconds_positive(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        result = GraphBuilder(config).build(extract_concepts=False)
        assert result.elapsed_seconds >= 0.0
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_builder.py -v --tb=short 2>&1 | head -15
```

- [ ] **Step 3: Implement `src/trelix/graph/builder.py`**

```python
"""GraphBuilder — orchestrates the full knowledge graph construction pipeline."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from trelix.core.config import IndexConfig
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import assign_communities, detect_communities, get_community_summary
from trelix.graph.concepts import ConceptExtractor, save_concepts
from trelix.graph.persistence import save_graph_metadata
from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.builder")


@dataclass
class GraphBuildResult:
    code_graph: CodeGraph
    community_count: int
    node_count: int
    edge_count: int
    concept_count: int
    elapsed_seconds: float
    community_summary: list[dict] = field(default_factory=list)


class GraphBuilder:
    """
    Orchestrates the full knowledge graph construction:
      1. Build CodeGraph from existing DB edges
      2. Run community detection
      3. (Optional) Extract semantic concepts via LLM
      4. Persist graph metadata to DB
    """

    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        self._db = Database(config.db_path_absolute)

    def build(self, extract_concepts: bool = False) -> GraphBuildResult:
        start = time.perf_counter()
        logger.info("Building CodeGraph from %s", self._config.repo_path)

        # Step 1: build graph
        cg = CodeGraph(self._db)
        logger.info("CodeGraph: %d nodes, %d edges", cg.node_count, cg.edge_count)

        # Step 2: community detection
        communities = detect_communities(cg, algorithm="louvain")
        assign_communities(cg, communities)
        community_count = len(set(communities.values())) if communities else 0
        community_summary = get_community_summary(cg)
        logger.info("Detected %d communities", community_count)

        # Step 3: persist metadata
        save_graph_metadata(self._db, cg)

        # Step 4: optional concept extraction
        concept_count = 0
        if extract_concepts:
            symbols_with_files = self._db.iter_all_symbols_with_files()
            symbols = [s for s, _ in symbols_with_files]
            if symbols:
                extractor = ConceptExtractor(self._config.llm)
                # Batch into groups of 20 (LLM context limit)
                concepts = []
                for i in range(0, min(len(symbols), 200), 20):
                    batch = symbols[i : i + 20]
                    concepts.extend(extractor.extract_from_symbols(batch))
                if concepts:
                    save_concepts(self._db, concepts)
                    concept_count = len(concepts)
                    logger.info("Extracted %d semantic concepts", concept_count)

        elapsed = time.perf_counter() - start
        logger.info("Graph built in %.2fs", elapsed)

        return GraphBuildResult(
            code_graph=cg,
            community_count=community_count,
            node_count=cg.node_count,
            edge_count=cg.edge_count,
            concept_count=concept_count,
            elapsed_seconds=elapsed,
            community_summary=community_summary,
        )
```

- [ ] **Step 4: Update `src/trelix/graph/__init__.py`**

```python
"""trelix Knowledge Graph — unified code property graph over indexed codebases."""
from trelix.graph.builder import GraphBuildResult, GraphBuilder
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import assign_communities, detect_communities, get_community_summary
from trelix.graph.concepts import ConceptExtractor, SemanticConcept, load_concepts, save_concepts
from trelix.graph.persistence import load_graph_metadata, save_graph_metadata

__all__ = [
    "CodeGraph",
    "GraphBuilder",
    "GraphBuildResult",
    "ConceptExtractor",
    "SemanticConcept",
    "detect_communities",
    "assign_communities",
    "get_community_summary",
    "save_graph_metadata",
    "load_graph_metadata",
    "save_concepts",
    "load_concepts",
]
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_builder.py -v --tb=short
```

Expected: all 4 PASS.

- [ ] **Step 6: Run full graph test suite**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_code_graph.py tests/unit/test_graph_persistence.py tests/unit/test_graph_community.py tests/unit/test_graph_concepts.py tests/unit/test_graph_builder.py -v --tb=short
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/trelix/graph/__init__.py src/trelix/graph/builder.py tests/unit/test_graph_builder.py
git commit -m "feat(graph): add GraphBuilder — orchestrate full graph construction pipeline

- GraphBuilder(config).build(extract_concepts=False) → GraphBuildResult
- Runs: CodeGraph → community detection → graph_metadata persist → optional LLM concepts
- GraphBuildResult: node_count, edge_count, community_count, concept_count, elapsed_seconds"
```

---

## Phase 3 — Graph Visualization (Task 6)

### Task 6: `GraphVisualizer` — Pyvis interactive HTML export

**Files:**
- Create: `src/trelix/graph/visualizer.py`
- Create: `tests/unit/test_graph_visualizer.py`
- Modify: `pyproject.toml` — add `[graph-viz]` optional extra

**Interfaces:**
- Consumes: `CodeGraph` from Task 1, `GraphBuildResult` from Task 5
- Produces:
  - `GraphVisualizer.export_html(cg: CodeGraph, output_path: str, max_nodes: int = 500) -> str`
    — returns absolute path to written HTML file
  - `GraphVisualizer.export_community_report(result: GraphBuildResult, output_path: str) -> str`
    — returns absolute path to JSON community report

- [ ] **Step 1: Add `[graph-viz]` extra to `pyproject.toml`**

In `pyproject.toml`, find the `[project.optional-dependencies]` section and add:

```toml
graph-viz = ["pyvis>=0.3.2", "seaborn>=0.13.0"]
graph = ["pyvis>=0.3.2", "seaborn>=0.13.0"]
```

- [ ] **Step 2: Write failing test**

Create `tests/unit/test_graph_visualizer.py`:

```python
"""Tests for GraphVisualizer — Pyvis HTML export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.builder import GraphBuildResult, GraphBuilder
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import assign_communities, detect_communities
from trelix.graph.visualizer import GraphVisualizer
from trelix.store.db import Database


def _build_simple_graph(tmp_path: Path) -> tuple[Database, CodeGraph]:
    db = Database(str(tmp_path / "index.db"))
    fid = db.upsert_file(IndexedFile(path="/r/a.py", rel_path="a.py",
                                      language=Language.PYTHON, hash="x", size_bytes=10))
    sid1 = db.insert_symbol(Symbol(file_id=fid, name="fn_a", qualified_name="fn_a",
                                    kind=SymbolKind.FUNCTION, line_start=1, line_end=5,
                                    signature="def fn_a()", body="def fn_a(): pass"))
    sid2 = db.insert_symbol(Symbol(file_id=fid, name="fn_b", qualified_name="fn_b",
                                    kind=SymbolKind.FUNCTION, line_start=7, line_end=12,
                                    signature="def fn_b()", body="def fn_b(): pass"))
    db.insert_call_edges([CallEdge(caller_id=sid1, callee_name="fn_b", callee_id=sid2, line=3)])
    cg = CodeGraph(db)
    mapping = detect_communities(cg)
    assign_communities(cg, mapping)
    return db, cg


class TestGraphVisualizer:
    def test_export_html_creates_file(self, tmp_path: Path) -> None:
        _, cg = _build_simple_graph(tmp_path)
        out = str(tmp_path / "graph.html")
        viz = GraphVisualizer()
        result_path = viz.export_html(cg, out)
        assert Path(result_path).exists()
        content = Path(result_path).read_text()
        assert "<html" in content.lower()

    def test_export_html_max_nodes_truncates(self, tmp_path: Path) -> None:
        _, cg = _build_simple_graph(tmp_path)
        out = str(tmp_path / "graph_small.html")
        viz = GraphVisualizer()
        # max_nodes=1 should not crash even when graph has 2 nodes
        result_path = viz.export_html(cg, out, max_nodes=1)
        assert Path(result_path).exists()

    def test_export_html_empty_graph(self, tmp_path: Path) -> None:
        db = Database(str(tmp_path / "index.db"))
        cg = CodeGraph(db)
        out = str(tmp_path / "empty.html")
        viz = GraphVisualizer()
        result_path = viz.export_html(cg, out)
        assert Path(result_path).exists()

    def test_export_community_report_json(self, tmp_path: Path) -> None:
        _, cg = _build_simple_graph(tmp_path)
        from trelix.graph.community import get_community_summary
        import time
        result = GraphBuildResult(
            code_graph=cg,
            community_count=1,
            node_count=cg.node_count,
            edge_count=cg.edge_count,
            concept_count=0,
            elapsed_seconds=0.1,
            community_summary=get_community_summary(cg),
        )
        out = str(tmp_path / "report.json")
        viz = GraphVisualizer()
        report_path = viz.export_community_report(result, out)
        assert Path(report_path).exists()
        data = json.loads(Path(report_path).read_text())
        assert "node_count" in data
        assert "communities" in data
```

- [ ] **Step 3: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_visualizer.py -v --tb=short 2>&1 | head -15
```

- [ ] **Step 4: Implement `src/trelix/graph/visualizer.py`**

```python
"""Pyvis-based interactive visualization for CodeGraph."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from trelix.graph.builder import GraphBuildResult
from trelix.graph.code_graph import CodeGraph

logger = logging.getLogger("trelix.graph.visualizer")

# Community color palette (pastel fill, darker stroke pairs)
_PALETTE = [
    "#a5d8ff", "#d0bfff", "#b2f2bb", "#ffd8a8",
    "#c3fae8", "#ffc9c9", "#ffe8cc", "#e5dbff",
    "#d3f9d8", "#fff3bf",
]

_EDGE_COLORS: dict[str, str] = {
    "CALLS": "#4a9eed",
    "IMPORTS": "#8b5cf6",
    "EXTENDS": "#22c55e",
    "IMPLEMENTS": "#06b6d4",
    "TRAIT_IMPL": "#f59e0b",
    "EMBEDDED": "#ef4444",
}


class GraphVisualizer:
    """Export CodeGraph to interactive Pyvis HTML or JSON community report."""

    def export_html(
        self,
        cg: CodeGraph,
        output_path: str,
        max_nodes: int = 500,
    ) -> str:
        """
        Generate an interactive Pyvis HTML visualization.

        Nodes are colored by community. Edges are colored by type.
        If the graph has more than max_nodes nodes, sample the highest-degree nodes.
        Returns the absolute path of the written file.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "pyvis is required for graph visualization. "
                "Install with: pip install 'trelix[graph-viz]'"
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Sample if too large
        g = cg.nx
        if g.number_of_nodes() > max_nodes:
            # Keep highest-degree nodes
            top_nodes = sorted(g.nodes(), key=lambda n: g.degree(n), reverse=True)[:max_nodes]
            g = cg.subgraph(top_nodes)

        net = Network(
            notebook=False,
            cdn_resources="remote",
            height="900px",
            width="100%",
            select_menu=True,
            filter_menu=False,
            bgcolor="#1a1a2e",
            font_color="#e0e0e0",
        )

        # Add nodes
        for node_id, attrs in g.nodes(data=True):
            community = attrs.get("community") or 0
            color = _PALETTE[int(community) % len(_PALETTE)]
            degree = g.degree(node_id)
            size = max(10, min(50, 10 + degree * 3))
            label = attrs.get("name", str(node_id))
            title = (
                f"<b>{attrs.get('qualified_name', label)}</b><br>"
                f"Kind: {attrs.get('kind', '?')}<br>"
                f"File: {attrs.get('file', '?')}<br>"
                f"Community: {community}"
            )
            net.add_node(
                node_id,
                label=label[:25],
                title=title,
                color=color,
                size=size,
                borderWidth=2,
            )

        # Add edges
        for src, dst, edge_attrs in g.edges(data=True):
            label = edge_attrs.get("label", "")
            color = _EDGE_COLORS.get(label, "#666666")
            net.add_edge(src, dst, title=label, color=color, width=1.5)

        net.force_atlas_2based(central_gravity=0.015, gravity=-31)
        net.save_graph(output_path)
        logger.info("Graph HTML written to %s (%d nodes)", output_path, g.number_of_nodes())
        return str(Path(output_path).resolve())

    def export_community_report(
        self,
        result: GraphBuildResult,
        output_path: str,
    ) -> str:
        """Write a JSON community report. Returns absolute path."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {
            "node_count": result.node_count,
            "edge_count": result.edge_count,
            "community_count": result.community_count,
            "concept_count": result.concept_count,
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "communities": result.community_summary,
        }
        Path(output_path).write_text(json.dumps(report, indent=2))
        logger.info("Community report written to %s", output_path)
        return str(Path(output_path).resolve())
```

- [ ] **Step 5: Install pyvis and run tests**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
.venv/bin/pip install pyvis>=0.3.2 -q
.venv/bin/python -m pytest tests/unit/test_graph_visualizer.py -v --tb=short
```

Expected: all 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trelix/graph/visualizer.py tests/unit/test_graph_visualizer.py pyproject.toml
git commit -m "feat(graph): add GraphVisualizer — Pyvis interactive HTML + community JSON export

- export_html(): community-colored nodes, edge-type-colored arrows, physics layout
- export_community_report(): machine-readable JSON with node/edge/community stats
- max_nodes sampling for large graphs (top by degree)
- pyproject.toml: new [graph-viz] / [graph] optional extras"
```

---

## Phase 4 — Graph-Aware Retrieval (Task 7)

### Task 7: `graph_search` — 4th retrieval leg using community + graph structure

**Files:**
- Create: `src/trelix/graph/search.py`
- Create: `tests/unit/test_graph_search.py`
- Modify: `src/trelix/retrieval/retriever.py` — wire `graph_search` as optional 4th leg

**Interfaces:**
- Consumes: `CodeGraph`, `trelix.store.db.Database`, `trelix.core.models.SearchResult`
- Produces:
  - `graph_search(db: Database, cg: CodeGraph, query_symbol_ids: list[int], depth: int = 2, max_results: int = 15) -> list[SearchResult]`
  - Uses graph traversal (BFS over CodeGraph) to find structurally related symbols and returns hydrated `SearchResult` objects with `source = "graph_search"`
  - `get_community_context(cg: CodeGraph, symbol_id: int) -> list[int]` — returns all symbol IDs in the same community

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_graph_search.py`:

```python
"""Tests for graph-aware search using CodeGraph traversal."""
from __future__ import annotations

from pathlib import Path

from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import assign_communities, detect_communities
from trelix.graph.search import get_community_context, graph_search
from trelix.store.db import Database


def _build_db(tmp_path: Path) -> tuple[Database, list[int]]:
    db = Database(str(tmp_path / "index.db"))
    fid = db.upsert_file(IndexedFile(path="/r/auth.py", rel_path="auth.py",
                                      language=Language.PYTHON, hash="x", size_bytes=100))
    sids = []
    for name in ["login", "logout", "hash_password", "check_token"]:
        s = Symbol(file_id=fid, name=name, qualified_name=name, kind=SymbolKind.FUNCTION,
                   line_start=1, line_end=5, signature=f"def {name}()", body=f"def {name}(): pass")
        sids.append(db.insert_symbol(s))
        db.insert_chunk_for_symbol(sids[-1], f"def {name}(): pass", 5)
    # login → hash_password → check_token
    db.insert_call_edges([
        CallEdge(caller_id=sids[0], callee_name="hash_password", callee_id=sids[2], line=3),
        CallEdge(caller_id=sids[2], callee_name="check_token", callee_id=sids[3], line=2),
    ])
    return db, sids


class TestGraphSearch:
    def test_graph_search_returns_search_results(self, tmp_path: Path) -> None:
        db, sids = _build_db(tmp_path)
        cg = CodeGraph(db)
        results = graph_search(db, cg, query_symbol_ids=[sids[0]], depth=2, max_results=10)
        assert isinstance(results, list)
        # Should find hash_password and check_token as neighbors
        found_ids = {r.symbol.id for r in results}
        assert sids[2] in found_ids  # hash_password

    def test_graph_search_source_label(self, tmp_path: Path) -> None:
        db, sids = _build_db(tmp_path)
        cg = CodeGraph(db)
        results = graph_search(db, cg, query_symbol_ids=[sids[0]], depth=1, max_results=10)
        for r in results:
            assert r.source == "graph_search"

    def test_graph_search_empty_query(self, tmp_path: Path) -> None:
        db, sids = _build_db(tmp_path)
        cg = CodeGraph(db)
        results = graph_search(db, cg, query_symbol_ids=[], depth=1, max_results=10)
        assert results == []

    def test_get_community_context(self, tmp_path: Path) -> None:
        db, sids = _build_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assign_communities(cg, mapping)
        # All 4 symbols in one file with edges — likely same community
        community_members = get_community_context(cg, sids[0])
        assert isinstance(community_members, list)
        assert sids[0] in community_members
```

- [ ] **Step 2: Add `insert_chunk_for_symbol` to `src/trelix/store/db.py`**

This helper is needed by tests to insert chunks directly. Find the existing `insert_chunk` method and add:

```python
def insert_chunk_for_symbol(self, symbol_id: int, chunk_text: str, token_count: int) -> int:
    """Insert a chunk directly for a symbol — test/graph helper."""
    cur = self._conn.execute(
        "INSERT OR IGNORE INTO chunks (symbol_id, chunk_text, token_count) VALUES (?, ?, ?)",
        (symbol_id, chunk_text, token_count),
    )
    self._conn.commit()
    return cur.lastrowid or 0
```

- [ ] **Step 3: Implement `src/trelix/graph/search.py`**

```python
"""Graph-aware search: BFS over CodeGraph to surface structurally related symbols."""
from __future__ import annotations

import logging
from collections import deque

from trelix.core.models import Chunk, SearchResult
from trelix.graph.code_graph import CodeGraph
from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.search")


def graph_search(
    db: Database,
    cg: CodeGraph,
    query_symbol_ids: list[int],
    depth: int = 2,
    max_results: int = 15,
) -> list[SearchResult]:
    """
    BFS over CodeGraph starting from query_symbol_ids.

    Returns hydrated SearchResult objects for all reachable neighbors
    within `depth` hops, scored by hop distance (closer = higher score).
    Source label: "graph_search".
    """
    if not query_symbol_ids:
        return []

    seen: set[int] = set(query_symbol_ids)
    # Queue: (symbol_id, hop_distance)
    queue: deque[tuple[int, int]] = deque()
    for sid in query_symbol_ids:
        for neighbor in cg.neighbors(sid):
            if neighbor not in seen:
                queue.append((neighbor, 1))
                seen.add(neighbor)

    candidates: list[tuple[int, int]] = []  # (symbol_id, hop)

    while queue and len(candidates) < max_results * 3:
        symbol_id, hop = queue.popleft()
        candidates.append((symbol_id, hop))
        if hop < depth:
            for neighbor in cg.neighbors(symbol_id):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, hop + 1))

    # Score: 0.5^hop (closer hops score higher)
    results: list[SearchResult] = []
    for symbol_id, hop in candidates[:max_results]:
        sym_file = db.get_symbol_with_file(symbol_id)
        if sym_file is None:
            continue
        symbol, file = sym_file
        chunk = db.get_first_chunk_for_symbol(symbol_id)
        if chunk is None:
            chunk = Chunk(
                symbol_id=symbol_id,
                chunk_text=symbol.body[:512],
                token_count=len(symbol.body.split()),
                id=None,
            )
        score = 0.5 ** hop
        results.append(
            SearchResult(
                chunk=chunk,
                symbol=symbol,
                file=file,
                score=score,
                rank=len(results) + 1,
                source="graph_search",
            )
        )

    return results


def get_community_context(cg: CodeGraph, symbol_id: int) -> list[int]:
    """Return all symbol IDs in the same community as symbol_id."""
    if symbol_id not in cg.nx:
        return [symbol_id]
    target_community = cg.nx.nodes[symbol_id].get("community")
    if target_community is None:
        return [symbol_id]
    return [
        nid
        for nid, attrs in cg.nx.nodes(data=True)
        if attrs.get("community") == target_community
    ]
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_search.py -v --tb=short
```

Expected: all 4 PASS.

- [ ] **Step 5: Wire graph_search as optional 4th leg in `src/trelix/retrieval/retriever.py`**

Open `src/trelix/retrieval/retriever.py`. Find the `_retrieve_standard` method (or wherever the three legs are assembled). Add graph search as an optional 4th leg, gated by a config flag.

First, add to `RetrievalConfig` in `src/trelix/core/config.py`:

```python
# In RetrievalConfig class, after rerank_provider:
graph_search_enabled: bool = False  # Enable CodeGraph as 4th retrieval leg
graph_search_depth: int = 2         # BFS depth for graph expansion
graph_search_max_results: int = 15  # Max results from graph search leg
```

Then in `retriever.py`, at the point where legs are assembled (after grep results, before RRF fusion), add:

```python
# Graph search leg (optional 4th leg)
if config.retrieval.graph_search_enabled:
    try:
        from trelix.graph.builder import GraphBuilder
        from trelix.graph.search import graph_search
        from trelix.graph.persistence import load_graph_metadata

        cg_builder = GraphBuilder(config)
        # Use cached graph if available, else quick build
        build_result = cg_builder.build(extract_concepts=False)
        cg = build_result.code_graph

        # Seed from top fused results
        seed_ids = [r.chunk.symbol_id for r in fused_results[:10]]
        graph_results = graph_search(
            db=self._db,
            cg=cg,
            query_symbol_ids=seed_ids,
            depth=config.retrieval.graph_search_depth,
            max_results=config.retrieval.graph_search_max_results,
        )
        all_results.extend(graph_results)
    except Exception as exc:
        logger.debug("Graph search leg failed (non-fatal): %s", exc)
```

- [ ] **Step 6: Run full retrieval tests to ensure no regression**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=short -x 2>&1 | tail -10
```

Expected: all existing tests still pass (graph_search_enabled defaults False, so no behavior change).

- [ ] **Step 7: Commit**

```bash
git add src/trelix/graph/search.py tests/unit/test_graph_search.py \
        src/trelix/core/config.py src/trelix/retrieval/retriever.py \
        src/trelix/store/db.py
git commit -m "feat(graph): add graph_search — 4th retrieval leg via CodeGraph BFS

- graph_search(db, cg, query_symbol_ids, depth, max_results) → list[SearchResult]
- BFS over CodeGraph, score = 0.5^hop, source='graph_search'
- get_community_context() returns all co-community symbol IDs
- RetrievalConfig.graph_search_enabled (default False) gates the new leg
- Retriever wires graph search after RRF fusion when enabled
- insert_chunk_for_symbol() helper added to Database"
```

---

## Phase 5 — REST API + MCP + CLI (Tasks 8–9)

### Task 8: REST API graph endpoints

**Files:**
- Modify: `src/trelix/api/app.py`
- Create: `tests/unit/test_api_graph.py`

**Interfaces:**
- Consumes: `GraphBuilder`, `GraphVisualizer`, `CodeGraph`, all of Tasks 1–6
- Produces new endpoints:
  - `GET /graph?repo=` — build/return graph stats JSON
  - `GET /graph/communities?repo=` — community summary JSON
  - `GET /graph/visualize?repo=&output=` — build + export HTML, return `{"path": "..."}`
  - `GET /graph/search?repo=&symbol_id=42&depth=2` — graph BFS from a symbol

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_api_graph.py`:

```python
"""Tests for graph REST API endpoints."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from trelix.api.app import create_app
from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.store.db import Database


def _make_indexed_repo(tmp_path: Path) -> Path:
    db = Database(str(tmp_path / "index.db"))
    fid = db.upsert_file(IndexedFile(path=str(tmp_path / "a.py"), rel_path="a.py",
                                      language=Language.PYTHON, hash="x", size_bytes=10))
    db.insert_symbol(Symbol(file_id=fid, name="fn", qualified_name="fn",
                             kind=SymbolKind.FUNCTION, line_start=1, line_end=5,
                             signature="def fn()", body="def fn(): pass"))
    return tmp_path


class TestGraphApiEndpoints:
    def test_graph_stats(self, tmp_path: Path) -> None:
        repo = _make_indexed_repo(tmp_path)
        app = create_app()
        client = TestClient(app)
        response = client.get(f"/graph?repo={repo}")
        assert response.status_code == 200
        data = response.json()
        assert "node_count" in data
        assert "edge_count" in data
        assert "community_count" in data

    def test_graph_communities(self, tmp_path: Path) -> None:
        repo = _make_indexed_repo(tmp_path)
        app = create_app()
        client = TestClient(app)
        response = client.get(f"/graph/communities?repo={repo}")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_graph_search_endpoint(self, tmp_path: Path) -> None:
        repo = _make_indexed_repo(tmp_path)
        app = create_app()
        client = TestClient(app)
        # symbol_id=1 is the first inserted symbol
        response = client.get(f"/graph/search?repo={repo}&symbol_id=1&depth=1")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
```

- [ ] **Step 2: Add graph endpoints to `src/trelix/api/app.py`**

Open `app.py` and add after the existing `/stats` endpoint:

```python
@app.get("/graph")
def graph_stats(repo: str) -> Any:
    """Build CodeGraph and return stats."""
    from trelix.graph.builder import GraphBuilder
    config = IndexConfig(repo_path=repo)
    result = GraphBuilder(config).build(extract_concepts=False)
    return {
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "community_count": result.community_count,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
    }


@app.get("/graph/communities")
def graph_communities(repo: str) -> Any:
    """Return community summary list."""
    from trelix.graph.builder import GraphBuilder
    config = IndexConfig(repo_path=repo)
    result = GraphBuilder(config).build(extract_concepts=False)
    return result.community_summary


@app.get("/graph/visualize")
def graph_visualize(repo: str, output: str = "") -> Any:
    """Build graph and export Pyvis HTML. Returns path to file."""
    from trelix.graph.builder import GraphBuilder
    from trelix.graph.visualizer import GraphVisualizer
    config = IndexConfig(repo_path=repo)
    result = GraphBuilder(config).build(extract_concepts=False)
    out = output or str(Path(repo) / ".trelix" / "graph.html")
    viz = GraphVisualizer()
    path = viz.export_html(result.code_graph, out)
    return {"path": path, "node_count": result.node_count}


@app.get("/graph/search")
def graph_search_endpoint(repo: str, symbol_id: int, depth: int = 2) -> Any:
    """BFS graph search from a symbol ID."""
    from trelix.graph.builder import GraphBuilder
    from trelix.graph.search import graph_search
    config = IndexConfig(repo_path=repo)
    result = GraphBuilder(config).build(extract_concepts=False)
    from trelix.store.db import Database
    db = Database(config.db_path_absolute)
    results = graph_search(db, result.code_graph, [symbol_id], depth=depth, max_results=20)
    return [
        {
            "symbol": r.symbol.qualified_name,
            "file": r.file.rel_path,
            "kind": r.symbol.kind.value,
            "score": round(r.score, 4),
            "source": r.source,
        }
        for r in results
    ]
```

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_api_graph.py -v --tb=short
```

Expected: all 3 PASS.

- [ ] **Step 4: Commit**

```bash
git add src/trelix/api/app.py tests/unit/test_api_graph.py
git commit -m "feat(api): add /graph REST endpoints

- GET /graph: build CodeGraph stats (node_count, edge_count, community_count)
- GET /graph/communities: community summary JSON
- GET /graph/visualize: export Pyvis HTML, return file path
- GET /graph/search?symbol_id=&depth=: BFS from symbol ID"
```

---

### Task 9: MCP graph tools + CLI `graph` command

**Files:**
- Modify: `packages/trelix-mcp/src/trelix_mcp/server.py`
- Modify: `src/trelix/cli/main.py`
- Modify: `packages/trelix-mcp/tests/test_server.py`

**Interfaces:**
- Consumes: `GraphBuilder`, `GraphVisualizer`, `graph_search` (all Tasks 1–8)
- Produces:
  - MCP tool: `build_knowledge_graph(repo_path, extract_concepts=False) -> dict`
  - MCP tool: `graph_search_mcp(query, repo_path, k=10) -> list[dict]`
  - CLI: `trelix graph ./repo [--visualize] [--output .trelix/graph.html] [--concepts]`

- [ ] **Step 1: Add two MCP tools to `packages/trelix-mcp/src/trelix_mcp/server.py`**

After the existing `blast_radius` tool, add:

```python
@mcp.tool()
def build_knowledge_graph(repo_path: str, extract_concepts: bool = False) -> dict:
    """
    Build a knowledge graph for an indexed codebase.

    ⚠️ IMPORTANT: Run index_codebase first. repo_path must be absolute.

    🎯 What this builds:
    - Unified code property graph (calls + imports + type hierarchy)
    - Community detection: clusters modules into architectural groups
    - Optional LLM concept extraction (set extract_concepts=True, requires LLM config)

    ✨ Returns:
    - node_count: number of symbols in the graph
    - edge_count: number of structural relationships
    - community_count: detected architectural clusters
    - community_summary: top files + symbols per cluster
    """
    from trelix.core.config import IndexConfig
    from trelix.graph.builder import GraphBuilder

    _log.info("build_knowledge_graph repo=%s concepts=%s", repo_path, extract_concepts)
    config = IndexConfig(repo_path=repo_path)
    result = GraphBuilder(config).build(extract_concepts=extract_concepts)
    return {
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "community_count": result.community_count,
        "concept_count": result.concept_count,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "community_summary": result.community_summary,
    }


@mcp.tool()
def graph_search_mcp(query: str, repo_path: str, k: int = 10) -> list[dict]:
    """
    Graph-traversal search: find structurally related symbols by starting
    from semantically similar seeds and following code relationships.

    ⚠️ IMPORTANT: Run index_codebase and optionally build_knowledge_graph first.

    🎯 When to use:
    - "What other code is connected to X?" — follow call/import/type edges
    - "Find the blast radius of a class" — who calls or imports it?
    - "What lives in the same architectural cluster as X?"
    """
    from trelix.core.config import IndexConfig
    from trelix.graph.builder import GraphBuilder
    from trelix.graph.search import graph_search
    from trelix.retrieval.retriever import Retriever
    from trelix.store.db import Database

    _log.info("graph_search_mcp query=%r repo=%s k=%d", query, repo_path, k)
    config = IndexConfig(repo_path=repo_path)

    # First find seed symbols via standard retrieval
    ctx = Retriever(config).retrieve(query)
    seed_ids = [r.chunk.symbol_id for r in ctx.results[:5]]

    if not seed_ids:
        return []

    # Then expand via graph
    build_result = GraphBuilder(config).build(extract_concepts=False)
    db = Database(config.db_path_absolute)
    graph_results = graph_search(db, build_result.code_graph, seed_ids, depth=2, max_results=k)

    return [
        {
            "file": r.file.rel_path,
            "symbol": r.symbol.qualified_name,
            "kind": r.symbol.kind.value,
            "score": round(r.score, 4),
            "source": r.source,
            "body": r.symbol.body[:600],
        }
        for r in graph_results[:k]
    ]
```

- [ ] **Step 2: Update MCP test to include new tool names**

In `packages/trelix-mcp/tests/test_server.py`, find `test_all_four_tools_registered` and update to:

```python
def test_all_four_tools_registered(self) -> None:
    from trelix_mcp.server import mcp
    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert "search_code" in tool_names
    assert "index_codebase" in tool_names
    assert "get_symbol" in tool_names
    assert "blast_radius" in tool_names
    assert "build_knowledge_graph" in tool_names
    assert "graph_search_mcp" in tool_names
```

- [ ] **Step 3: Add `trelix graph` CLI command to `src/trelix/cli/main.py`**

In `main.py`, after the existing `serve` command, add:

```python
@app.command()
def graph(
    repo_path: str = typer.Argument(..., help="Path to indexed repository"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Export Pyvis HTML"),
    output: str = typer.Option("", "--output", "-o", help="Output path for HTML (default: .trelix/graph.html)"),
    concepts: bool = typer.Option(False, "--concepts", "-c", help="Extract LLM semantic concepts"),
    json_output: bool = typer.Option(False, "--json", help="Output stats as JSON"),
) -> None:
    """Build the knowledge graph for an indexed repository."""
    from trelix.core.config import IndexConfig
    from trelix.graph.builder import GraphBuilder

    config = IndexConfig(repo_path=repo_path)
    builder = GraphBuilder(config)

    import rich.console
    console = rich.console.Console()

    with console.status("Building knowledge graph..."):
        result = builder.build(extract_concepts=concepts)

    if json_output:
        import json as _json
        data = {
            "node_count": result.node_count,
            "edge_count": result.edge_count,
            "community_count": result.community_count,
            "concept_count": result.concept_count,
        }
        console.print(_json.dumps(data))
        return

    console.print(f"[green]Knowledge Graph built[/green]")
    console.print(f"  Nodes     : {result.node_count}")
    console.print(f"  Edges     : {result.edge_count}")
    console.print(f"  Communities: {result.community_count}")
    if concepts:
        console.print(f"  Concepts  : {result.concept_count}")
    console.print(f"  Time      : {result.elapsed_seconds:.2f}s")

    if result.community_summary:
        console.print("\n[bold]Top Communities:[/bold]")
        for c in result.community_summary[:5]:
            files = ", ".join(c["top_files"][:3])
            console.print(f"  [{c['community_id']}] {c['size']} nodes — {files}")

    if visualize:
        from trelix.graph.visualizer import GraphVisualizer
        from pathlib import Path
        out = output or str(Path(repo_path) / ".trelix" / "graph.html")
        viz = GraphVisualizer()
        path = viz.export_html(result.code_graph, out)
        console.print(f"\n[blue]Graph visualization:[/blue] {path}")
```

- [ ] **Step 4: Run full test suite to verify no regressions**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -15
```

Expected: all pass including all new graph tests.

- [ ] **Step 5: Smoke test the CLI**

```bash
# Index trelix itself first (for smoke testing)
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
.venv/bin/trelix index . 2>/dev/null || echo "index done"
.venv/bin/python -m trelix.cli.main graph . --json
```

Expected: JSON with node_count, edge_count, community_count values.

- [ ] **Step 6: Commit**

```bash
git add packages/trelix-mcp/src/trelix_mcp/server.py \
        packages/trelix-mcp/tests/test_server.py \
        src/trelix/cli/main.py
git commit -m "feat(mcp+cli): add graph tools and CLI command

MCP:
- build_knowledge_graph(repo_path, extract_concepts) → graph stats + community summary
- graph_search_mcp(query, repo_path, k) → vector seed + graph BFS expansion

CLI:
- trelix graph ./repo [--visualize] [--output path] [--concepts] [--json]
  Shows node/edge/community counts, top communities, optional Pyvis HTML export"
```

---

## Phase 6 — Documentation + pyproject.toml (Task 10)

### Task 10: pyproject.toml extras, CHANGELOG, and docs update

**Files:**
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`
- Modify: `docs/architecture.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: all Tasks 1–9 (must be complete)
- Produces: `[knowledge-graph]` extra in pyproject; CHANGELOG entry; architecture diagram update

- [ ] **Step 1: Update `pyproject.toml` with `[knowledge-graph]` extra**

Add to `[project.optional-dependencies]`:

```toml
knowledge-graph = ["networkx>=3.3.0", "pyvis>=0.3.2", "seaborn>=0.13.0"]
```

Note: `networkx` is already in core deps — the extra serves as a convenience install hint. Also ensure `pyvis` and `seaborn` are listed in `graph-viz` and `knowledge-graph` extras.

- [ ] **Step 2: Add CHANGELOG entry**

In `CHANGELOG.md`, add a new `[Unreleased]` section (or under an existing one):

```markdown
## [Unreleased]

### Added
- **Knowledge Graph**: new `trelix/graph/` module unifying call/import/type edges into a traversable `CodeGraph` (NetworkX MultiDiGraph)
- **Community Detection**: Louvain algorithm clusters codebase into architectural modules; `trelix graph ./repo` CLI command shows top communities
- **Semantic Concepts**: `ConceptExtractor` — LLM-powered extraction of architectural concepts from symbol batches (crash-safe, returns `[]` on any failure)
- **Graph Visualization**: `GraphVisualizer.export_html()` — Pyvis interactive HTML with community coloring and edge-type coloring; `pip install trelix[knowledge-graph]`
- **4th Retrieval Leg**: `graph_search_enabled=True` in `RetrievalConfig` enables CodeGraph BFS as a 4th search leg after RRF fusion
- **REST API**: `GET /graph`, `GET /graph/communities`, `GET /graph/visualize`, `GET /graph/search` endpoints
- **MCP Tools**: `build_knowledge_graph` and `graph_search_mcp` tools in `trelix-mcp`
- **Graph Persistence**: `graph_metadata` SQLite table stores community and degree centrality per symbol
```

- [ ] **Step 3: Update `docs/architecture.md`**

Add a new section after the existing retrieval pipeline description:

```markdown
## Knowledge Graph Layer

trelix v3.0 adds a Knowledge Graph layer on top of the existing call/import/type edge tables.

### CodeGraph (trelix/graph/code_graph.py)
Wraps three SQLite tables into a unified NetworkX MultiDiGraph:
- `calls` → CALLS edges (caller_id → callee_id)
- `imports` → IMPORTS edges (file_id → imported_file_id)
- `type_edges` → EXTENDS / IMPLEMENTS / TRAIT_IMPL / EMBEDDED edges

### Community Detection (trelix/graph/community.py)
Louvain algorithm (fast, O(n log n)) or Girvan-Newman (quality) clusters the graph into
architectural communities. Communities represent logical module groupings — auth layer,
data layer, API layer — without any human labeling.

### Semantic Concepts (trelix/graph/concepts.py)
LLM extracts high-level concepts (JWT authentication, event sourcing, CQRS pattern) from
symbol batches. Stored in `graph_concepts` SQLite table. Crash-safe: returns `[]` if LLM
is unavailable.

### Graph-Aware Search (trelix/graph/search.py)
4th retrieval leg: BFS over CodeGraph starting from top RRF results.
Score = 0.5^hop (closer = higher). Enabled via `graph_search_enabled=True`.

### Visualization (trelix/graph/visualizer.py)
Pyvis interactive HTML with:
- Community-colored nodes (pastel palette)
- Edge-type-colored arrows (CALLS=blue, IMPORTS=purple, EXTENDS=green)
- Physics simulation (Force Atlas 2)
- Node sizing by degree

Install: `pip install 'trelix[knowledge-graph]'`
CLI: `trelix graph ./repo --visualize`
REST: `GET /graph/visualize?repo=...`
```

- [ ] **Step 4: Commit**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
git add pyproject.toml CHANGELOG.md docs/architecture.md README.md
git commit -m "docs: update for knowledge graph feature

- pyproject.toml: [knowledge-graph] extra (networkx + pyvis + seaborn)
- CHANGELOG: unreleased section with all new graph capabilities
- architecture.md: Knowledge Graph Layer section
- README.md: knowledge graph quick start and feature table entry"
```

---

## Verification Checklist

After all 10 tasks, run this full validation:

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix

# 1. All new graph tests pass
.venv/bin/python -m pytest tests/unit/test_graph_*.py -v --tb=short
# Expected: 23+ tests, all PASS

# 2. Full unit test suite — no regressions
.venv/bin/python -m pytest tests/unit/ -q --tb=short
# Expected: 900+ pass, 0 fail

# 3. Lint and format
.venv/bin/ruff check src/trelix/graph/ tests/unit/test_graph_*.py
.venv/bin/ruff format src/trelix/graph/ tests/unit/test_graph_*.py && git diff --exit-code

# 4. Type check
.venv/bin/python -m mypy src/trelix/graph/ --strict --ignore-missing-imports

# 5. CLI smoke test
.venv/bin/python -m trelix.cli.main graph . --json
# Expected: {"node_count": N, "edge_count": M, "community_count": K, "concept_count": 0}

# 6. MCP tools test
cd packages/trelix-mcp
pip install -e "." -q
python -m pytest tests/ -v --tb=short
# Expected: all 6+ tools registered including build_knowledge_graph, graph_search_mcp

# 7. REST API graph endpoints
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
.venv/bin/python -c "
from fastapi.testclient import TestClient
from trelix.api.app import create_app
client = TestClient(create_app())
# Stats
r = client.get('/graph?repo=.')
print('/graph:', r.status_code, list(r.json().keys()))
# Communities
r = client.get('/graph/communities?repo=.')
print('/graph/communities:', r.status_code, type(r.json()))
"
```

---

## Self-Review: Spec Coverage

| Feature | Task |
|---------|------|
| Unified MultiDiGraph (call+import+type) | Task 1 |
| CALLS / IMPORTS / EXTENDS / IMPLEMENTS edges | Task 1 |
| Node attributes (name, kind, file, language, community) | Task 1 |
| Shortest path, subgraph, neighbors API | Task 1 |
| Persist community + centrality to SQLite | Task 2 |
| Louvain / Girvan-Newman community detection | Task 3 |
| Community summary (top_files, top_symbols) | Task 3 |
| LLM semantic concept extraction | Task 4 |
| Crash-safe concept extraction (returns [] on failure) | Task 4 |
| Graph concepts SQLite persistence | Task 4 |
| GraphBuilder orchestration pipeline | Task 5 |
| GraphBuildResult dataclass | Task 5 |
| Pyvis interactive HTML with community coloring | Task 6 |
| Edge-type colored arrows | Task 6 |
| JSON community report export | Task 6 |
| `[knowledge-graph]` / `[graph-viz]` extras | Task 6 |
| BFS graph search as 4th retrieval leg | Task 7 |
| `graph_search_enabled` config flag | Task 7 |
| REST `/graph`, `/graph/communities`, `/graph/visualize`, `/graph/search` | Task 8 |
| MCP `build_knowledge_graph` tool | Task 9 |
| MCP `graph_search_mcp` tool | Task 9 |
| `trelix graph` CLI command with --visualize | Task 9 |
| pyproject.toml extras update | Task 10 |
| CHANGELOG entry | Task 10 |
| docs/architecture.md update | Task 10 |

All spec requirements are covered across the 10 tasks. No gaps found.
