# Phase 3: Graph API — Design Spec

**Date:** 2026-06-28
**Status:** Approved
**Scope:** Expose call_edges and import_edges as a queryable public Python API and CLI subcommand.

---

## Problem

The indexer resolves and stores call edges (233 in trelix self-index) and import paths (416), but they are **not exposed as user-queryable API**. The data lives inside `Database` in `store/db.py` behind integer `symbol_id` keys.

A user who wants to answer "what calls this function?" or "what imports this module?" today must:

1. Know that `Database` exists and how to instantiate it.
2. Know `get_symbol_by_name()` to resolve a name to an integer `symbol_id`.
3. Call `db.get_callers(symbol_id)` or `db.get_callees(symbol_id)` — returning raw `list[int]`.
4. Hydrate each id themselves via `db.get_symbol_with_file(symbol_id)`.

None of these are in `__all__`, documented at the `Retriever` level, or accessible from the CLI.

The graph IS built and works — it is just not surfaced.

---

## Goals

1. Add three public methods to `Retriever`: `get_callers`, `get_callees`, `get_importers`.
2. Return `list[SearchResult]` — the same type `retrieve()` returns — so callers can reuse all downstream tooling (rendering, token budgeting, synthesis).
3. Add a `trelix graph <repo> <symbol>` CLI subcommand backed by Rich tables.
4. Write unit tests (mock db) and one integration test (index trelix itself).

### Non-goals

- Multi-hop traversal (depth > 1). Phase 3 is 1-hop only. Depth parameter can be added later.
- Type edge queries (`get_subclasses`, `get_supertypes`). These live in `type_edges` and belong in a separate phase.
- REST API or MCP tool wrapping. Out of scope.

---

## Current data model (reference)

### `calls` table

```sql
CREATE TABLE calls (
    id          INTEGER PRIMARY KEY,
    caller_id   INTEGER NOT NULL,   -- symbol.id of the calling symbol
    callee_name TEXT    NOT NULL,   -- raw call target name (always populated)
    callee_id   INTEGER,            -- resolved symbol.id (NULL if external/stdlib)
    line        INTEGER,
    callee_type_hint TEXT
);
```

`callee_id` is `NULL` for calls that resolve to external libraries or unresolved names. Only resolved (non-NULL) edges can be hydrated to `SearchResult`.

### `imports` table

```sql
CREATE TABLE imports (
    id               INTEGER PRIMARY KEY,
    file_id          INTEGER NOT NULL,   -- importing file
    imported_from    TEXT    NOT NULL,   -- module path string
    imported_names   TEXT    NOT NULL,   -- JSON list of names
    imported_file_id INTEGER             -- resolved file.id (NULL if external)
);
```

`imported_file_id` is `NULL` for third-party packages. Only resolved imports can be surfaced.

### Existing `db.py` methods being wrapped

| `Database` method | Signature | Notes |
|---|---|---|
| `get_callers` | `(symbol_id: int) -> list[int]` | Returns caller symbol ids |
| `get_callees` | `(symbol_id: int) -> list[int]` | Returns callee symbol ids (resolved only) |
| `get_symbol_by_name` | `(name: str) -> list[Symbol]` | Name match, returns all overloads |
| `get_files_importing` | `(file_id: int) -> list[int]` | Reverse import: who imports this file |
| `get_symbols_for_file` | `(file_id: int) -> list[Symbol]` | All symbols in a file |
| `get_symbol_with_file` | `(symbol_id: int) -> tuple[Symbol, IndexedFile] \| None` | Hydrates symbol+file together |
| `get_first_chunk_for_symbol` | `(symbol_id: int) -> Chunk \| None` | First text chunk for embedding |

---

## Design

### 1. Three new methods on `Retriever`

File: `src/trelix/retrieval/retriever.py`

#### 1.1 `get_callers`

```python
def get_callers(self, symbol_name: str) -> list[SearchResult]:
    """
    Return the symbols that call ``symbol_name`` (1-hop incoming call edges).

    ``symbol_name`` may be a bare name (``"retrieve"``) or a qualified name
    (``"Retriever.retrieve"``).  All matching symbols are tried; results are
    deduplicated by symbol id and sorted by file path + line for determinism.

    Returns an empty list when the symbol is not found or has no callers.
    """
```

**Algorithm:**

1. Resolve `symbol_name` to `list[Symbol]` via `self.db.get_symbol_by_name(symbol_name)`.
   - If empty, return `[]`.
2. For each resolved symbol (there may be multiple overloads), collect `caller_ids = self.db.get_callers(sym.id)`.
3. Deduplicate caller ids across all resolved symbols.
4. Hydrate each caller id to `SearchResult` via `_hydrate_symbol_id(symbol_id, source="graph_callers")`.
5. Sort by `result.file.rel_path, result.symbol.line_start` for stable output.
6. Return the list.

#### 1.2 `get_callees`

```python
def get_callees(self, symbol_name: str) -> list[SearchResult]:
    """
    Return the symbols that ``symbol_name`` calls (1-hop outgoing call edges,
    resolved internal calls only — external/stdlib calls are excluded).

    Same name resolution and deduplication rules as ``get_callers``.
    """
```

**Algorithm:** same as `get_callers` but call `self.db.get_callees(sym.id)` instead of `get_callers`.

#### 1.3 `get_importers`

```python
def get_importers(self, module_path: str) -> list[SearchResult]:
    """
    Return the top symbol from each file that imports ``module_path``.

    ``module_path`` is matched against ``files.rel_path`` (e.g.
    ``"src/trelix/retrieval/retriever.py"`` or just ``"retriever"``).
    Matching is by suffix: a partial path matches any file whose rel_path ends
    with the given value (after stripping a leading ``/`` if present).

    For each importing file, only the first symbol (lowest line_start) is
    returned, to keep result count bounded.  Callers that need all symbols
    from an importing file should use the ``Retriever.retrieve()`` pipeline.

    Returns an empty list when the module is not indexed or has no importers.
    """
```

**Algorithm:**

1. Find `file_id` for `module_path` via `self.db.get_file_by_rel_path_suffix(module_path)`.
   - If no match, return `[]`.
2. Collect `importer_file_ids = self.db.get_files_importing(file_id)`.
3. For each `importer_file_id`, get symbols via `self.db.get_symbols_for_file(fid)`, sort by `line_start`, take the first one.
4. Hydrate that symbol id to `SearchResult` via `_hydrate_symbol_id(symbol_id, source="graph_importers")`.
5. Sort by `result.file.rel_path` for determinism.
6. Return the list.

#### 1.4 Shared private helper: `_hydrate_symbol_id`

This helper already exists implicitly in `expand_with_call_graph` (graph.py lines 72-95). Pull it into `Retriever` as a private method so the three new public methods share the same hydration path.

```python
def _hydrate_symbol_id(self, symbol_id: int, source: str) -> SearchResult | None:
    """
    Hydrate a raw symbol_id into a SearchResult.
    Returns None when the symbol is no longer in the db (stale index).
    Score is fixed at 1.0 — graph queries are exact, not ranked.
    """
    sym_file = self.db.get_symbol_with_file(symbol_id)
    if sym_file is None:
        return None
    symbol, file = sym_file
    chunk = self.db.get_first_chunk_for_symbol(symbol_id)
    if chunk is None:
        chunk = Chunk(
            symbol_id=symbol_id,
            chunk_text=symbol.body[:2000],
            token_count=0,
        )
    return SearchResult(
        chunk=chunk,
        symbol=symbol,
        file=file,
        score=1.0,
        rank=0,       # set by caller after sorting
        source=source,
    )
```

After collecting the list, callers assign monotonically increasing `rank` values (1-indexed).

### 2. New `db.py` helper: `get_file_by_rel_path_suffix`

Required by `get_importers`. Does not exist yet.

```python
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
        return rows[0][0]
    return None
```

This is a new 8-line method — minimal surface area, no schema change.

### 3. `__init__.py` — no changes to `__all__`

`Retriever` is already exported. The three new methods are discoverable via normal `help(Retriever)` introspection. No import-level change is needed.

Document them in the module docstring so they appear in `trelix --help` equivalents and IDE hover text.

### 4. CLI: `trelix graph`

File: `src/trelix/cli/main.py`

New subcommand appended after the existing `query` command.

```python
@app.command()
def graph(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    symbol: str = typer.Argument(..., help="Symbol name or module path to inspect"),
    provider: str = typer.Option("local", help=_PROVIDER_HELP),
    direction: str = typer.Option(
        "all",
        "--direction", "-d",
        help="callers | callees | importers | all",
    ),
) -> None:
    """Show call graph and import edges for a symbol or module."""
```

**Output contract:**

When `--direction all` (default), print three Rich tables in order:

1. **Callers** — symbols that call `<symbol>`
2. **Callees** — symbols that `<symbol>` calls
3. **Importers** — files that import `<symbol>` (module suffix match)

Each table has columns: `File` (rel_path, dim), `Symbol` (qualified_name, bold), `Lines` (start-end, right-align), `Kind`.

When a table is empty, print a dim `"(none)"` row rather than hiding the table — this signals that the graph edge type was queried and found no results, rather than being silently skipped.

Exit code 0 always (empty results are not errors). Exit code 1 only on config or db errors.

**Example output:**

```
 Graph: Retriever.retrieve

 Callers (2)
┌─────────────────────────────────────┬──────────────────────────┬───────┬──────────┐
│ File                                │ Symbol                   │ Lines │ Kind     │
│ tests/integration/test_retriever.py │ test_retrieve_basic      │ 14-28 │ function │
│ src/trelix/cli/main.py              │ query                    │ 167-… │ function │
└─────────────────────────────────────┴──────────────────────────┴───────┴──────────┘

 Callees (5)
┌─────────────────────────────────────┬────────────────────────┬────────┬──────────┐
│ File                                │ Symbol                 │ Lines  │ Kind     │
│ src/trelix/retrieval/retriever.py   │ Retriever._run_vector  │ 200-… │ method   │
│ …                                   │ …                      │ …      │ …        │
└─────────────────────────────────────┴────────────────────────┴────────┴──────────┘

 Importers of "retriever" (3)
┌────────────────────────────────┬──────────────┬───────┬──────────┐
│ File                           │ Symbol       │ Lines │ Kind     │
│ src/trelix/cli/main.py         │ query        │ 167-… │ function │
│ …                              │ …            │ …     │ …        │
└────────────────────────────────┴──────────────┴───────┴──────────┘
```

---

## File change map

| File | Change |
|---|---|
| `src/trelix/retrieval/retriever.py` | Add `get_callers`, `get_callees`, `get_importers`, `_hydrate_symbol_id` |
| `src/trelix/store/db.py` | Add `get_file_by_rel_path_suffix` |
| `src/trelix/cli/main.py` | Add `graph` subcommand |
| `tests/unit/test_graph_api.py` | New: unit tests with mocked db |
| `tests/integration/test_graph_api_integration.py` | New: index trelix itself, assert non-empty callers |

No schema migration. No new dependencies. No changes to `__init__.py` or `__all__`.

---

## Testing

### Unit tests (`tests/unit/test_graph_api.py`)

Use `unittest.mock.MagicMock` to replace `self.db` on a `Retriever` instance constructed with a minimal `IndexConfig`.

#### Test cases

**`test_get_callers_returns_search_results`**
- Mock `db.get_symbol_by_name("foo")` → `[Symbol(id=10, ...)]`
- Mock `db.get_callers(10)` → `[20, 30]`
- Mock `db.get_symbol_with_file(20)` and `(30)` → valid `(Symbol, IndexedFile)` pairs
- Mock `db.get_first_chunk_for_symbol` → valid `Chunk` objects
- Assert: `retriever.get_callers("foo")` returns a list of two `SearchResult` objects.
- Assert: `source == "graph_callers"` on every result.
- Assert: `rank` values are 1 and 2.

**`test_get_callers_unknown_symbol_returns_empty`**
- Mock `db.get_symbol_by_name("nonexistent")` → `[]`
- Assert: result is `[]`.

**`test_get_callees_returns_search_results`**
- Same structure as callers test but calling `get_callees` and checking `source == "graph_callees"`.

**`test_get_callees_no_resolved_edges_returns_empty`**
- Mock `db.get_callees(10)` → `[]` (all calls were to external libs, no resolved ids)
- Assert: result is `[]`.

**`test_get_importers_returns_search_results`**
- Mock `db.get_file_by_rel_path_suffix("retriever")` → `5`
- Mock `db.get_files_importing(5)` → `[7, 8]`
- Mock `db.get_symbols_for_file(7)` → `[Symbol(id=40, line_start=10, ...)]`
- Mock `db.get_symbols_for_file(8)` → `[Symbol(id=50, line_start=5, ...)]`
- Assert: two results, `source == "graph_importers"`, sorted by `file.rel_path`.

**`test_get_importers_unknown_module_returns_empty`**
- Mock `db.get_file_by_rel_path_suffix("nomodule")` → `None`
- Assert: result is `[]`.

**`test_hydrate_symbol_id_stale_id_returns_none`**
- Mock `db.get_symbol_with_file(999)` → `None`
- Assert: `retriever._hydrate_symbol_id(999, "test")` returns `None`.

### Integration test (`tests/integration/test_graph_api_integration.py`)

Uses the real trelix repo as the indexed target (same pattern as existing integration tests).

```python
@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    """Index the trelix repo itself and return a Retriever."""
    repo = Path(__file__).parent.parent.parent  # trelix repo root
    db_dir = tmp_path_factory.mktemp("trelix_self_index")
    config = IndexConfig(
        repo_path=str(repo),
        db_path=str(db_dir / "trelix.db"),
        embedder=EmbedderConfig(provider="local"),
    )
    Indexer(config).index()
    return Retriever(config)
```

**`test_get_callers_retriever_retrieve_non_empty`**
- Call `retriever.get_callers("Retriever.retrieve")`.
- Assert: result is non-empty (the CLI `query` command and integration tests call it).
- Assert: all items have `source == "graph_callers"`.
- Assert: all items have `score == 1.0`.

**`test_get_callees_retriever_retrieve_non_empty`**
- Call `retriever.get_callees("Retriever.retrieve")`.
- Assert: result is non-empty (retrieve calls multiple internal methods).

**`test_get_importers_retriever_module_non_empty`**
- Call `retriever.get_importers("retrieval/retriever")`.
- Assert: result is non-empty (cli/main.py and tests import the module).

**`test_get_callers_nonexistent_symbol_returns_empty`**
- Call `retriever.get_callers("__this_symbol_does_not_exist__")`.
- Assert: result is `[]`, no exception raised.

---

## Invariants

- `get_callers`, `get_callees`, `get_importers` never raise. They return `[]` on any not-found condition.
- `source` field on returned `SearchResult` is exactly `"graph_callers"`, `"graph_callees"`, or `"graph_importers"` — distinct from `"graph_expansion"` used internally by `expand_with_call_graph`.
- `score` is `1.0` for all graph API results — graph queries are exact, not ranked.
- `rank` is 1-indexed, contiguous, assigned after sorting.
- Results from `get_callers("X")` are a strict subset of what `expand_with_call_graph` would return given a seed result for X — same db calls, same hydration path, just exposed directly.

---

## Implementation order

1. Add `get_file_by_rel_path_suffix` to `db.py` (8 lines, no schema change).
2. Add `_hydrate_symbol_id` to `Retriever` (20 lines, no new imports).
3. Add `get_callers`, `get_callees`, `get_importers` to `Retriever` (each ~20 lines).
4. Write unit tests. Run: `python -m pytest tests/unit/test_graph_api.py -v`.
5. Write integration test. Run: `python -m pytest tests/integration/test_graph_api_integration.py -v`.
6. Add `graph` CLI subcommand (approx 70 lines).
7. Update module docstring in `__init__.py` to mention the three new methods.

Total estimated diff: ~200 lines of production code, ~150 lines of tests.
