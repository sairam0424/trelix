# trelix Phase 1 — Watch Bridge, DB Index, Router Config Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire MCP file-change notifications into the watch pipeline, add a missing `files.rel_path` DB index, and fix `AdaptiveRouter` building its own isolated `RetrievalConfig` instead of sharing the Retriever's config.

**Architecture:** Three independent fixes in one phase. Task 1 connects `_subscription_registry.notify_file_changed()` into `FileWatcher._do_reindex()` — the infrastructure exists but nothing calls it. Task 2 adds a single `CREATE INDEX IF NOT EXISTS` on `files.rel_path` that eliminates a full table scan on every watch event. Task 3 changes `AdaptiveRouter.__init__` to accept an optional `RetrievalConfig` parameter instead of always constructing its own, and updates `Retriever.__init__` to pass its config through.

**Tech Stack:** Python 3.11+, SQLite, pytest + unittest.mock. No new dependencies.

## Global Constraints

- Python ≥ 3.11
- **DO NOT bump any version numbers** — no changes to `pyproject.toml`, `__init__.py`, or `CHANGELOG.md` version fields
- Tests run from trelix root: `python -m pytest tests/unit/<file>.py -v`
- Conventional commits: `feat(watcher):`, `perf(store):`, `fix(retrieval):`
- `FileWatcher._do_reindex` must remain non-fatal on all errors — any new code goes inside a try/except
- The `notify_file_changed` call must not block the reindex path — import lazily
- The DB index migration is additive only — `CREATE INDEX IF NOT EXISTS` (no data loss)
- `AdaptiveRouter.__init__` signature change must be backward-compatible — new param must have default `None`

---

## Task 1: Wire `notify_file_changed` into `FileWatcher._do_reindex`

**Files:**
- Modify: `src/trelix/indexing/watcher.py` — `_do_reindex()` method (~line 193, after `GraphUpdater` call)
- Test: `tests/unit/test_watcher.py` — add `TestWatchBridgeMCPNotification` class

**Interfaces:**
- Consumes: `notify_file_changed(registry, repo_path, changed_file)` from `trelix_mcp.subscriptions` (optional import — trelix-mcp may not be installed)
- Consumes: `_subscription_registry` from `trelix_mcp.server` (same optional import)
- Produces: After a successful re-index, all MCP subscribers watching `trelix://repo/{repo_path}/manifest` receive `notifications/resources/updated`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_watcher.py`:

```python
class TestWatchBridgeMCPNotification:
    """notify_file_changed must be called after a successful re-index."""

    def _make_watcher(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from trelix.indexing.watcher import FileWatcher

        indexer = MagicMock()
        indexer.config.repo_path = str(tmp_path)
        indexer.db = MagicMock()
        indexer.embedder = MagicMock()
        indexer.embedder.dimension = 384
        walker = MagicMock()

        with patch("trelix.indexing.watcher.DimensionGuard.check"):
            return FileWatcher(indexer, walker)

    def test_notify_called_after_successful_reindex(self, tmp_path):
        from unittest.mock import MagicMock, patch

        watcher = self._make_watcher(tmp_path)
        watcher._indexer.index_file.return_value = {
            "status": "ok", "symbols_updated": 2, "chunks_updated": 5, "skipped": False
        }

        mock_notify = MagicMock()
        mock_registry = MagicMock()

        with patch.dict("sys.modules", {
            "trelix_mcp": MagicMock(),
            "trelix_mcp.server": MagicMock(_subscription_registry=mock_registry),
            "trelix_mcp.subscriptions": MagicMock(notify_file_changed=mock_notify),
        }), patch("trelix.graph.updater.GraphUpdater"):
            watcher._do_reindex(str(tmp_path / "src" / "auth.py"))

        mock_notify.assert_called_once()
        call_kwargs = mock_notify.call_args
        assert call_kwargs is not None

    def test_notify_not_called_when_file_skipped(self, tmp_path):
        from unittest.mock import MagicMock, patch

        watcher = self._make_watcher(tmp_path)
        watcher._indexer.index_file.return_value = {
            "status": "ok", "symbols_updated": 0, "chunks_updated": 0, "skipped": True
        }

        mock_notify = MagicMock()

        with patch.dict("sys.modules", {
            "trelix_mcp": MagicMock(),
            "trelix_mcp.server": MagicMock(_subscription_registry=MagicMock()),
            "trelix_mcp.subscriptions": MagicMock(notify_file_changed=mock_notify),
        }), patch("trelix.graph.updater.GraphUpdater"):
            watcher._do_reindex(str(tmp_path / "src" / "auth.py"))

        mock_notify.assert_not_called()

    def test_notify_non_fatal_when_trelix_mcp_not_installed(self, tmp_path):
        """trelix watch must work even without trelix-mcp installed."""
        from unittest.mock import MagicMock, patch
        import sys

        watcher = self._make_watcher(tmp_path)
        watcher._indexer.index_file.return_value = {
            "status": "ok", "symbols_updated": 1, "chunks_updated": 2, "skipped": False
        }

        # Simulate trelix_mcp not installed
        with patch.dict("sys.modules", {"trelix_mcp": None}), \
             patch("trelix.graph.updater.GraphUpdater"):
            # Must not raise — trelix watch works without MCP
            watcher._do_reindex(str(tmp_path / "src" / "auth.py"))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/unit/test_watcher.py::TestWatchBridgeMCPNotification -v
```

Expected: `FAILED` — `mock_notify.assert_called_once()` fails because `notify_file_changed` is never called.

- [ ] **Step 3: Add the notification hook to `_do_reindex` in `watcher.py`**

In `src/trelix/indexing/watcher.py`, find `_do_reindex`. After the `GraphUpdater` block (~line 193), add:

```python
            # Fire MCP notifications/resources/updated for any subscribed clients.
            # Non-fatal: trelix watch works without trelix-mcp installed.
            if not skipped:
                try:
                    from trelix_mcp.server import _subscription_registry  # type: ignore[import]
                    from trelix_mcp.subscriptions import notify_file_changed  # type: ignore[import]

                    notify_file_changed(
                        registry=_subscription_registry,
                        repo_path=str(Path(self._indexer.config.repo_path).resolve()),
                        changed_file=rel,
                    )
                except Exception:
                    pass  # trelix-mcp not installed or notification failed — non-fatal
```

The `if not skipped:` guard ensures notifications only fire when the file actually changed (not on hash-identical no-ops).

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/unit/test_watcher.py::TestWatchBridgeMCPNotification -v
```

Expected: `3 passed`

- [ ] **Step 5: Run full watcher test suite to confirm no regression**

```bash
python -m pytest tests/unit/test_watcher.py -v -q 2>&1 | tail -5
```

Expected: all existing + 3 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/trelix/indexing/watcher.py tests/unit/test_watcher.py
git commit -m "feat(watcher): fire MCP notify_file_changed after successful re-index

Wires the MCP subscription notification bridge into FileWatcher._do_reindex.
After a file is re-indexed (not skipped), notify_file_changed() fires
notifications/resources/updated for all subscribed MCP clients.
Non-fatal when trelix-mcp is not installed — trelix watch still works."
```

---

## Task 2: Add `idx_files_rel_path` index to `files` table

**Files:**
- Modify: `src/trelix/store/db.py` — schema string and `init_schema()` method
- Test: `tests/unit/test_store.py` — add `TestFilesRelPathIndex` class

**Interfaces:**
- Produces: `CREATE INDEX IF NOT EXISTS idx_files_rel_path ON files(rel_path)` present after `init_schema()`
- Existing queries like `WHERE rel_path = ?` and `WHERE rel_path LIKE ?` benefit automatically

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_store.py`:

```python
class TestFilesRelPathIndex:
    def test_files_rel_path_index_exists(self, tmp_path):
        from trelix.store.db import Database

        db = Database(str(tmp_path / "test.db"))
        db.init_schema()

        # Query sqlite_master to confirm the index exists
        row = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_files_rel_path'"
        ).fetchone()
        assert row is not None, (
            "idx_files_rel_path index not found — "
            "add 'CREATE INDEX IF NOT EXISTS idx_files_rel_path ON files(rel_path)' to init_schema()"
        )

    def test_files_rel_path_index_covers_rel_path_column(self, tmp_path):
        from trelix.store.db import Database

        db = Database(str(tmp_path / "test.db"))
        db.init_schema()

        # EXPLAIN QUERY PLAN shows index usage for WHERE rel_path = ?
        plan = db._conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM files WHERE rel_path = ?",
            ("src/auth.py",)
        ).fetchall()
        plan_text = " ".join(str(row) for row in plan)
        assert "idx_files_rel_path" in plan_text, (
            f"Query plan does not use idx_files_rel_path: {plan_text}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/unit/test_store.py::TestFilesRelPathIndex -v
```

Expected: `FAILED` — `idx_files_rel_path` not found in `sqlite_master`.

- [ ] **Step 3: Add the index to `db.py`**

In `src/trelix/store/db.py`, find the schema string (around line 84–130 where other `CREATE INDEX IF NOT EXISTS` statements live). Add after the existing file-related indices:

```python
CREATE INDEX IF NOT EXISTS idx_files_rel_path   ON files(rel_path);
```

Also add it to any `init_schema()` migration path that handles existing databases (look for `ALTER TABLE` or `CREATE INDEX IF NOT EXISTS` in the method body — add the same line there).

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/unit/test_store.py::TestFilesRelPathIndex -v
```

Expected: `2 passed`

- [ ] **Step 5: Run full store tests**

```bash
python -m pytest tests/unit/test_store.py -v -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/trelix/store/db.py tests/unit/test_store.py
git commit -m "perf(store): add idx_files_rel_path index on files.rel_path

Every GraphUpdater.update_file() call queries files WHERE rel_path = ?
without an index, causing a full table scan at 10k+ file repos.
CREATE INDEX IF NOT EXISTS is safe on existing databases."
```

---

## Task 3: Fix `AdaptiveRouter` isolated `RetrievalConfig` construction

**Files:**
- Modify: `src/trelix/retrieval/planner/agent.py` — `AdaptiveRouter.__init__` (~line 90), accept optional `retrieval_config`
- Modify: `src/trelix/retrieval/retriever.py` — `Retriever.__init__` (~line 97), pass `config.retrieval` to planner
- Test: `tests/unit/test_planner_adaptive.py` — add `TestAdaptiveRouterConfigPassthrough`

**Interfaces:**
- Produces: `AdaptiveRouter.__init__(self, config: EmbedderConfig, retrieval_config: RetrievalConfig | None = None)` — new optional param, backward-compatible
- Produces: `Retriever.__init__` passes `config.retrieval` to `QueryPlanner(config.embedder, retrieval_config=config.retrieval)`
- When `retrieval_config` is provided, it is used directly; when `None`, falls back to constructing from env (existing behavior)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_planner_adaptive.py`:

```python
class TestAdaptiveRouterConfigPassthrough:
    def test_router_uses_provided_retrieval_config(self):
        """When retrieval_config is passed, router uses it instead of building from env."""
        from trelix.core.config import EmbedderConfig, RetrievalConfig
        from trelix.retrieval.planner.agent import AdaptiveRouter

        cfg = RetrievalConfig()
        cfg.short_query_lexical_enabled = True
        cfg.short_query_token_threshold = 2  # very short threshold

        router = AdaptiveRouter(EmbedderConfig(), retrieval_config=cfg)

        # The router should use our cfg, not build a new one from env
        assert router._retrieval_config is cfg

    def test_router_without_retrieval_config_falls_back_to_env(self):
        """When retrieval_config=None, router builds from env (existing behavior)."""
        from trelix.core.config import EmbedderConfig
        from trelix.retrieval.planner.agent import AdaptiveRouter

        router = AdaptiveRouter(EmbedderConfig())
        # _retrieval_config may be None (if env build fails) or a RetrievalConfig
        # Either is acceptable — the key is no TypeError
        assert router._retrieval_config is None or hasattr(
            router._retrieval_config, "short_query_lexical_enabled"
        )

    def test_router_programmatic_config_not_ignored(self):
        """Programmatic short_query config is honored when passed via retrieval_config."""
        from unittest.mock import patch
        from trelix.core.config import EmbedderConfig, RetrievalConfig
        from trelix.retrieval.planner.agent import AdaptiveRouter

        cfg = RetrievalConfig()
        cfg.short_query_lexical_enabled = True
        cfg.short_query_token_threshold = 5

        router = AdaptiveRouter(EmbedderConfig(), retrieval_config=cfg)

        with patch("trelix.retrieval.planner.agent.is_short_query", return_value=True):
            plan = router.route("login")

        for sq in plan.sub_queries:
            assert sq.lexical_only is True, (
                "lexical_only not set — router ignored the provided retrieval_config"
            )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/unit/test_planner_adaptive.py::TestAdaptiveRouterConfigPassthrough -v
```

Expected: `FAILED` — `AdaptiveRouter.__init__` does not accept `retrieval_config` parameter.

- [ ] **Step 3: Update `AdaptiveRouter.__init__` in `agent.py`**

In `src/trelix/retrieval/planner/agent.py`, change `__init__` (currently line 90):

```python
def __init__(
    self,
    config: EmbedderConfig,
    retrieval_config: "RetrievalConfig | None" = None,
) -> None:
    self._config = config
    # Lazy — only built when an LLM call is actually needed.
    self._planner: QueryPlanner | None = None
    # Use provided retrieval config, or fall back to building from env vars.
    # Accepting it as a parameter fixes the silent-ignore bug where programmatic
    # config overrides were lost because each AdaptiveRouter built its own instance.
    if retrieval_config is not None:
        self._retrieval_config: RetrievalConfig | None = retrieval_config
    else:
        try:
            from trelix.core.config import RetrievalConfig

            self._retrieval_config = RetrievalConfig()
        except Exception:
            self._retrieval_config = None
```

Add `RetrievalConfig` to the TYPE_CHECKING imports at the top of the file (if not already present):

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trelix.core.config import RetrievalConfig
```

- [ ] **Step 4: Update `Retriever.__init__` to pass `config.retrieval` through**

In `src/trelix/retrieval/retriever.py`, find where `QueryPlanner` is constructed (~line 97):

```python
self._planner = QueryPlanner(config.embedder)
```

Change to:

```python
self._planner = QueryPlanner(config.embedder, retrieval_config=config.retrieval)
```

Then find `QueryPlanner.__init__` in `src/trelix/retrieval/planner/agent.py` (the `QueryPlanner` class wraps `AdaptiveRouter` — check line ~350):

```bash
grep -n "class QueryPlanner\|def __init__" src/trelix/retrieval/planner/agent.py | head -10
```

Update `QueryPlanner.__init__` to accept and forward the `retrieval_config` parameter to `AdaptiveRouter`:

```python
def __init__(
    self,
    config: EmbedderConfig,
    retrieval_config: "RetrievalConfig | None" = None,
) -> None:
    self._router = AdaptiveRouter(config, retrieval_config=retrieval_config)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/unit/test_planner_adaptive.py::TestAdaptiveRouterConfigPassthrough -v
```

Expected: `3 passed`

- [ ] **Step 6: Run full planner + retriever test suite**

```bash
python -m pytest tests/unit/test_planner_adaptive.py tests/unit/test_planner.py tests/unit/test_retriever_core.py -v -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/trelix/retrieval/planner/agent.py src/trelix/retrieval/retriever.py \
        tests/unit/test_planner_adaptive.py
git commit -m "fix(retrieval): pass Retriever's RetrievalConfig into AdaptiveRouter

AdaptiveRouter was constructing its own RetrievalConfig() from env vars,
silently ignoring any programmatic override made on IndexConfig.retrieval
after construction. Now accepts retrieval_config= parameter (default None
falls back to env build for backward compat). Retriever passes config.retrieval
through QueryPlanner -> AdaptiveRouter."
```

---

## Task 4: Update ROADMAP to mark Phase 1 complete

**Files:**
- Modify: `docs/ROADMAP.md`

- [ ] **Step 1: Mark Phase 1 items shipped**

In `docs/ROADMAP.md`, update the v2.5.1 section:

```markdown
## 🐛 v2.5.1 — Backlog (bugs / hardening from v2.5.0)

- [x] **Watch bridge wired** — `notify_file_changed` fires from `FileWatcher._do_reindex` after successful re-index
- [x] **`files.rel_path` index** — `idx_files_rel_path` eliminates full table scan on watch events
- [x] **`AdaptiveRouter` config passthrough** — `RetrievalConfig` shared from `Retriever` instead of re-built from env
- [ ] **SparseEmbedder TOCTOU under parallel multi-query** — add `threading.Lock` around lazy-init path
- [ ] **`send_resource_notification` stdout isolation** — fix asyncio transport conflict when FastMCP writes notifications to stdout concurrently
- [ ] **`SubscriptionRegistry` max-subscriber cap / TTL eviction** — unbounded subscription growth
```

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/unit/ -x -q 2>&1 | tail -5
```

Expected: new test count increases by 5+ (3 watch bridge + 2 DB index + 3 router config).

- [ ] **Step 3: Lint check**

```bash
ruff check src/ tests/ && ruff format --check src/ tests/ && echo "CLEAN"
```

- [ ] **Step 4: Commit**

```bash
git add docs/ROADMAP.md
git commit -m "docs: mark Phase 1 items complete in ROADMAP"
```
