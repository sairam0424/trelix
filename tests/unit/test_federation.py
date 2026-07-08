"""Tests for multi-repo federation registry and retriever."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.federation.registry import RepoEntry, RepoRegistry
from trelix.federation.retriever import FederatedRetriever

# ---------------------------------------------------------------------------
# Helpers for cache tests
# ---------------------------------------------------------------------------


def _registry_with_paths(*paths: str) -> RepoRegistry:
    """Build a RepoRegistry with given paths without touching disk."""
    reg = RepoRegistry.__new__(RepoRegistry)
    reg._config_path = "/tmp/fake-registry.json"
    reg._entries = [RepoEntry(alias=f"r{i}", path=p, weight=1.0) for i, p in enumerate(paths)]
    return reg


def _mock_retriever_results(n: int = 3):
    results = []
    for i in range(n):
        r = MagicMock()
        r.symbol_id = f"sym_{i}"
        results.append(r)
    return results


class TestRepoEntry:
    def test_dataclass_fields(self) -> None:
        entry = RepoEntry(alias="myrepo", path="/Users/me/myrepo", weight=1.5)
        assert entry.alias == "myrepo"
        assert entry.weight == 1.5

    def test_default_weight_is_one(self) -> None:
        entry = RepoEntry(alias="r", path="/r")
        assert entry.weight == 1.0


class TestRepoRegistry:
    def test_load_empty_when_file_missing(self, tmp_path: Path) -> None:
        config = tmp_path / "repos.json"
        registry = RepoRegistry.load(str(config))
        assert registry.list() == []

    def test_add_and_list(self, tmp_path: Path) -> None:
        config = tmp_path / "repos.json"
        registry = RepoRegistry.load(str(config))
        registry.add("trelix", "/Users/me/trelix")
        registry.add("myapp", "/Users/me/myapp", weight=0.8)
        entries = registry.list()
        assert len(entries) == 2
        assert entries[0].alias == "trelix"
        assert entries[1].weight == 0.8

    def test_save_and_reload(self, tmp_path: Path) -> None:
        config = tmp_path / "repos.json"
        registry = RepoRegistry.load(str(config))
        registry.add("repo1", "/r1")
        registry.add("repo2", "/r2", weight=2.0)
        registry.save()
        reloaded = RepoRegistry.load(str(config))
        entries = reloaded.list()
        assert len(entries) == 2
        assert entries[1].alias == "repo2"
        assert entries[1].weight == 2.0

    def test_duplicate_alias_raises(self, tmp_path: Path) -> None:
        config = tmp_path / "repos.json"
        registry = RepoRegistry.load(str(config))
        registry.add("r", "/r1")
        with pytest.raises(ValueError, match="already registered"):
            registry.add("r", "/r2")

    def test_remove_entry(self, tmp_path: Path) -> None:
        config = tmp_path / "repos.json"
        registry = RepoRegistry.load(str(config))
        registry.add("r1", "/r1")
        registry.add("r2", "/r2")
        registry.remove("r1")
        assert len(registry.list()) == 1
        assert registry.list()[0].alias == "r2"

    def test_load_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        config = tmp_path / "repos.json"
        config.write_text("not json")
        registry = RepoRegistry.load(str(config))
        assert registry.list() == []


class TestFederatedRetriever:
    def test_retrieve_fans_out_to_all_repos(self, tmp_path: Path) -> None:
        from trelix.federation.retriever import FederatedRetriever

        registry = RepoRegistry.load(str(tmp_path / "repos.json"))
        registry.add("r1", str(tmp_path / "r1"))
        registry.add("r2", str(tmp_path / "r2"))

        mock_result = MagicMock()
        mock_result.chunk.symbol_id = 1
        mock_result.score = 0.9
        mock_result.source = "vector"
        mock_ctx = MagicMock()
        mock_ctx.results = [mock_result]

        with patch("trelix.federation.retriever.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            fed = FederatedRetriever(registry, max_workers=2)
            results = fed.retrieve("how does auth work", k=5)

        assert isinstance(results, list)
        # Called once per repo
        assert MockRetriever.return_value.retrieve.call_count == 2

    def test_retrieve_returns_empty_on_all_failures(self, tmp_path: Path) -> None:
        from trelix.federation.retriever import FederatedRetriever

        registry = RepoRegistry.load(str(tmp_path / "repos.json"))
        registry.add("bad", "/nonexistent/path")

        with patch("trelix.federation.retriever.Retriever") as MockRetriever:
            MockRetriever.side_effect = Exception("Index not found")
            fed = FederatedRetriever(registry, max_workers=1)
            results = fed.retrieve("test query", k=5)

        assert isinstance(results, list)  # Never raises

    def test_retrieve_empty_registry_returns_empty(self, tmp_path: Path) -> None:
        from trelix.federation.retriever import FederatedRetriever

        registry = RepoRegistry.load(str(tmp_path / "repos.json"))
        fed = FederatedRetriever(registry)
        results = fed.retrieve("test", k=5)
        assert results == []


# ---------------------------------------------------------------------------
# TTL cache tests
# ---------------------------------------------------------------------------


def test_federated_cache_hit_on_second_call() -> None:
    """Second identical query returns cached results without calling _query_repos."""
    reg = _registry_with_paths("/fake/repo1")
    fed = FederatedRetriever(reg, cache_ttl=60.0)

    mock_results = _mock_retriever_results(3)

    call_count = 0

    def fake_retrieve(query: str, k: int = 10):
        nonlocal call_count
        call_count += 1
        return mock_results

    with patch.object(fed, "_query_repos", side_effect=fake_retrieve):
        r1 = fed.retrieve("how does auth work", k=5)
        r2 = fed.retrieve("how does auth work", k=5)

    assert r1 == r2
    assert call_count == 1, "Second call should hit cache, not re-execute"


def test_federated_cache_miss_on_different_query() -> None:
    """Different query strings produce separate cache entries."""
    reg = _registry_with_paths("/fake/repo1")
    fed = FederatedRetriever(reg, cache_ttl=60.0)
    call_count = 0

    def fake_retrieve(query: str, k: int = 10):
        nonlocal call_count
        call_count += 1
        return _mock_retriever_results(2)

    with patch.object(fed, "_query_repos", side_effect=fake_retrieve):
        fed.retrieve("auth flow", k=5)
        fed.retrieve("login function", k=5)

    assert call_count == 2


def test_federated_cache_disabled_when_ttl_zero() -> None:
    """cache_ttl=0 disables caching — every call executes."""
    reg = _registry_with_paths("/fake/repo1")
    fed = FederatedRetriever(reg, cache_ttl=0)
    call_count = 0

    def fake_retrieve(query: str, k: int = 10):
        nonlocal call_count
        call_count += 1
        return _mock_retriever_results(2)

    with patch.object(fed, "_query_repos", side_effect=fake_retrieve):
        fed.retrieve("auth", k=5)
        fed.retrieve("auth", k=5)

    assert call_count == 2


def test_federated_cache_expires_after_ttl() -> None:
    """Cached entry expires after TTL seconds."""
    reg = _registry_with_paths("/fake/repo1")
    fed = FederatedRetriever(reg, cache_ttl=1.0)
    call_count = 0

    def fake_retrieve(query: str, k: int = 10):
        nonlocal call_count
        call_count += 1
        return _mock_retriever_results(2)

    with patch.object(fed, "_query_repos", side_effect=fake_retrieve):
        fed.retrieve("auth", k=5)
        # Simulate TTL expiry by advancing internal clock
        with patch("trelix.federation.retriever.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 120.0
            fed.retrieve("auth", k=5)

    assert call_count == 2


def test_federated_cache_stats() -> None:
    """cache_stats() returns hits/misses/size correctly."""
    reg = _registry_with_paths("/fake/repo1")
    fed = FederatedRetriever(reg, cache_ttl=60.0)

    def fake_retrieve(query: str, k: int = 10):
        return _mock_retriever_results(2)

    with patch.object(fed, "_query_repos", side_effect=fake_retrieve):
        fed.retrieve("q1", k=5)  # miss
        fed.retrieve("q1", k=5)  # hit
        fed.retrieve("q2", k=5)  # miss

    stats = fed.cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["size"] == 2


def test_federated_clear_cache() -> None:
    """clear_cache() forces next call to re-execute."""
    reg = _registry_with_paths("/fake/repo1")
    fed = FederatedRetriever(reg, cache_ttl=60.0)
    call_count = 0

    def fake_retrieve(query: str, k: int = 10):
        nonlocal call_count
        call_count += 1
        return _mock_retriever_results(2)

    with patch.object(fed, "_query_repos", side_effect=fake_retrieve):
        fed.retrieve("q1", k=5)
        fed.clear_cache()
        fed.retrieve("q1", k=5)

    assert call_count == 2


# ---------------------------------------------------------------------------
# Cross-repo symbol resolution (Plan A — Task A-1)
# ---------------------------------------------------------------------------


class TestCrossRepoSymbolResolution:
    def test_make_scip_symbol_id_is_deterministic(self):
        from trelix.federation.retriever import make_scip_symbol_id

        id1 = make_scip_symbol_id("myapp", "1.0.0", "AuthService.verify")
        id2 = make_scip_symbol_id("myapp", "1.0.0", "AuthService.verify")
        assert id1 == id2

    def test_make_scip_symbol_id_different_packages_differ(self):
        from trelix.federation.retriever import make_scip_symbol_id

        id1 = make_scip_symbol_id("app-a", "1.0.0", "login")
        id2 = make_scip_symbol_id("app-b", "1.0.0", "login")
        assert id1 != id2

    def test_resolve_symbol_returns_repo_that_defines_it(self, tmp_path):
        from unittest.mock import MagicMock

        from trelix.federation.retriever import FederatedRetriever, make_scip_symbol_id

        registry = MagicMock()
        registry.list.return_value = [MagicMock(alias="auth-service", path=str(tmp_path / "auth"))]
        fed = FederatedRetriever(registry)

        # Insert a symbol directly via the in-memory connection
        fed._fed_conn.execute(
            "INSERT INTO federation_symbols VALUES (?, ?, ?, ?, ?, ?)",
            (
                make_scip_symbol_id("auth-service", "", "AuthService.verify"),
                "auth-service",
                "",
                "AuthService.verify",
                "auth-service",
                "src/auth.py",
            ),
        )
        fed._fed_conn.commit()

        results = fed.resolve_symbol("AuthService.verify")
        assert len(results) == 1
        assert results[0]["alias"] == "auth-service"
        assert results[0]["file_path"] == "src/auth.py"

    def test_resolve_symbol_empty_when_not_found(self, tmp_path):
        from unittest.mock import MagicMock

        from trelix.federation.retriever import FederatedRetriever

        registry = MagicMock()
        registry.list.return_value = []
        fed = FederatedRetriever(registry)
        results = fed.resolve_symbol("NonExistentClass.method")
        assert results == []
