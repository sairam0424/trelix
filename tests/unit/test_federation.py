"""Tests for multi-repo federation registry and retriever."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.federation.registry import RepoEntry, RepoRegistry


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
