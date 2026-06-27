"""Unit tests for core models and config."""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.config import (
    EmbedderConfig,
    IndexConfig,
    RetrievalConfig,
    StoreConfig,
    WalkerConfig,
)
from trelix.core.models import (
    CallEdge,
    Chunk,
    IndexedFile,
    Language,
    Symbol,
    SymbolKind,
    TypeEdge,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestSymbolKind:
    def test_all_values_are_strings(self) -> None:
        for kind in SymbolKind:
            assert isinstance(kind.value, str)

    def test_expected_members(self) -> None:
        expected = {
            "function",
            "method",
            "class",
            "interface",
            "struct",
            "enum",
            "constant",
            "variable",
            "module",
            "section",
            "unknown",
        }
        assert {k.value for k in SymbolKind} == expected


class TestLanguage:
    def test_all_values_are_strings(self) -> None:
        for lang in Language:
            assert isinstance(lang.value, str)

    def test_common_languages_present(self) -> None:
        assert Language.PYTHON in Language
        assert Language.TYPESCRIPT in Language
        assert Language.GO in Language
        assert Language.RUST in Language
        assert Language.CSHARP in Language


class TestIndexedFile:
    def test_construction(self) -> None:
        f = IndexedFile(
            path="/repo/src/main.py",
            rel_path="src/main.py",
            language=Language.PYTHON,
            hash="abc123",
            size_bytes=1024,
        )
        assert f.id is None
        assert f.indexed_at is None
        assert f.language == Language.PYTHON


class TestSymbol:
    def test_construction_defaults(self) -> None:
        s = Symbol(
            file_id=1,
            name="authenticate",
            qualified_name="AuthService.authenticate",
            kind=SymbolKind.METHOD,
            line_start=10,
            line_end=30,
            signature="def authenticate(self, token: str) -> User",
            body="def authenticate(self, token: str) -> User:\n    ...",
        )
        assert s.id is None
        assert s.parent_id is None
        assert s.is_public is True
        assert s.decorators == []
        assert s.docstring is None


class TestCallEdge:
    def test_unresolved_callee(self) -> None:
        e = CallEdge(caller_id=1, callee_name="login", line=42)
        assert e.callee_id is None


class TestTypeEdge:
    def test_edge_kinds(self) -> None:
        for kind in ("extends", "implements", "trait_impl", "embedded"):
            e = TypeEdge(from_symbol_id=1, to_type_name="Base", edge_kind=kind)
            assert e.edge_kind == kind


class TestChunk:
    def test_no_embedding_by_default(self) -> None:
        c = Chunk(symbol_id=1, chunk_text="def foo(): ...", token_count=8)
        assert c.embedding is None
        assert c.id is None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestWalkerConfig:
    def test_default_languages_include_common(self) -> None:
        cfg = WalkerConfig()
        assert Language.PYTHON in cfg.languages
        assert Language.TYPESCRIPT in cfg.languages
        assert Language.GO in cfg.languages

    def test_trelix_dir_ignored(self) -> None:
        cfg = WalkerConfig()
        assert ".trelix" in cfg.extra_ignore_dirs

    def test_lock_files_ignored(self) -> None:
        cfg = WalkerConfig()
        assert "package-lock.json" in cfg.extra_ignore_filenames
        assert "yarn.lock" in cfg.extra_ignore_filenames


class TestEmbedderConfig:
    def test_default_provider_is_local(self) -> None:
        cfg = EmbedderConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.provider == "local"

    def test_local_dimension(self) -> None:
        cfg = EmbedderConfig(provider="local")
        assert cfg.effective_dimension == 384

    def test_openai_dimension(self) -> None:
        cfg = EmbedderConfig(provider="openai")
        assert cfg.effective_dimension == 3072

    def test_azure_dimension(self) -> None:
        cfg = EmbedderConfig(provider="azure")
        assert cfg.effective_dimension == 3072

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "openai")
        cfg = EmbedderConfig()
        assert cfg.provider == "openai"


class TestStoreConfig:
    def test_default_db_path(self) -> None:
        cfg = StoreConfig()
        assert cfg.db_path == ".trelix/index.db"

    def test_custom_db_path_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_STORE_DB_PATH", ".trelix/custom.db")
        cfg = StoreConfig()
        assert cfg.db_path == ".trelix/custom.db"


class TestRetrievalConfig:
    def test_defaults(self) -> None:
        cfg = RetrievalConfig()
        assert cfg.top_k_vector == 20
        assert cfg.rrf_k == 60
        assert cfg.context_token_budget == 12_000
        assert cfg.rerank is True


class TestIndexConfig:
    def test_repo_must_exist(self, tmp_path: Path) -> None:
        cfg = IndexConfig(repo_path=str(tmp_path))
        assert Path(cfg.repo_path).exists()

    def test_nonexistent_repo_raises(self) -> None:
        with pytest.raises(Exception):
            IndexConfig(repo_path="/nonexistent/path/xyz")

    def test_db_path_absolute_creates_dir(self, tmp_path: Path) -> None:
        cfg = IndexConfig(repo_path=str(tmp_path))
        db = cfg.db_path_absolute
        assert db.parent.exists()
        assert str(db).endswith("index.db")

    def test_db_path_gitignore_created(self, tmp_path: Path) -> None:
        cfg = IndexConfig(repo_path=str(tmp_path))
        cfg.db_path_absolute  # trigger creation
        gitignore = tmp_path / ".trelix" / ".gitignore"
        assert gitignore.exists()
        assert gitignore.read_text() == "*\n"

    def test_default_provider_is_local(self, tmp_path: Path) -> None:
        cfg = IndexConfig(repo_path=str(tmp_path), embedder={"_env_file": None})  # type: ignore[arg-type]
        assert cfg.embedder.provider == "local"


class TestRetrievalConfigQueryCache:
    def test_default_query_cache_size_is_256(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.query_cache_size == 256

    def test_zero_disables_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_QUERY_CACHE_SIZE", "0")
        cfg = RetrievalConfig()
        assert cfg.query_cache_size == 0

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_QUERY_CACHE_SIZE", "512")
        cfg = RetrievalConfig()
        assert cfg.query_cache_size == 512


class TestRetrievalConfigPlanCache:
    def test_default_plan_cache_size_is_128(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.plan_cache_size == 128

    def test_zero_disables_plan_cache(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig(plan_cache_size=0)
        assert cfg.plan_cache_size == 0

    def test_negative_plan_cache_size_raises(self) -> None:
        from pydantic import ValidationError

        from trelix.core.config import RetrievalConfig

        with pytest.raises(ValidationError):
            RetrievalConfig(plan_cache_size=-1)
