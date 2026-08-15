"""Unit tests for core models and config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

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
from trelix.retrieval.planner.models import (
    INTENT_STRATEGIES,
    IntentType,
    compression_ratio_for_intent,
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

    def test_federation_max_repos_default(self) -> None:
        cfg = RetrievalConfig()
        assert cfg.federation_max_repos == 50

    def test_federation_max_repos_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_FEDERATION_MAX_REPOS", "10")
        cfg = RetrievalConfig()
        assert cfg.federation_max_repos == 10

    def test_flare_max_iter_env_emits_deprecation_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test: TRELIX_RETRIEVAL_FLARE_MAX_ITER emits DeprecationWarning.

        This test ensures that the old environment variable name triggers a
        deprecation warning naming a removal target. The old name should still work
        (backward compat via AliasChoices) but warn at runtime.

        The target version is matched by SHAPE, not as a literal. This test used to
        assert the string "v3.0.0"; v3.0.0 then shipped without the removal, the
        message was retargeted to v4.0.0, and the test broke on a change that was
        correcting a false statement. What matters is that the warning tells the user
        *when* the name goes away — not which release that is this quarter.
        """
        import re
        import warnings

        monkeypatch.setenv("TRELIX_RETRIEVAL_FLARE_MAX_ITER", "1")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = RetrievalConfig()

        # Verify:
        # 1. A DeprecationWarning was emitted
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1, (
            f"Expected at least 1 DeprecationWarning, got {len(deprecation_warnings)}"
        )

        # 2. Warning message mentions the old env var name
        warning_msg = str(deprecation_warnings[0].message)
        assert "TRELIX_RETRIEVAL_FLARE_MAX_ITER" in warning_msg, (
            f"Expected old env var name in warning: {warning_msg}"
        )

        # 3. Warning message names SOME removal target version
        assert re.search(r"removed in v\d+\.\d+\.\d+", warning_msg), (
            f"Expected the warning to name a removal target version: {warning_msg}"
        )

        # 4. Backward compat worked: the value was parsed correctly
        assert cfg.flare_max_retries == 1

    def test_agent_session_max_age_default(self) -> None:
        cfg = RetrievalConfig()
        assert cfg.agent_session_max_age_seconds == 604_800.0

    def test_agent_session_max_age_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS", "0")
        cfg = RetrievalConfig()
        assert cfg.agent_session_max_age_seconds == 0.0


class TestContextTokenBudgetEnvCoercion:
    """Regression: the DOCUMENTED ``=null`` env route used to raise.

    ``TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null`` — the exact spelling in the
    field's own docstring and in docs/CONFIGURATION.md — raised
    ValidationError ("Input should be a valid integer"), because env values
    arrive as strings and an ``int | None`` field cannot coerce ``"null"``.
    Since the whole model-aware auto-budget path is gated on
    ``context_token_budget is None``, v3.0's headline feature was reachable
    ONLY from the Python API — never from env, and therefore never from the
    CLI. A ``mode="before"`` validator now maps the auto sentinels to None.

    Budget *resolution* itself is covered in tests/unit/test_model_aware_budget.py;
    what is pinned here is the env plumbing that feeds it.
    """

    VAR = "TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET"

    @pytest.mark.parametrize(
        "raw",
        ["null", "none", "auto", "~", "", "NULL", "None", "AUTO", " null ", "  auto\t"],
    )
    def test_auto_sentinels_become_none(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(self.VAR, raw)
        assert RetrievalConfig().context_token_budget is None

    def test_explicit_int_string_is_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(self.VAR, "12000")
        budget = RetrievalConfig().context_token_budget
        assert budget == 12_000
        assert isinstance(budget, int)

    def test_unset_var_keeps_the_12000_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # delenv alone would not prove "unset": pydantic-settings reads ./.env
        # directly, and this repo's root .env carries 30+ TRELIX_* vars, so
        # _env_file=None is what actually isolates the file source.
        monkeypatch.delenv(self.VAR, raising=False)
        cfg = RetrievalConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.context_token_budget == 12_000

    def test_genuinely_invalid_value_still_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The coercion must not swallow a real typo into a silent auto-budget."""
        monkeypatch.setenv(self.VAR, "banana")
        with pytest.raises(ValidationError):
            RetrievalConfig()

    def test_env_null_actually_reaches_the_auto_budget_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The point of the fix: the env route must engage the model-aware budget.

        Without the validator this raised at IndexConfig construction, so no
        env/CLI user could ever get an auto-derived budget.
        """
        from trelix.retrieval.retriever import Retriever

        monkeypatch.setenv(self.VAR, "null")
        # Pinned so a developer's .env cannot change the arithmetic below.
        monkeypatch.setenv("TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION", "0.5")
        repo = tmp_path / "repo"
        repo.mkdir()

        cfg = IndexConfig(repo_path=str(repo))
        assert cfg.retrieval.context_token_budget is None  # nested model saw the env var
        cfg.llm.model = "gpt-4o"

        # gpt-4o's 128k window x the 0.5 default fraction.
        assert Retriever(cfg)._effective_budget == 64_000


class TestSparseConfig:
    def test_sparse_defaults(self, tmp_path: Path) -> None:
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path), _env_file=None)
        assert cfg.retrieval.sparse_enabled is False
        assert cfg.retrieval.top_k_sparse == 20
        assert cfg.sparse.top_k_tokens == 128
        assert "splade" in cfg.sparse.model.lower()


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

    def test_file_summaries_enabled_kwarg_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing file_summaries_enabled=True by field name (not alias) must
        take effect — regression test for the missing populate_by_name=True
        on IndexConfig's model_config, which previously let this kwarg be
        silently ignored (falling back to the alias env var / default)."""
        monkeypatch.delenv("TRELIX_FILE_SUMMARIES_ENABLED", raising=False)
        cfg = IndexConfig(repo_path=str(tmp_path), file_summaries_enabled=True, _env_file=None)
        assert cfg.file_summaries_enabled is True

    def test_telemetry_enabled_kwarg_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same regression coverage as above for telemetry_enabled."""
        monkeypatch.delenv("TRELIX_TELEMETRY_ENABLED", raising=False)
        cfg = IndexConfig(repo_path=str(tmp_path), telemetry_enabled=True, _env_file=None)
        assert cfg.telemetry_enabled is True


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


class TestRetrievalConfigFileTypeWeighting:
    def test_file_type_weighting_enabled_default_true(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.file_type_weighting_enabled is True

    def test_weighting_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING", "false")
        cfg = RetrievalConfig()
        assert cfg.file_type_weighting_enabled is False

    def test_default_weights_contain_all_expected_languages(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        expected_languages = {
            "python",
            "javascript",
            "typescript",
            "tsx",
            "go",
            "rust",
            "java",
            "kotlin",
            "ruby",
            "cpp",
            "c",
            "csharp",
            "razor",
            "cshtml",
            "csproj",
            "html",
            "css",
            "json",
            "yaml",
            "toml",
            "markdown",
            "unknown",
        }
        assert expected_languages.issubset(set(cfg.file_type_weights.keys()))

    def test_default_python_weight_is_1_0(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.file_type_weights["python"] == 1.0

    def test_default_markdown_weight_is_0_3(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.file_type_weights["markdown"] == 0.3

    def test_default_unknown_weight_is_0_8(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.file_type_weights["unknown"] == 0.8

    def test_per_language_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN", "0.1")
        cfg = RetrievalConfig()
        assert cfg.file_type_weights["markdown"] == 0.1
        # Other keys must still be at defaults
        assert cfg.file_type_weights["python"] == 1.0

    def test_full_json_dict_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv(
            "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS",
            '{"markdown": 0.05, "yaml": 0.6}',
        )
        cfg = RetrievalConfig()
        assert cfg.file_type_weights["markdown"] == 0.05
        assert cfg.file_type_weights["yaml"] == 0.6
        # Defaults for other keys untouched
        assert cfg.file_type_weights["python"] == 1.0

    def test_per_language_override_beats_json_dict_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-language env var is highest priority — applied after JSON dict override."""
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv(
            "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS",
            '{"markdown": 0.15}',
        )
        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN", "0.02")
        cfg = RetrievalConfig()
        # Per-language override wins
        assert cfg.file_type_weights["markdown"] == 0.02


class TestRetrievalConfigLegWeights:
    def test_default_all_legs_weight_1_0(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.leg_weights == {
            "vector": 1.0,
            "bm25": 1.0,
            "grep": 1.0,
            "summary": 1.0,
            "sub_chunk": 1.0,
            "sparse": 1.0,
        }

    def test_per_leg_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_BM25", "0.7")
        cfg = RetrievalConfig()
        assert cfg.leg_weights["bm25"] == 0.7
        # Other legs must still be at defaults
        assert cfg.leg_weights["vector"] == 1.0
        assert cfg.leg_weights["grep"] == 1.0

    def test_multiple_per_leg_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_VECTOR", "1.2")
        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_SPARSE", "0.0")
        cfg = RetrievalConfig()
        assert cfg.leg_weights["vector"] == 1.2
        assert cfg.leg_weights["sparse"] == 0.0
        assert cfg.leg_weights["bm25"] == 1.0


class TestRetrievalConfigDeclarationBoost:
    def test_default_disabled_weight_1_0(self) -> None:
        cfg = RetrievalConfig()
        assert cfg.declaration_boost_enabled is False
        assert cfg.declaration_boost_weight == 1.0

    def test_env_override_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_DECLARATION_BOOST", "true")
        monkeypatch.setenv("TRELIX_RETRIEVAL_DECLARATION_BOOST_WEIGHT", "3.0")
        cfg = RetrievalConfig()
        assert cfg.declaration_boost_enabled is True
        assert cfg.declaration_boost_weight == 3.0

    def test_weight_below_lower_bound_raises(self) -> None:
        with pytest.raises(Exception):
            RetrievalConfig(declaration_boost_weight=0.5)


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


# ---------------------------------------------------------------------------
# RetrievalConfig — flare_max_retries rename (v2.4.0)
# ---------------------------------------------------------------------------


class TestRetrievalConfigFlareMaxRetries:
    def test_flare_max_retries_new_name(self) -> None:
        """New field name is accessible as flare_max_retries."""
        cfg = RetrievalConfig(flare_max_retries=2)
        assert cfg.flare_max_retries == 2

    def test_flare_max_retries_new_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TRELIX_RETRIEVAL_FLARE_MAX_RETRIES env var is accepted."""
        monkeypatch.setenv("TRELIX_RETRIEVAL_FLARE_MAX_RETRIES", "3")
        cfg = RetrievalConfig()
        assert cfg.flare_max_retries == 3

    def test_flare_max_iterations_old_env_var_still_works(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Old TRELIX_RETRIEVAL_FLARE_MAX_ITER env var still works (backward compat)."""
        monkeypatch.setenv("TRELIX_RETRIEVAL_FLARE_MAX_ITER", "2")
        cfg = RetrievalConfig()
        assert cfg.flare_max_retries == 2

    def test_flare_max_iterations_old_env_var_emits_deprecation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Old TRELIX_RETRIEVAL_FLARE_MAX_ITER emits DeprecationWarning."""
        import warnings

        monkeypatch.setenv("TRELIX_RETRIEVAL_FLARE_MAX_ITER", "2")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RetrievalConfig()
        assert any(
            issubclass(warning.category, DeprecationWarning)
            and "TRELIX_RETRIEVAL_FLARE_MAX_ITER" in str(warning.message)
            for warning in w
        ), "Expected DeprecationWarning mentioning old env var name"

    def test_flare_max_retries_default(self) -> None:
        """Default value is still 1."""
        cfg = RetrievalConfig()
        # Make sure old attribute name does NOT exist
        assert not hasattr(cfg, "flare_max_iterations"), (
            "Old field name 'flare_max_iterations' must be removed"
        )
        assert cfg.flare_max_retries == 1


class TestShortQueryConfig:
    def test_short_query_disabled_by_default(self) -> None:
        cfg = RetrievalConfig()
        assert cfg.short_query_lexical_enabled is False

    def test_short_query_token_threshold_default_is_5(self) -> None:
        cfg = RetrievalConfig()
        assert cfg.short_query_token_threshold == 5

    def test_short_query_env_var_enables_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_SHORT_QUERY_LEXICAL", "true")
        cfg = RetrievalConfig()
        assert cfg.short_query_lexical_enabled is True

    def test_short_query_threshold_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_SHORT_QUERY_TOKENS", "3")
        cfg = RetrievalConfig()
        assert cfg.short_query_token_threshold == 3


# ---------------------------------------------------------------------------
# Context compression (SeleCom)
# ---------------------------------------------------------------------------


class TestCompressionConfig:
    def test_compression_disabled_by_default(self) -> None:
        assert RetrievalConfig().compression_enabled is False

    def test_provider_defaults_to_extractive(self) -> None:
        assert RetrievalConfig().compression_provider == "extractive"

    def test_target_ratio_default_is_0_45(self) -> None:
        assert RetrievalConfig().compression_target_ratio == 0.45

    def test_min_tokens_default_is_120(self) -> None:
        assert RetrievalConfig().compression_min_tokens == 120

    def test_enabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION", "true")
        assert RetrievalConfig().compression_enabled is True

    def test_provider_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_PROVIDER", "extractive")
        assert RetrievalConfig().compression_provider == "extractive"

    def test_unknown_provider_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_PROVIDER", "abstractive")
        with pytest.raises(ValidationError):
            RetrievalConfig()

    def test_target_ratio_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO", "0.7")
        assert RetrievalConfig().compression_target_ratio == 0.7

    def test_min_tokens_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_MIN_TOKENS", "0")
        assert RetrievalConfig().compression_min_tokens == 0

    def test_ratio_below_lower_bound_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO", "0.05")
        with pytest.raises(ValidationError):
            RetrievalConfig()

    def test_ratio_above_upper_bound_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO", "1.5")
        with pytest.raises(ValidationError):
            RetrievalConfig()

    def test_ratio_accepts_both_bounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO", "0.1")
        assert RetrievalConfig().compression_target_ratio == 0.1
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO", "1.0")
        assert RetrievalConfig().compression_target_ratio == 1.0

    def test_negative_min_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_MIN_TOKENS", "-1")
        with pytest.raises(ValidationError):
            RetrievalConfig()

    def test_kwarg_by_name(self) -> None:
        cfg = RetrievalConfig(compression_enabled=True, compression_target_ratio=0.9)
        assert cfg.compression_enabled is True
        assert cfg.compression_target_ratio == 0.9


class TestPerIntentCompressionRatio:
    """Per-intent ratios live on RetrievalStrategy; 1.0 means 'never compress'."""

    EXPECTED: dict[IntentType, float] = {
        IntentType.SYMBOL_LOOKUP: 1.0,
        IntentType.CONFIG_LOOKUP: 1.0,
        IntentType.FILE_OVERVIEW: 1.0,
        IntentType.PROJECT_OVERVIEW: 1.0,
        IntentType.FEATURE_FLOW: 0.45,
        IntentType.DEPENDENCY_MAP: 0.30,
        IntentType.BLAST_RADIUS: 0.30,
        IntentType.COMPARISON: 0.65,
    }

    def test_every_intent_has_a_baked_ratio(self) -> None:
        assert set(INTENT_STRATEGIES) == set(self.EXPECTED)

    def test_baked_defaults_match_the_spec(self) -> None:
        actual = {intent: s.compression_ratio for intent, s in INTENT_STRATEGIES.items()}
        assert actual == self.EXPECTED

    def test_resolver_returns_the_baked_ratio(self) -> None:
        for intent, expected in self.EXPECTED.items():
            assert compression_ratio_for_intent(intent) == expected

    def test_resolver_accepts_the_intent_string(self) -> None:
        assert compression_ratio_for_intent("blast_radius") == 0.30

    def test_unknown_intent_returns_none_for_config_fallback(self) -> None:
        assert compression_ratio_for_intent("not_an_intent") is None

    def test_absent_intent_returns_none_for_config_fallback(self) -> None:
        assert compression_ratio_for_intent(None) is None

    def test_per_intent_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO_BLAST_RADIUS", "0.8")
        assert compression_ratio_for_intent(IntentType.BLAST_RADIUS) == 0.8
        # Other intents are unaffected — one var per intent, like leg_weights.
        assert compression_ratio_for_intent(IntentType.DEPENDENCY_MAP) == 0.30

    def test_env_override_can_disable_compression_for_one_intent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO_FEATURE_FLOW", "1.0")
        assert compression_ratio_for_intent(IntentType.FEATURE_FLOW) == 1.0

    def test_env_override_can_enable_compression_for_an_opted_out_intent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO_SYMBOL_LOOKUP", "0.5")
        assert compression_ratio_for_intent(IntentType.SYMBOL_LOOKUP) == 0.5

    def test_unparseable_env_override_falls_back_to_baked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO_COMPARISON", "aggressive")
        assert compression_ratio_for_intent(IntentType.COMPARISON) == 0.65

    def test_out_of_range_env_override_falls_back_to_baked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO_COMPARISON", "0.0")
        assert compression_ratio_for_intent(IntentType.COMPARISON) == 0.65
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO_COMPARISON", "2.0")
        assert compression_ratio_for_intent(IntentType.COMPARISON) == 0.65

    def test_scalar_ratio_env_var_is_not_read_as_a_per_intent_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TRELIX_RETRIEVAL_COMPRESSION_RATIO must not leak into any intent."""
        monkeypatch.setenv("TRELIX_RETRIEVAL_COMPRESSION_RATIO", "0.99")
        assert compression_ratio_for_intent(IntentType.FEATURE_FLOW) == 0.45
