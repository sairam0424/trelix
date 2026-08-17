"""
Regression tests for per-language / per-leg retrieval weight env parsing.

Both `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_*` and `TRELIX_RETRIEVAL_LEG_WEIGHT_*` used to
reach a bare `float(val)` in `RetrievalConfig.model_post_init`. A typo in either one took
down every trelix command with "could not convert string to float: 'abc'" and no mention
of WHICH variable — leaving the user to bisect ~96 TRELIX_* aliases by hand. These tests
pin the variable name and the offending value into the message so that never regresses.

They also pin rejection of `nan`/`inf`, which `float()` accepts happily and which then
silently destroys ranking: every fused score becomes NaN, all comparisons return False,
and the result list collapses to insertion order with no error anywhere.
"""

from __future__ import annotations

import math

import pytest


class TestFileTypeWeightParseErrors:
    """A malformed TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_* must name itself in the error."""

    def test_non_numeric_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_PYTHON", "abc")
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        message = str(exc_info.value)
        assert "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_PYTHON" in message
        assert "abc" in message

    def test_partially_numeric_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'0.7x' is the realistic typo — a trailing keystroke, not obvious junk."""
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN", "0.7x")
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        message = str(exc_info.value)
        assert "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN" in message
        assert "0.7x" in message

    def test_empty_value_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`export TRELIX_..._WEIGHT_GO=` sets an empty string, which float() rejects."""
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_GO", "")
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        assert "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_GO" in str(exc_info.value)

    def test_nan_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """float('nan') parses, then makes every fused score NaN and ranking arbitrary."""
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_PYTHON", "nan")
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        message = str(exc_info.value)
        assert "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_PYTHON" in message
        assert "nan" in message

    def test_infinity_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN", "inf")
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        assert "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN" in str(exc_info.value)


class TestLegWeightParseErrors:
    """The second, identically-unguarded site: TRELIX_RETRIEVAL_LEG_WEIGHT_*."""

    def test_non_numeric_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_BM25", "heavy")
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        message = str(exc_info.value)
        assert "TRELIX_RETRIEVAL_LEG_WEIGHT_BM25" in message
        assert "heavy" in message

    def test_partially_numeric_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_VECTOR", "1.2.3")
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        message = str(exc_info.value)
        assert "TRELIX_RETRIEVAL_LEG_WEIGHT_VECTOR" in message
        assert "1.2.3" in message

    def test_nan_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_SPARSE", "NaN")
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        assert "TRELIX_RETRIEVAL_LEG_WEIGHT_SPARSE" in str(exc_info.value)

    def test_negative_infinity_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_GREP", "-Infinity")
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        assert "TRELIX_RETRIEVAL_LEG_WEIGHT_GREP" in str(exc_info.value)


class TestBadWeightIsNotSilentlyIgnored:
    """
    Rejecting is deliberate, not incidental. Weights exist so the user can A/B two
    rankings; a bad value that got warned-about-and-dropped would leave them measuring
    the DEFAULT while believing they measured their override — a wrong conclusion drawn
    from a run that looked successful. Failing is the only outcome that cannot mislead.
    """

    def test_malformed_weight_does_not_fall_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_PYTHON", "abc")
        with pytest.raises(ValueError):
            RetrievalConfig()

    def test_error_does_not_dump_unrelated_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The message must point at one variable, not paste the whole environment."""
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_LLM_OPENAI_API_KEY", "sk-must-not-appear-in-error")
        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_BM25", "oops")
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        assert "sk-must-not-appear-in-error" not in str(exc_info.value)


class TestValidWeightsStillParse:
    """Guard the fix against over-reach: everything legitimate must keep working."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0.5", 0.5),
            ("1", 1.0),
            ("0", 0.0),
            ("1e-2", 0.01),
            ("  0.25  ", 0.25),  # float() tolerates padding; shells make this easy to hit
            ("-0.5", -0.5),  # sign is not ours to police — only non-finite is broken
        ],
    )
    def test_file_type_weight_accepts_valid_floats(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: float
    ) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_PYTHON", raw)
        cfg = RetrievalConfig()
        assert cfg.file_type_weights["python"] == expected
        # Untouched languages must still hold their defaults.
        assert cfg.file_type_weights["markdown"] == 0.3

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("0.7", 0.7), ("1.2", 1.2), ("0", 0.0), ("  1.0 ", 1.0)],
    )
    def test_leg_weight_accepts_valid_floats(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: float
    ) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_BM25", raw)
        cfg = RetrievalConfig()
        assert cfg.leg_weights["bm25"] == expected
        assert cfg.leg_weights["vector"] == 1.0

    def test_no_weight_env_vars_leaves_defaults_finite(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert all(math.isfinite(v) for v in cfg.file_type_weights.values())
        assert all(math.isfinite(v) for v in cfg.leg_weights.values())
