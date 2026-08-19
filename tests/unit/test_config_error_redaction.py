"""
A configuration error must never render API-key material.

Measured leak (pydantic 2.13.4 / pydantic-settings 2.14.2) that these tests pin shut:
a ValueError raised from `RetrievalConfig.model_post_init` was caught by pydantic and
re-raised as a ValidationError carrying the model's entire input dict, so one typo'd
`TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_*` printed key material:

    str(exc)     -> "... input_value={'COHERE_API_KEY': 'co-FA...3b6a1f5e7c2d40aabbccdd'}"
    exc.errors() -> [{'input': {'COHERE_API_KEY': 'co-FAKE-<the whole key>'}}]

str() and the traceback showed the trailing ~22 characters verbatim; errors() and
json() showed the key in full. `cohere_api_key` (alias COHERE_API_KEY) is a field of
RetrievalConfig itself, so no settings-source filtering can keep it out of that dict —
the fix has to be at the raise site, and that is what these tests assert.

The assertions are on the ABSENCE of the key, over every rendering a human or a log
might see, not on the presence of a nice message: a future refactor that reintroduces
pydantic wrapping must fail here.
"""

from __future__ import annotations

import traceback

import pytest
from pydantic import ValidationError

# Long enough that the tail survives the 50-character repr truncation pydantic applies to
# input_value, which is the property this test needs — that truncation keeps the head AND the
# tail, so a short value would not demonstrate the leak.
#
# It deliberately carries NO vendor prefix. The first version began `co-`, and GitGuardian
# failed the PR on it — correctly, by its own rules: a high-entropy string with a Cohere-key
# prefix, committed. The right response is not to allowlist the path. A scanner that has been
# taught to ignore `tests/` is a scanner that will miss a real key parked there, and this is
# the one control standing between a typo and a published credential. So the fixture changes
# instead: the prefix contributed nothing the assertions rely on, since what they need is
# length and entropy, and the env var name is what makes trelix treat it as a Cohere key.
FAKE_KEY = "NOT-A-REAL-KEY-Nq7Rv2sX8dLm4Tp1Wz6Yb3Kc9Ae5Gh0J"

_FRAGMENT_WIDTH = 8


def _fragments(secret: str, width: int = _FRAGMENT_WIDTH) -> set[str]:
    """Every contiguous `width`-char window of `secret`.

    Substring windows, not the whole string: the observed leak was truncated in the
    middle, so an `if FAKE_KEY in text` check passes while 22 characters of the key sit
    in the log. Any window escaping is a leak.
    """
    return {secret[i : i + width] for i in range(len(secret) - width + 1)}


def _renderings(exc: BaseException) -> dict[str, str]:
    """Every form of `exc` that can reach a terminal, a CI log or a pasted issue."""
    out = {
        "str": str(exc),
        "repr": repr(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    if isinstance(exc, ValidationError):
        # The CLI renders errors()[0]; structured logging and exc.json() dump the lot.
        out["errors"] = repr(exc.errors())
        out["json"] = exc.json()
    return out


def _assert_no_key(exc: BaseException) -> None:
    leaks = {
        name: sorted(f for f in _fragments(FAKE_KEY) if f in text)
        for name, text in _renderings(exc).items()
    }
    leaked = {name: found for name, found in leaks.items() if found}
    assert not leaked, f"API-key fragments rendered by {type(exc).__name__}: {leaked}"


@pytest.fixture
def isolated_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A cwd with no .env and a fake COHERE_API_KEY in the environment.

    chdir matters twice: RetrievalConfig has `env_file=".env"`, so running from the
    repo root would load the developer's real keys into the settings input dict — the
    very thing under test — and a failure here would print them.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COHERE_API_KEY", FAKE_KEY)
    return tmp_path


class TestMalformedWeightDoesNotLeakKeys:
    """The reported case: a bad retrieval weight aborts config construction."""

    @pytest.mark.parametrize(
        "env_var",
        [
            "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_PYTHON",
            "TRELIX_RETRIEVAL_LEG_WEIGHT_VECTOR",
        ],
    )
    @pytest.mark.parametrize("bad_value", ["0.7x", "nan", "-inf"])
    def test_retrieval_config(
        self, isolated_env, monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str
    ) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv(env_var, bad_value)
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        _assert_no_key(exc_info.value)
        # The error still has to be actionable — that is the point of raising at all.
        assert env_var in str(exc_info.value)

    def test_index_config(self, isolated_env, monkeypatch: pytest.MonkeyPatch) -> None:
        """IndexConfig builds RetrievalConfig via default_factory — the CLI's path."""
        from trelix.core.config import IndexConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN", "0.3x")
        with pytest.raises(ValueError) as exc_info:
            IndexConfig(repo_path=str(isolated_env))

        _assert_no_key(exc_info.value)

    def test_dotenv_secret_is_not_rendered(
        self, isolated_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dotenv file is the other input route: every key in it lands in the input dict.

        pydantic-settings' DotEnvSettingsSource copies unmatched dotenv keys into the
        settings input verbatim, so OPENAI_API_KEY reached RetrievalConfig even though
        RetrievalConfig has no such field.

        The file is named explicitly via ``_env_file``. Since SEC-03a the models no
        longer read the cwd's ``.env`` at all, so writing one here would leave this
        test passing while exercising nothing — the redaction path under test is the
        dotenv source, not the location trelix picks it up from.
        """
        from trelix.core.config import RetrievalConfig

        monkeypatch.delenv("COHERE_API_KEY")
        dotenv = isolated_env / ".env"
        dotenv.write_text(f"OPENAI_API_KEY={FAKE_KEY}\n", encoding="utf-8")
        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_BM25", "abc")

        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig(_env_file=str(dotenv))  # type: ignore[call-arg]

        _assert_no_key(exc_info.value)


class TestErrorTypeKeepsTheInputDictOut:
    """Pin the mechanism, not just the symptom."""

    def test_not_a_validation_error(self, isolated_env, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ValidationError always carries `input`; a plain ValueError cannot.

        Raising before pydantic sees the value is what makes redaction structural
        rather than a matter of remembering to scrub a message.
        """
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_GO", "1.0.0")
        with pytest.raises(ValueError) as exc_info:
            RetrievalConfig()

        assert not isinstance(exc_info.value, ValidationError)
        assert type(exc_info.value) is ValueError

    def test_still_a_valueerror_for_cli_handlers(
        self, isolated_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cli/main.py pairs every ValidationError handler with `except (ValueError, ...)`.

        Raising a non-ValueError would escape all ten of those handlers as a traceback.
        """
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_GREP", "")
        with pytest.raises(ValueError):
            RetrievalConfig()


class TestValidWeightsStillApply:
    """Guard the refactor: moving the parse earlier must not drop the override."""

    def test_file_type_and_leg_overrides(
        self, isolated_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_PYTHON", "0.5")
        monkeypatch.setenv("TRELIX_RETRIEVAL_LEG_WEIGHT_VECTOR", "1.25")
        config = RetrievalConfig()

        assert config.file_type_weights["python"] == 0.5
        assert config.leg_weights["vector"] == 1.25
        # Untouched keys keep their defaults (the merge, not a replace).
        assert config.file_type_weights["markdown"] == 0.3
        assert config.leg_weights["bm25"] == 1.0
