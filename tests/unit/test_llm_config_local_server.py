"""`LLMConfig.base_url` (env `TRELIX_LLM_BASE_URL`): the shapes it accepts, the shapes it
refuses, and the rule that a refusal never echoes the value.

The value is operator configuration that may carry a pasted credential
(`http://user:pw@host/v1`). pydantic 2.13 renders `input_value=...` in `str(ValidationError)`
unless the model hides it, and `trelix review` built its IndexConfig with no handler, so such
a value reached stderr and the Actions log. Every expected value below is a literal.

`LLMConfig.local_context_tokens` (env `TRELIX_LLM_LOCAL_CONTEXT_TOKENS`) describes the server
behind that URL: an optional int, 1024..2_000_000, blank is unset, refused without the URL.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.core.config import IndexConfig, LLMConfig

_SHAPE_ERROR = "TRELIX_LLM_BASE_URL must be an http:// or https:// URL with a host"
_USERINFO_ERROR = "TRELIX_LLM_BASE_URL must not carry a user name or password"
_WHITESPACE_ERROR = "TRELIX_LLM_BASE_URL must not contain whitespace or non-printable characters"
_LOCAL_CONTEXT_TOKENS_ERROR = (
    "TRELIX_LLM_LOCAL_CONTEXT_TOKENS needs TRELIX_LLM_BASE_URL: "
    "it describes the server behind that URL"
)
_WITH_CREDENTIAL = "http://user:pw@host/v1"
_LOCAL_URL = "http://127.0.0.1:11434/v1"
_ONE_HUNK_DIFF = (
    "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,2 @@\n x = 1\n+y = 2\n"
)


def _llm(**fields: object) -> LLMConfig:
    return LLMConfig(_env_file=None, **fields)  # type: ignore[call-arg]


class TestAcceptedValues:
    @pytest.mark.parametrize(
        ("raw", "stored"),
        [
            ("", None),
            ("   ", None),
            ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1"),
            ("https://llm.internal/v1", "https://llm.internal/v1"),
            ("http://[::1]:11434/v1", "http://[::1]:11434/v1"),
            # The scheme check lower-cases a copy; the value is stored as written.
            ("HTTP://Host/v1", "HTTP://Host/v1"),
            # No trailing-slash edits either; the SDK normalises for itself.
            ("http://127.0.0.1:11434/v1/", "http://127.0.0.1:11434/v1/"),
            # `@` in the path is not userinfo; the SDK accepts this URL, so the validator must.
            ("http://127.0.0.1:11434/v1@foo", "http://127.0.0.1:11434/v1@foo"),
        ],
    )
    def test_blank_is_unset_and_a_url_is_stored_verbatim(
        self, raw: str, stored: str | None
    ) -> None:
        """MUTATION: drop the mode="before" blank validator (the blank rows then hit the shape
        error); normalise the URL (the verbatim rows fail); check `"@" in value` instead of
        `parts.netloc` (the `/v1@foo` row is refused as userinfo)."""
        assert _llm(base_url=raw).base_url == stored

    def test_the_default_is_unset(self) -> None:
        assert LLMConfig.model_fields["base_url"].default is None

    def test_another_provider_may_carry_the_url(self) -> None:
        """MUTATION: a cross-field `provider != "openai"` error in LLMConfig.

        Three retrieval shims build `LLMConfig(provider=<the embedder's provider>)` with the
        URL still in the environment; a config error there would break `search` and `ask`
        for an Azure-embedder operator whose review configuration is valid.
        """
        assert _llm(provider="anthropic", base_url=_LOCAL_URL).base_url == _LOCAL_URL


class TestRefusedValues:
    @pytest.mark.parametrize(
        "raw",
        [
            "localhost:11434",
            "ftp://x/v1",
            "http://",
            "/v1",
            "127.0.0.1:11434/v1",
            # A port the SDK's own URL parser would reject; refused here, not as exit 3.
            "http://127.0.0.1:abc/v1",
            "http://host:99999/v1",
            # urlsplit's own ValueError (`Invalid IPv6 URL`), mapped to the shape error.
            "http://[::1/v1",
        ],
    )
    def test_not_an_http_url_with_a_host(self, raw: str) -> None:
        """MUTATION: accept any scheme (ftp passes); drop the hostname check (`http://`
        passes); drop the port check (`:abc` and `:99999` pass); move `urlsplit` out of the
        `try` (`http://[::1/v1` reports `Invalid IPv6 URL` instead of the shape error)."""
        with pytest.raises(ValidationError) as exc_info:
            _llm(base_url=raw)
        assert _SHAPE_ERROR in str(exc_info.value)

    @pytest.mark.parametrize(
        "raw",
        [
            # urlsplit drops every \t, \r and \n and leading C0/space characters before parsing,
            # so each of these passed the shape check and reached the SDK: httpx refuses the
            # control characters (which `trelix review` reported as "LLM not configured") and
            # percent-encodes the spaces into every request path (`/v1%20/chat/completions`).
            "http://127.0.0.1:11434/v1\n",
            "http://127.0.0.1:11434/v1\r",
            "http://127.0.0.1:11434/v1\t",
            "\x01http://127.0.0.1:11434/v1",
            " http://127.0.0.1:11434/v1",
            "http://127.0.0.1:11434/v1 ",
            "http://127.0.0.1:11434/v 1",
            # U+200B zero-width space: not whitespace to `strip()`, not printable.
            "http://127.0.0.1:11434/v1​",
        ],
    )
    def test_whitespace_or_a_control_character_anywhere_is_refused(self, raw: str) -> None:
        """MUTATION: drop the guard (all rows construct); drop the `" " in value` half (the
        three space rows construct); drop the `isprintable()` half (the other five construct);
        replace the guard with `value != value.strip()` (the interior space and U+200B rows
        construct)."""
        with pytest.raises(ValidationError) as exc_info:
            _llm(base_url=raw)
        assert _WHITESPACE_ERROR in str(exc_info.value)

    def test_userinfo_is_refused_and_the_value_is_not_echoed(self) -> None:
        """MUTATION: drop the `@` check (constructs); drop `hide_input_in_errors` from
        LLMConfig.model_config (`input_value='http://user:pw@host/v1'` appears in the text)."""
        with pytest.raises(ValidationError) as exc_info:
            _llm(base_url=_WITH_CREDENTIAL)
        text = str(exc_info.value)
        assert _USERINFO_ERROR in text
        assert "pw" not in text
        assert _WITH_CREDENTIAL not in text
        assert "input_value" not in text

    @pytest.mark.parametrize(
        "raw",
        [
            "ftp://user:pw@host/v1",
            # U+2100 (℀) NFKC-normalises to `a/c`, so urlsplit itself raises, and its message
            # (`netloc 'user:pw@host℀' contains invalid characters ...`) echoes the credential.
            "http://user:pw@host℀/v1",
        ],
    )
    def test_a_credential_in_a_malformed_url_is_not_echoed_either(self, raw: str) -> None:
        """The shape error is raised before the `@` check; it must hide the value too.

        MUTATION: move `urlsplit` out of the `try` (the U+2100 row then surfaces urlsplit's own
        message, which contains `pw`)."""
        with pytest.raises(ValidationError) as exc_info:
            _llm(base_url=raw)
        assert _SHAPE_ERROR in str(exc_info.value)
        assert "pw" not in str(exc_info.value)


class TestEnvironmentRoute:
    """`IndexConfig(repo_path=...)` is what every command builds; the name is the prefix route."""

    def test_the_env_name_reaches_the_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_LLM_BASE_URL", _LOCAL_URL)
        assert IndexConfig(repo_path=str(tmp_path)).llm.base_url == _LOCAL_URL

    def test_a_blank_env_value_is_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_LLM_BASE_URL", "")
        assert IndexConfig(repo_path=str(tmp_path)).llm.base_url is None

    def test_a_malformed_env_value_is_a_configuration_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_LLM_BASE_URL", "localhost:11434")
        with pytest.raises(ValidationError) as exc_info:
            IndexConfig(repo_path=str(tmp_path))
        assert _SHAPE_ERROR in str(exc_info.value)

    def test_a_credential_in_the_env_value_is_not_echoed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_LLM_BASE_URL", _WITH_CREDENTIAL)
        with pytest.raises(ValidationError) as exc_info:
            IndexConfig(repo_path=str(tmp_path))
        assert _USERINFO_ERROR in str(exc_info.value)
        assert "pw" not in str(exc_info.value)


class TestLocalContextTokens:
    """The context length of the server behind the URL; the retriever's budget reads it."""

    @pytest.mark.parametrize(
        ("raw", "stored"),
        [
            ("", None),
            ("   ", None),
            (1024, 1024),
            ("1024", 1024),
            (32768, 32768),
            (2_000_000, 2_000_000),
        ],
        ids=["blank", "spaces", "floor-int", "floor-str", "typical", "ceiling"],
    )
    def test_blank_is_unset_and_a_value_in_range_is_stored(
        self, raw: str | int, stored: int | None
    ) -> None:
        """MUTATION: drop the mode="before" blank validator (the blank rows fail to parse as an
        int); `ge=1024` -> `gt=1024` (the 1024 rows); `le=2_000_000` -> `lt` (the last row)."""
        assert _llm(base_url=_LOCAL_URL, local_context_tokens=raw).local_context_tokens == stored

    def test_the_default_is_unset(self) -> None:
        assert LLMConfig.model_fields["local_context_tokens"].default is None

    def test_a_blank_value_without_the_url_is_simply_unset(self) -> None:
        """The unit suite pins both variables to "" (tests/_env_isolation.py); blank must read
        as unset BEFORE the needs-the-URL check, or every test would fail at config time."""
        assert _llm(local_context_tokens="").local_context_tokens is None

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            (1023, "Input should be greater than or equal to 1024"),
            (0, "Input should be greater than or equal to 1024"),
            (2_000_001, "Input should be less than or equal to 2000000"),
        ],
    )
    def test_out_of_range_is_refused(self, raw: int, message: str) -> None:
        """MUTATION: floor 1024 -> 1 (1023 constructs); ceiling 2_000_000 -> 3_000_000
        (2_000_001 constructs)."""
        with pytest.raises(ValidationError) as exc_info:
            _llm(base_url=_LOCAL_URL, local_context_tokens=raw)
        assert message in str(exc_info.value)

    def test_without_the_url_is_refused_and_the_value_is_not_echoed(self) -> None:
        """MUTATION: drop the model validator (constructs); drop `hide_input_in_errors` from
        LLMConfig.model_config (`input_value={...'local_context_tokens': 32768...}` appears)."""
        with pytest.raises(ValidationError) as exc_info:
            _llm(local_context_tokens=32768)
        text = str(exc_info.value)
        assert _LOCAL_CONTEXT_TOKENS_ERROR in text
        assert "32768" not in text
        assert "input_value" not in text


class TestLocalContextTokensEnvironmentRoute:
    """`IndexConfig(repo_path=...)` reads `TRELIX_LLM_LOCAL_CONTEXT_TOKENS` by the prefix route."""

    def test_the_env_name_reaches_the_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_LLM_BASE_URL", _LOCAL_URL)
        monkeypatch.setenv("TRELIX_LLM_LOCAL_CONTEXT_TOKENS", "32768")
        assert IndexConfig(repo_path=str(tmp_path)).llm.local_context_tokens == 32768

    def test_an_out_of_range_env_value_is_a_configuration_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_LLM_BASE_URL", _LOCAL_URL)
        monkeypatch.setenv("TRELIX_LLM_LOCAL_CONTEXT_TOKENS", "1023")
        with pytest.raises(ValidationError) as exc_info:
            IndexConfig(repo_path=str(tmp_path))
        assert "Input should be greater than or equal to 1024" in str(exc_info.value)

    def test_the_env_value_without_the_url_is_a_configuration_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`TRELIX_LLM_BASE_URL` is pinned to "" (unset) by the suite's env isolation."""
        monkeypatch.setenv("TRELIX_LLM_LOCAL_CONTEXT_TOKENS", "32768")
        with pytest.raises(ValidationError) as exc_info:
            IndexConfig(repo_path=str(tmp_path))
        assert _LOCAL_CONTEXT_TOKENS_ERROR in str(exc_info.value)
        assert "32768" not in str(exc_info.value)


class TestReviewReportsTheConfigurationError:
    def test_review_prints_configuration_error_and_exits_1(self, tmp_path: Path) -> None:
        """MUTATION: drop the `_PydanticValidationError` handler around `IndexConfig(...)` in
        `review` (the exception then escapes Typer: no "Configuration error" line).

        The diff file is never read: config fails first.
        """
        diff = tmp_path / "changes.diff"
        diff.write_text("", encoding="utf-8")

        result = CliRunner().invoke(
            app,
            ["review", str(tmp_path), "--diff", str(diff), "--json"],
            env={"TRELIX_LLM_BASE_URL": "http://user:pw@127.0.0.1:1/v1"},
        )

        assert result.exit_code == 1, result.stderr
        stderr = " ".join(result.stderr.split())
        assert "Configuration error" in stderr
        assert _USERINFO_ERROR in stderr
        assert "pw@" not in result.stderr
        assert "input_value" not in result.stderr
        assert result.stdout == ""

    def test_review_reports_a_trailing_newline_as_a_configuration_error(
        self, tmp_path: Path
    ) -> None:
        """MUTATION: drop the whitespace guard in `_base_url_is_a_plain_http_url`.

        The value then reaches `OpenAI(base_url=...)`, httpx raises InvalidURL, and
        `DiffReviewer._get_client` swallows it: the command exits 3 with "the LLM is not
        configured (set OPENAI_API_KEY ...)" and prints `[]`, the misleading message this
        feature set out to remove. A secret store or `echo >> .env` routinely leaves the
        newline. The diff is real so that the mutant takes exactly that path.
        """
        diff = tmp_path / "changes.diff"
        diff.write_text(_ONE_HUNK_DIFF, encoding="utf-8")

        result = CliRunner().invoke(
            app,
            ["review", str(tmp_path), "--diff", str(diff), "--json"],
            env={"TRELIX_LLM_BASE_URL": "http://127.0.0.1:1/v1\n"},
        )

        assert result.exit_code == 1, result.stderr
        stderr = " ".join(result.stderr.split())
        assert "Configuration error" in stderr
        assert _WHITESPACE_ERROR in stderr
        assert "not configured" not in stderr
        assert result.stdout == ""

    def test_review_prints_error_for_a_plain_valueerror_and_exits_1(self, tmp_path: Path) -> None:
        """MUTATION: drop the `except (ValueError, FileNotFoundError)` clause in `review` (the
        plain ValueError then escapes Typer: no "Error:" line, a traceback in a real terminal).

        `RetrievalConfig` raises a plain ValueError, not a ValidationError, for a bad weight
        (tests/unit/test_config_error_redaction.py pins that), so the ValidationError handler
        alone does not catch it. The diff file is never read: config fails first.
        """
        diff = tmp_path / "changes.diff"
        diff.write_text("", encoding="utf-8")

        result = CliRunner().invoke(
            app,
            ["review", str(tmp_path), "--diff", str(diff), "--json"],
            env={"TRELIX_RETRIEVAL_LEG_WEIGHT_VECTOR": "abc"},
        )

        assert result.exit_code == 1, result.stderr
        stderr = " ".join(result.stderr.split())
        assert (
            "Error: TRELIX_RETRIEVAL_LEG_WEIGHT_VECTOR='abc' is not a valid retrieval weight"
            in stderr
        )
        assert "Configuration error" not in stderr
        assert "Traceback" not in stderr
        assert result.stdout == ""
