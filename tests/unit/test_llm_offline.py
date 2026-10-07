"""`trelix.llm.offline`: the model-size parser and the size-floor warning (R-C4-03), the
prompt-token estimate and the prompt-truncation rule (R-C4-02).

Pure functions over the model tag and over message lists. Every expected value is a literal,
including the warning texts, so a wording change has to be made here on purpose.
"""

from __future__ import annotations

import logging

import pytest
import tiktoken

from trelix.llm import offline
from trelix.llm.offline import (
    estimate_prompt_tokens,
    model_size_billions,
    prompt_truncation_signal,
    small_model_warning,
)

# Loaded at collection, the shape of test_chunker_token_budget_boundary.py: the process cache is
# warm before pytest-socket's per-test ban, so `estimate_prompt_tokens` never needs the network.
_ENC = tiktoken.get_encoding("cl100k_base")

# (tag, billions). A size token is a number followed by b/B or m/M, not glued to a letter,
# digit or dot on either side; the largest wins; `m` is divided by 1000.
_SIZE_TABLE = [
    ("qwen2.5-coder:7b", 7.0),
    ("qwen3-coder:30b-a3b", 30.0),  # largest, not first: the active-parameter `a3b` is 3
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", 30.0),
    ("gpt-oss:20b", 20.0),
    ("gpt-oss:20B", 20.0),
    ("gpt-oss:120b", 120.0),
    ("devstral-small-2:24b", 24.0),
    ("llama3.1:8b-instruct-q4_K_M", 8.0),
    ("qwen2.5:0.5b", 0.5),
    ("qwen2.5:7b-instruct-1m", 7.0),  # the 1m context tag is 0.001 and loses
    ("1m-7b", 7.0),  # ... whatever the order: the largest token wins, not the first
    ("gemma3:270m", 0.27),
    ("SmolLM2-360M-Instruct", 0.36),  # upper-case M is millions too
    ("deepseek-r1:671b", 671.0),
    ("x:19.9b", 19.9),
    ("gpt-4o", None),
    ("gpt-4o-mini", None),
    ("glm-4.5-air", None),
    ("q4_K_M", None),
    ("", None),
    # An NxMb mixture tag is unparseable, not N x M: the `7b` is glued to the `x`.
    ("mixtral-8x7b", None),
    ("mixtral:8x22b", None),
    ("nomic-embed-text:v1.5", None),
    # Glued on the right (`7bq4`) or preceded by a dot (`x.5b`): not a size token either.
    ("x:7bq4", None),
    ("x.5b", None),
]


@pytest.mark.parametrize(("model", "billions"), _SIZE_TABLE)
def test_model_size_billions(model: str, billions: float | None) -> None:
    """MUTATION: take the first token instead of the largest (`30b-a3b` -> 3.0); drop the
    `m` unit (`gemma3:270m` -> None); make it case-sensitive (`unit == "m"`:
    `SmolLM2-360M-Instruct` -> 360.0); drop the look-behind (`mixtral-8x7b` -> 7.0); drop the
    look-ahead (`x:7bq4` -> 7.0); drop the `.` from the look-behind (`x.5b` -> 5.0)."""
    assert model_size_billions(model) == billions


_W1_QWEN_7B = (
    "Local model 'qwen2.5-coder:7b' is about 7B parameters, under the 20B floor "
    "docs/OFFLINE.md assumes for trelix review; expect more unreviewed hunks and weaker findings."
)
_W1_19_9B = (
    "Local model 'x:19.9b' is about 19.9B parameters, under the 20B floor "
    "docs/OFFLINE.md assumes for trelix review; expect more unreviewed hunks and weaker findings."
)
_W1_GEMMA_270M = (
    "Local model 'gemma3:270m' is about 0.27B parameters, under the 20B floor "
    "docs/OFFLINE.md assumes for trelix review; expect more unreviewed hunks and weaker findings."
)
_W2_GLM = (
    "Local model 'glm-4.5-air': no parameter count in the tag, so trelix cannot tell "
    "whether it meets the 20B floor docs/OFFLINE.md assumes."
)
_W2_MIXTRAL = (
    "Local model 'mixtral-8x7b': no parameter count in the tag, so trelix cannot tell "
    "whether it meets the 20B floor docs/OFFLINE.md assumes."
)


class TestSmallModelWarning:
    def test_under_the_floor_is_w1_with_the_pinned_text(self) -> None:
        """MUTATION: floor 20 -> 7 (no warning for 7b, and the text says 7B floor)."""
        assert small_model_warning("qwen2.5-coder:7b") == _W1_QWEN_7B

    def test_just_under_the_floor_is_w1(self) -> None:
        assert small_model_warning("x:19.9b") == _W1_19_9B

    def test_millions_are_reported_in_billions(self) -> None:
        assert small_model_warning("gemma3:270m") == _W1_GEMMA_270M

    def test_at_the_floor_is_silent(self) -> None:
        """MUTATION: `<` -> `<=` (gpt-oss:20b then warns)."""
        assert small_model_warning("gpt-oss:20b") is None

    def test_over_the_floor_is_silent(self) -> None:
        assert small_model_warning("deepseek-r1:671b") is None
        assert small_model_warning("qwen3-coder:30b-a3b") is None

    def test_no_size_in_the_tag_is_w2_with_the_pinned_text(self) -> None:
        assert small_model_warning("glm-4.5-air") == _W2_GLM

    def test_a_mixture_tag_is_w2_not_a_product(self) -> None:
        assert small_model_warning("mixtral-8x7b") == _W2_MIXTRAL


# (estimated, reported, signal). Truncated means `reported < 0.85 * estimated`; anything that is
# not a positive int on either side is unknown, never truncated.
_TRUNCATION_TABLE = [
    (1000, 849, "prompt_truncated"),
    (1000, 850, None),  # exactly at the floor is not truncated
    (1000, 700, "prompt_truncated"),
    (1000, 1, "prompt_truncated"),
    (1000, 1200, None),
    (20, 17, None),  # 17 < 17.0 is false
    (None, 5, None),
    (1000, None, None),
    (1000, 0, None),
    (0, 0, None),
    (0, 5, None),
    (1000, True, None),  # a bool is not a count
]


@pytest.mark.parametrize(("estimated", "reported", "signal"), _TRUNCATION_TABLE)
def test_prompt_truncation_signal(estimated: object, reported: object, signal: str | None) -> None:
    """MUTATION: floor 0.85 -> 0.5 (`(1000, 700)` and `(1000, 849)` stop signalling); `<` -> `<=`
    (`(1000, 850)` signals); `prompt_tokens == 0` read as truncated (`(1000, 0)` signals);
    `isinstance(..., int)` without the bool exclusion (`(1000, True)` signals)."""
    assert prompt_truncation_signal(estimated, reported) == signal  # type: ignore[arg-type]


_TWO_MESSAGES = [
    {"role": "system", "content": "hello world"},
    {"role": "user", "content": "hi"},
]
_W5_OSERROR = (
    "tiktoken cl100k_base is not available (OSError); the prompt-truncation check is off for this "
    "run (docs/OFFLINE.md: prefetch)"
)


class TestEstimatePromptTokens:
    def test_counts_the_content_of_every_message(self) -> None:
        """MUTATION: count the last message only (1); count the system message out (1)."""
        assert estimate_prompt_tokens(_TWO_MESSAGES) == 3

    def test_no_messages_is_zero(self) -> None:
        assert estimate_prompt_tokens([]) == 0

    def test_a_special_token_in_the_text_is_counted_not_refused(self) -> None:
        """MUTATION: drop `disallowed_special=()` (tiktoken raises ValueError on the text)."""
        assert estimate_prompt_tokens([{"role": "user", "content": "<|endoftext|>"}]) == 7


class TestEncoderUnavailable:
    """The encoder cannot load (no cache file and no network): the estimate is None, W5 once."""

    @pytest.fixture
    def cold_encoder(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []

        def refuse(name: str) -> object:
            calls.append(name)
            raise OSError("canary: no cache file and no network")

        monkeypatch.setattr(offline, "_encoder", None)
        monkeypatch.setattr(offline, "_ENCODER_UNAVAILABLE", False)
        monkeypatch.setattr(tiktoken, "get_encoding", refuse)
        return calls

    def test_two_estimates_load_once_and_warn_once(
        self, cold_encoder: list[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUTATION: drop the `_ENCODER_UNAVAILABLE` flag (two `get_encoding` calls and two W5
        records); return 0 instead of None (the rule then reads 0 as a count)."""
        with caplog.at_level(logging.WARNING, logger="trelix.llm.offline"):
            first = estimate_prompt_tokens(_TWO_MESSAGES)
            second = estimate_prompt_tokens(_TWO_MESSAGES)

        assert (first, second) == (None, None)
        assert cold_encoder == ["cl100k_base"]
        warnings = [r.getMessage() for r in caplog.records if r.name == "trelix.llm.offline"]
        assert warnings == [_W5_OSERROR]

    @pytest.mark.parametrize("error", [RuntimeError, ValueError])
    def test_other_load_errors_are_none_and_name_the_class(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        error: type[Exception],
    ) -> None:
        def refuse(_name: str) -> object:
            raise error("canary")

        monkeypatch.setattr(offline, "_encoder", None)
        monkeypatch.setattr(offline, "_ENCODER_UNAVAILABLE", False)
        monkeypatch.setattr(tiktoken, "get_encoding", refuse)

        with caplog.at_level(logging.WARNING, logger="trelix.llm.offline"):
            assert estimate_prompt_tokens(_TWO_MESSAGES) is None

        assert f"is not available ({error.__name__});" in caplog.text
