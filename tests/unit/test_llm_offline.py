"""`trelix.llm.offline`: the model-size parser and the size-floor warning (R-C4-03).

Pure functions over the model tag. Every expected value is a literal, including the two
warning texts, so a wording change has to be made here on purpose.
"""

from __future__ import annotations

import pytest

from trelix.llm.offline import model_size_billions, small_model_warning

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
