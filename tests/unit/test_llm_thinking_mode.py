"""trelix.llm.thinking: which thinking request shape a Claude model id takes.

Expected values are literals on purpose. The classifier is the thing under test, so the
table below is the specification: adding a model to it is a decision a person makes.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from trelix.llm.thinking import (
    ThinkingModeMemory,
    is_enabled_thinking_rejection,
    thinking_mode_for_model,
)

# The Bedrock Converse ValidationException text, confirmed live on 2026-10-04 for
# us.anthropic.claude-sonnet-5 and us.anthropic.claude-sonnet-5-5 when a budget thinking
# request was sent.
LIVE_ENABLED_THINKING_REJECTION = (
    '"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" '
    'and "output_config.effort" to control thinking behavior.'
)

# Wall-clock ceiling for classifying the seven 50,000 character ids together.
LONG_ID_TIME_LIMIT_SECONDS = 0.5

ADAPTIVE_ONLY_IDS = [
    # Direct API ids.
    "claude-sonnet-5",
    "claude-sonnet-5-5",
    "claude-opus-5",
    "claude-haiku-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    # Bedrock model ids and inference profiles, bare and prefixed.
    "anthropic.claude-sonnet-5",
    "anthropic.claude-sonnet-5-5",
    "us.anthropic.claude-sonnet-5",
    "us.anthropic.claude-sonnet-5-5",
    "global.anthropic.claude-sonnet-5",
    "global.anthropic.claude-sonnet-5-5",
    "eu.anthropic.claude-opus-4-8",
    "us-gov.anthropic.claude-sonnet-5",
    "anthropic.claude-opus-4-7",
    "global.anthropic.claude-opus-4-7",
    # Dated and versioned suffixes.
    "claude-sonnet-5-20261001",
    "claude-sonnet-5-5-20261001",
    "anthropic.claude-sonnet-5-v1:0",
    "us.anthropic.claude-sonnet-5-5-v1:0",
    "us.anthropic.claude-sonnet-5-5-20261001-v1:0",
    "us.anthropic.claude-opus-4-8-v1:0",
    # Numeric, not textual: 50 is a version greater than 5, not a prefix match on "5".
    "claude-sonnet-50",
]

BUDGET_IDS = [
    # Models that work today and must keep their request byte for byte.
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307",
    "us.anthropic.claude-sonnet-4-6",
    "global.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-opus-4-6-v1",
    # ARNs are not read by name (docs/CONFIGURATION.md says so): they start as budget and rely
    # on the runtime retry-and-remember path.
    "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-5-5",
    "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-5",
    "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123xyz",
    # Not a Claude id, and ids the table has never seen: budget, so nothing changes for them.
    "gpt-4o",
    "gemini-2.5-pro",
    "us.amazon.nova-pro-v1:0",
    "claude-nova-7",
    "us.anthropic.claude-nova-7",
    "some-future-model",
    "claude",
    "claude-sonnet",
    "claude-sonnet-",
]

HOSTILE_IDS = [
    # (id, expected mode, why it is budget)
    ("", "budget", "empty string"),
    (" ", "budget", "whitespace only"),
    (" claude-sonnet-5", "budget", "leading space"),
    ("claude-sonnet-5 ", "budget", "trailing space"),
    ("claude-sonnet-5\n", "budget", "trailing newline (a '$' anchor would accept it)"),
    ("\nclaude-sonnet-5", "budget", "leading newline"),
    ("claude-sonnet-5x", "budget", "junk after the version"),
    ("claude-sonnet-5-5x", "budget", "junk after the minor version"),
    ("xclaude-sonnet-5", "budget", "junk before the id"),
    ("my-claude-sonnet-5", "budget", "a longer name that merely contains the id"),
    ("foo.anthropic.bar.claude-sonnet-5", "budget", "prefix is not <region>.anthropic."),
    ("CLAUDE-SONNET-5", "budget", "upper case: provider ids are lower case"),
    ("Claude-Sonnet-5", "budget", "mixed case"),
    ("us.Anthropic.claude-sonnet-5", "budget", "mixed case prefix"),
    ("claude-sonnet-05", "budget", "leading zero in the major version"),
    ("claude-sonnet-5-05", "budget", "leading zero in the minor version"),
    ("claude-sonnet-5-", "budget", "dangling separator"),
    ("claude-sonnet-5-5-5", "budget", "a third version component"),
    ("claude-sonnet-500", "budget", "three-digit major is not a model version"),
    ("claude-sonnet-20250514", "budget", "a date where the version should be"),
    ("claude-sonnet-2025", "budget", "a year that looks like a version"),
    ("claude-opus-20250514", "budget", "a date that looks like an opus version"),
    ("claude-haiku-5-202610", "budget", "a six digit tail is not a date"),
    ("claude-sonnet-5--5", "budget", "double separator"),
    ("claude-sonnet-5:0", "budget", "colon without the -v1 part"),
    ("claude-sonnet-4-20250514", "budget", "a date in the minor slot is 4.0, not 4.20250514"),
    ("claude-sonnet-4-99", "budget", "a two digit minor on a pre-5 sonnet is still below 5.0"),
    # int() reads digits of other scripts as numbers, so \d without re.ASCII would turn
    # these into versions or dates. Written as escapes so the source shows what they are.
    ("claude-sonnet-5\u0660", "budget", "Arabic-Indic zero after the major (would read as 50)"),
    (
        "claude-sonnet-5-\u0662\u0660\u0662\u0665\u0660\u0665\u0661\u0664",
        "budget",
        "an eight digit date written in Arabic-Indic digits",
    ),
    ("claude-sonnet-5-v\u0661", "budget", "an Arabic-Indic digit as the -v revision"),
    ("claude-sonnet-5-v1234", "budget", "a four digit -v revision is not a provider revision"),
    ("claude-sonnet-5-v1:1234", "budget", "a four digit second revision part"),
    ("anthropic.claude-sonnet-5-v" + "1" * 50_000, "budget", "an endless -v revision"),
]


@pytest.mark.parametrize("model_id", ADAPTIVE_ONLY_IDS)
def test_adaptive_only_ids_are_classified_adaptive(model_id: str) -> None:
    assert thinking_mode_for_model(model_id) == "adaptive"


@pytest.mark.parametrize("model_id", BUDGET_IDS)
def test_everything_else_is_classified_budget(model_id: str) -> None:
    assert thinking_mode_for_model(model_id) == "budget"


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [(model_id, expected) for model_id, expected, _why in HOSTILE_IDS],
    ids=[why for _id, _expected, why in HOSTILE_IDS],
)
def test_hostile_ids(model_id: str, expected: str) -> None:
    assert thinking_mode_for_model(model_id) == expected


def test_very_long_ids_are_classified_quickly_and_as_budget() -> None:
    """50,000 characters of several shapes, none of which is a real id. The pattern is
    anchored and has no nested quantifier, so none of these can take more than linear
    time. All seven together take about 20 microseconds; a quadratic pattern (measured: a
    leading ``[a-z]*[a-z]*`` took 11 seconds on 8,000 characters and 652 seconds on
    50,000) takes far longer. The wall-clock bound sits between the two, so a loaded
    machine does not trip it and a slow pattern does not get to hide behind the global
    pytest timeout."""
    long_ids = [
        "a" * 50_000,
        "claude-sonnet-" + "5" * 50_000,
        "claude-sonnet-5" + "-1" * 25_000,
        "claude-sonnet-5-" + "9" * 50_000,
        "us." * 17_000,
        "a" * 50_000 + ".anthropic.claude-sonnet-5",
        "claude-sonnet-5" + "x" * 50_000,
    ]

    started = time.perf_counter()
    modes = [thinking_mode_for_model(model_id) for model_id in long_ids]
    elapsed = time.perf_counter() - started

    assert modes == ["budget"] * len(long_ids)
    assert elapsed < LONG_ID_TIME_LIMIT_SECONDS


def test_opus_turns_adaptive_only_at_four_seven_not_before() -> None:
    assert thinking_mode_for_model("claude-opus-4-5") == "budget"
    assert thinking_mode_for_model("claude-opus-4-6") == "budget"
    assert thinking_mode_for_model("claude-opus-4-7") == "adaptive"
    assert thinking_mode_for_model("claude-opus-4-8") == "adaptive"
    assert thinking_mode_for_model("claude-opus-4-10") == "adaptive"


def test_the_4_7_threshold_is_opus_only() -> None:
    """Sonnet and haiku have no documented adaptive-only release below 5."""
    assert thinking_mode_for_model("claude-sonnet-4-7") == "budget"
    assert thinking_mode_for_model("claude-sonnet-4-8") == "budget"
    assert thinking_mode_for_model("claude-haiku-4-7") == "budget"


def test_version_is_read_as_numbers_so_a_date_cannot_pose_as_a_version() -> None:
    """4.5 with a date is below 5.0 even though 20250929 is numerically huge, and the
    bare major 5 followed by a date is 5.0, not 5.<date>."""
    assert thinking_mode_for_model("claude-sonnet-4-5-20250929") == "budget"
    assert thinking_mode_for_model("claude-sonnet-5-20261001") == "adaptive"


@pytest.mark.parametrize(
    "message",
    [
        LIVE_ENABLED_THINKING_REJECTION,
        "ValidationException: " + LIVE_ENABLED_THINKING_REJECTION,
        "Error code: 400 - " + LIVE_ENABLED_THINKING_REJECTION,
        "x" * 50_000 + LIVE_ENABLED_THINKING_REJECTION,
    ],
)
def test_the_live_rejection_text_is_recognised(message: str) -> None:
    assert is_enabled_thinking_rejection(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "",
        "thinking.type.enabled",
        "is not supported",
        '"thinking.type.adaptive" is not supported for this model.',
        "`temperature` is deprecated for this model.",
        "The provided model identifier is invalid.",
        "Invocation of model ID with on-demand throughput isn't supported.",
        "x" * 50_000,
    ],
)
def test_other_errors_are_not_mistaken_for_the_thinking_rejection(message: str) -> None:
    assert is_enabled_thinking_rejection(message) is False


class _SlowMembershipFrozenSet(frozenset[str]):
    def __contains__(self, item: object) -> bool:
        time.sleep(0.05)
        return super().__contains__(item)


class TestThinkingModeMemory:
    def test_falls_back_to_the_classifier_until_something_is_learned(self) -> None:
        memory = ThinkingModeMemory()
        assert memory.mode_for("claude-sonnet-5") == "adaptive"
        assert memory.mode_for("claude-sonnet-4-6") == "budget"
        assert memory.mode_for("some-future-model") == "budget"

    def test_a_learned_model_becomes_adaptive_and_only_that_model(self) -> None:
        memory = ThinkingModeMemory()
        assert memory.remember_adaptive("some-future-model") is True
        assert memory.mode_for("some-future-model") == "adaptive"
        assert memory.mode_for("claude-sonnet-4-6") == "budget"

    def test_remembering_twice_reports_the_second_as_already_known(self) -> None:
        memory = ThinkingModeMemory()
        assert memory.remember_adaptive("some-future-model") is True
        assert memory.remember_adaptive("some-future-model") is False
        assert memory.mode_for("some-future-model") == "adaptive"

    def test_instances_do_not_share_what_they_learned(self) -> None:
        first = ThinkingModeMemory()
        second = ThinkingModeMemory()
        first.remember_adaptive("some-future-model")
        assert second.mode_for("some-future-model") == "budget"

    def test_concurrent_rejections_report_exactly_one_first(self) -> None:
        """The membership check sleeps, which widens the check-then-write window far enough
        that, without the lock, all sixteen threads see "not learned yet" and all report
        being first. The GIL alone would hide that race on a plain frozenset."""
        memory = ThinkingModeMemory()
        memory._adaptive_models = _SlowMembershipFrozenSet()
        barrier = threading.Barrier(16)
        firsts: list[bool] = []

        def learn() -> None:
            barrier.wait()
            firsts.append(memory.remember_adaptive("some-future-model"))

        threads = [threading.Thread(target=learn) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert firsts.count(True) == 1
        assert firsts.count(False) == 15
        assert memory.mode_for("some-future-model") == "adaptive"


_REPO_ROOT = Path(__file__).resolve().parents[2]

# The user-facing files that name the adaptive-only Opus boundary. The classifier treats every
# Opus from 4.7 up as adaptive-only (test_opus_turns_adaptive_only_at_four_seven_not_before
# pins claude-opus-4-10), so they must say "4.7 and newer" rather than list 4.7 and 4.8.
_FILES_NAMING_THE_OPUS_BOUNDARY = [
    ".env.example",
    "docs/CONFIGURATION.md",
    "docs/PROVIDERS.md",
    "docs/USER_GUIDE.md",
]


@pytest.mark.parametrize("relative_path", _FILES_NAMING_THE_OPUS_BOUNDARY)
def test_user_facing_text_states_the_opus_boundary_as_4_7_and_newer(relative_path: str) -> None:
    text = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert "Opus 4.7 and newer" in text
    assert "Opus 4.7/4.8" not in text
    assert "Opus 4.7 and Opus 4.8" not in text
    assert "`claude-opus-4-7` / `claude-opus-4-8`" not in text


def test_the_docs_do_not_promise_reasoning_text_from_adaptive_models() -> None:
    """Adaptive-only models decide whether to think and trelix sends no ``display``
    setting, so the provider table may only promise the reasoning text a response carries."""
    providers = (_REPO_ROOT / "docs/PROVIDERS.md").read_text(encoding="utf-8")
    configuration = (_REPO_ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")

    unconditional = "returns the reasoning text on `ChatResponse.thinking`"
    conditional = "any reasoning text the response carries on `ChatResponse.thinking`"
    assert unconditional not in providers
    assert providers.count(conditional) == 2
    assert "what the default display returns for a harder question on Claude 5 was not" in providers
    assert "without a `display` setting" in configuration
    assert "do not rely on `ChatResponse.thinking` being populated" in configuration
