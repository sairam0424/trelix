"""The user-facing files that name `TRELIX_RETRIEVAL_CITATIONS` say what the code does.

Pattern: tests/unit/test_llm_thinking_mode.py's doc pins. Each assertion is a literal from the
file it reads; the code default itself is pinned in tests/unit/test_citation_tags.py.
"""

from __future__ import annotations

from pathlib import Path

from tests._env_isolation import BEAST_MODE_DEFAULTS

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_configuration_lists_the_flag_at_its_default_in_the_retrieval_table() -> None:
    text = (_REPO_ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")

    assert "| `TRELIX_RETRIEVAL_CITATIONS` | `false` |" in text
    assert (
        "`false` leaves the assembled context, the synthesis prompts and every command's "
        "output unchanged" in text
    )
    # The embedded .env example carries the same line as .env.example.
    assert "\nTRELIX_RETRIEVAL_CITATIONS=false\n" in text


def test_env_example_lists_the_flag_commented_out_at_its_default() -> None:
    text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "\n# TRELIX_RETRIEVAL_CITATIONS=false\n" in text


def test_the_unit_suite_pins_the_flag_off() -> None:
    """A developer's .env turning citations on must not change what the tests observe."""
    assert BEAST_MODE_DEFAULTS["TRELIX_RETRIEVAL_CITATIONS"] == "false"
