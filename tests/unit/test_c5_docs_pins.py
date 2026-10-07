"""The user-facing files that name `TRELIX_RETRIEVAL_CITATIONS` say what the code does, the
ones that describe `trelix ask` and `GET /ask` describe both abstentions, and docs/WHY_TRELIX.md
counts the FLARE phrases the list has.

Pattern: tests/unit/test_llm_thinking_mode.py's doc pins. Each assertion is a literal from the
file it reads; the code default itself is pinned in tests/unit/test_citation_tags.py and the
abstention behaviour in tests/unit/test_ask_abstention.py.
"""

from __future__ import annotations

from pathlib import Path

from tests._env_isolation import BEAST_MODE_DEFAULTS
from trelix.retrieval.flare import _DEFAULT_UNCERTAINTY_PHRASES

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


def _squash(text: str) -> str:
    """Undo markdown line wrapping so a sentence can be matched as one."""
    return " ".join(text.split())


def test_cli_reference_documents_both_abstentions_under_ask() -> None:
    text = (_REPO_ROOT / "docs/CLI_REFERENCE.md").read_text(encoding="utf-8")
    ask_section = _squash(text[text.index("### `trelix ask`") : text.index("### `trelix query`")])

    assert (
        "the answer is the one-line notice `[trelix] No relevant code found — cannot synthesize "
        "an answer.` on stdout, the command exits `0` and the LLM is not called" in ask_section
    )
    assert (
        "prints the notice once per retrieval round (twice at the default "
        "`TRELIX_RETRIEVAL_FLARE_MAX_RETRIES=1`)" in ask_section
    )
    assert "exactly one line starting `INSUFFICIENT_EVIDENCE:`" in ask_section
    assert "stderr stays empty and the command exits `0`" in ask_section


def test_backwards_compatibility_has_the_empty_retrieval_section() -> None:
    text = (_REPO_ROOT / "docs/BACKWARDS_COMPATIBILITY.md").read_text(encoding="utf-8")

    assert (
        "### Behaviour change shipped as a fix: `Synthesizer.stream()` on an empty retrieval"
        in text
    )
    assert "| REST `GET /ask` |" in text
    assert (
        "| unchanged: `stream()` checks the key before the retrieval result, as `synthesize()` "
        "always did |" in text
    )


def test_architecture_ask_row_names_the_notice_and_the_missing_llm_call() -> None:
    text = (_REPO_ROOT / "docs/architecture.md").read_text(encoding="utf-8")

    assert (
        "| GET | `/ask` | `StreamingResponse` | SSE token stream; an empty retrieval streams the "
        "`[trelix] No relevant code found — cannot synthesize an answer.` notice then `[DONE]` "
        "with no LLM call;" in text
    )


def test_eval_readme_says_graphrag_answers_cannot_abstain() -> None:
    text = _squash((_REPO_ROOT / "eval/README.md").read_text(encoding="utf-8"))

    assert (
        "get the `[C#]` cite lines but not the abstention sentence, so they cannot abstain by "
        "protocol" in text
    )


def test_why_trelix_counts_the_flare_phrases_the_list_has() -> None:
    """Four named and "eight more": `insufficient_evidence:` made the list twelve, and the
    sentence that counts it must move with the list."""
    text = (_REPO_ROOT / "docs/WHY_TRELIX.md").read_text(encoding="utf-8")

    assert (
        '(`"i don\'t know"`, `"cannot find"`, `"no relevant code"`, '
        '`"insufficient context"`, and eight more in `_DEFAULT_UNCERTAINTY_PHRASES`)' in text
    )
    assert len(_DEFAULT_UNCERTAINTY_PHRASES) == 12
