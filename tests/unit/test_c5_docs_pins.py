"""The user-facing files that name `TRELIX_RETRIEVAL_CITATIONS` say what the code does, the
ones that describe `trelix ask` and `GET /ask` describe both abstentions, and docs/WHY_TRELIX.md
counts the FLARE phrases the list has.

Pattern: tests/unit/test_llm_thinking_mode.py's doc pins. Each assertion is a literal from the
file it reads; the code default itself is pinned in tests/unit/test_citation_tags.py and the
abstention behaviour in tests/unit/test_ask_abstention.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests._env_isolation import BEAST_MODE_DEFAULTS
from trelix.eval.synthesis import validate_synthesis_entry
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


def test_cli_reference_documents_ask_json_and_the_sources_footer() -> None:
    text = (_REPO_ROOT / "docs/CLI_REFERENCE.md").read_text(encoding="utf-8")
    ask_section = _squash(text[text.index("### `trelix ask`") : text.index("### `trelix query`")])

    assert (
        "trelix ask <repo_path> <question> [--provider PROVIDER] [--agentic] [--session ID] "
        "[--json]" in ask_section
    )
    assert (
        "with exactly the keys `marker`, `status`, `path`, `lines`, `symbol`, `detail`"
        in ask_section
    )
    assert "`--json cannot be combined with --agentic or --session.`, exit `2`" in ask_section
    assert (
        "`Error: --json needs LLM synthesis; with the local embedder and FLARE off, trelix ask "
        "prints the retrieved context only. Use trelix search --json for machine-readable "
        "retrieval, or a non-local --provider.` to stderr and exits `1`" in ask_section
    )
    assert "`Error: --json is not available in agentic mode" in ask_section
    assert (
        "[C2] src/auth/middleware.py:70-80 AuthMiddleware.bearer` for a marker whose chunk still "
        "fits the file on disk (`valid`)" in ask_section
    )
    assert "`Sources: none cited.` for an answer without markers" in ask_section
    assert "The answer text itself is never rewritten: it has already streamed." in ask_section
    # The `unknown` detail is `no retrieved chunk has this tag`; only the two stale statuses end
    # `re-index` (src/trelix/retrieval/citations.py, pinned by test_cli_ask_footer.py).
    assert (
        '`detail` is `""` when valid, one line ending `re-index` for `file_missing` and '
        "`line_out_of_range` (the index is behind the tree) and `no retrieved chunk has this "
        "tag` for `unknown`" in ask_section
    )
    assert (
        "(`file_missing` and `line_out_of_range`, whose detail ends `re-index`; `unknown`, whose "
        "detail is `no retrieved chunk has this tag`)" in ask_section
    )
    # `_ask_emit` verifies nothing after an abstention (test_cli_ask_json.py pins the `[]`).
    assert (
        "The list is `[]` when the answer has no marker, when it abstained, and whenever "
        "`TRELIX_RETRIEVAL_CITATIONS` is off" in ask_section
    )


def test_cli_reference_exit_code_table_names_ask_json_with_agentic_as_a_usage_error() -> None:
    text = (_REPO_ROOT / "docs/CLI_REFERENCE.md").read_text(encoding="utf-8")

    assert (
        "a missing argument, `trelix ask --json` with `--agentic` or `--session`) also exits `2`"
        in text
    )


def test_user_guide_ask_output_names_the_footer_and_json() -> None:
    text = _squash((_REPO_ROOT / "docs/USER_GUIDE.md").read_text(encoding="utf-8"))

    assert (
        "**Output:** a streamed natural-language answer. With `TRELIX_RETRIEVAL_CITATIONS=true` "
        "the answer cites `[C#]` tags and ends with a `Sources:` footer that verifies each tag "
        "against the retrieved chunk and the file on disk. `--json` prints one JSON object "
        "(`query`, `answer`, `abstained`, `abstain_reason`, `citations`) instead of streaming."
        in text
    )


def test_security_ask_row_says_the_footer_and_json_print_index_data_only() -> None:
    text = (_REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")

    assert (
        "the `Sources:` footer and `trelix ask --json` print the index's path, lines and symbol "
        "for each marker, never text the model wrote |" in text
    )


def test_backwards_compatibility_lists_ask_json_and_the_footer_as_additive() -> None:
    text = _squash((_REPO_ROOT / "docs/BACKWARDS_COMPATIBILITY.md").read_text(encoding="utf-8"))

    assert (
        "Additive: `trelix ask --json` prints one JSON object (`query`, `answer`, `abstained`, "
        "`abstain_reason`, `citations`) and nothing else on stdout" in text
    )
    assert "`Synthesizer(stream_to_stdout=False)` keeps every stdout write in" in text


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


def test_cli_reference_eval_synthesis_names_the_v2_fields_the_sample_and_the_exit_code() -> None:
    text = (_REPO_ROOT / "docs/CLI_REFERENCE.md").read_text(encoding="utf-8")
    start, end = text.index("### `trelix eval-synthesis`"), text.index("### `trelix eval-validate`")
    section = _squash(text[start:end])

    for needle in ("--per-query-out", "`answerable`", "`gold_answer`", "`expected_citations`"):
        assert needle in section
    assert "trelix eval-synthesis . --golden eval/golden_synthesis_sample.jsonl" in section
    assert "missing golden file prints `Golden file not found: <path>` and exits `1`" in section
    assert "tests/golden_synthesis_queries.jsonl" not in section
    assert "prints a table of all-zero scores" not in section


def test_eval_readme_tabulates_the_three_synthesis_fields() -> None:
    text = (_REPO_ROOT / "eval/README.md").read_text(encoding="utf-8")
    section = text[
        text.index("## `golden_synthesis_sample.jsonl`") : text.index("## Related tooling")
    ]

    for name in ("answerable", "gold_answer", "expected_citations"):
        assert f"| `{name}` |" in section
    assert "nothing in the eval reads them yet" in _squash(section)


def test_the_sample_golden_has_four_valid_lines_the_last_unanswerable() -> None:
    raw = (_REPO_ROOT / "eval/golden_synthesis_sample.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(line) for line in raw.splitlines() if line.strip()]

    assert len(lines) == 4
    assert [validate_synthesis_entry(line) for line in lines] == [[], [], [], []]
    assert lines[3]["answerable"] is False
    assert [line["expected_citations"] for line in lines[:3]] == [
        line["relevant_files"] for line in lines[:3]
    ]


def test_user_guide_and_contributing_name_the_synthesis_v2_pieces() -> None:
    guide = (_REPO_ROOT / "docs/USER_GUIDE.md").read_text(encoding="utf-8")
    fields = guide[guide.index("**Fields:**") : guide.index("### Running the evaluation")]

    for name in ("answerable", "gold_answer", "expected_citations"):
        assert f"- `{name}` (optional" in fields
    output = guide[guide.index("### Running the evaluation") : guide.index("### Python API")]
    for row in ("Unanswerable queries", "Unscoreable queries"):
        assert row in output, f"the sample table lacks the {row!r} row the command prints"
    assert "| `synthesis_records.py` |" in (_REPO_ROOT / "CONTRIBUTING.md").read_text(
        encoding="utf-8"
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
