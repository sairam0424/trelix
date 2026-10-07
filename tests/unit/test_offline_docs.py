"""Doc pins for the local-server feature (PRs 1, 2, 3 and 5 of roadmap item C-7).

Every user-facing file that lists the LLM variables names `TRELIX_LLM_BASE_URL`, the two
that explain it name the bearer and the token-limit field, and every file that promised zero
LLM calls "with no chat credential" now names the variable as the second condition (a keyless
local-server client is a usable client; the planner calls it). The files that document the
review statuses name the `prompt_truncated` detail. The same five variable lists name
`TRELIX_LLM_LOCAL_CONTEXT_TOKENS`, and PROVIDERS.md says what to set it to (the server's own
context length, Ollama's `OLLAMA_CONTEXT_LENGTH`). PR 5 pins the guide itself: every needle
`docs/OFFLINE.md` must carry, the index row and the cross-reference sentences that link it, the
absence of a version stamp (so no release has to touch it), and that the three warnings in
`trelix.llm.offline` which name the guide point at a file and a heading that exist. Literal
(file, needle) pairs, the shape of test_llm_thinking_mode.py's Opus-boundary pins. MUTATION:
delete any row here and the matching doc line; the row for that file fails.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PINS = [
    (".env.example", "TRELIX_LLM_BASE_URL"),
    ("docs/CONFIGURATION.md", "TRELIX_LLM_BASE_URL"),
    ("docs/PROVIDERS.md", "TRELIX_LLM_BASE_URL"),
    ("docs/CLI_REFERENCE.md", "TRELIX_LLM_BASE_URL"),
    ("README.md", "TRELIX_LLM_BASE_URL"),
    ("CHANGELOG.md", "TRELIX_LLM_BASE_URL"),
    # The files that promise zero planner calls without a chat credential.
    ("docs/FAQ.md", "TRELIX_LLM_BASE_URL"),
    ("docs/GETTING_STARTED.md", "TRELIX_LLM_BASE_URL"),
    ("docs/USER_GUIDE.md", "TRELIX_LLM_BASE_URL"),
    ("docs/WHY_TRELIX.md", "TRELIX_LLM_BASE_URL"),
    ("SECURITY.md", "TRELIX_LLM_BASE_URL"),
    ("docs/CONFIGURATION.md", "trelix-local"),
    ("docs/PROVIDERS.md", "trelix-local"),
    ("docs/PROVIDERS.md", "max_tokens"),
    ("docs/PROVIDERS.md", "mixtral-8x7b"),
    # PR 2: the prompt-truncation guard's detail, wherever the review statuses are documented.
    ("docs/CLI_REFERENCE.md", "prompt_truncated"),
    ("docs/CONFIGURATION.md", "prompt_truncated"),
    ("docs/PROVIDERS.md", "prompt_truncated"),
    ("CHANGELOG.md", "prompt_truncated"),
    # PR 3: the context length of the server behind the URL sizes the auto-derived budget.
    (".env.example", "TRELIX_LLM_LOCAL_CONTEXT_TOKENS"),
    ("docs/CONFIGURATION.md", "TRELIX_LLM_LOCAL_CONTEXT_TOKENS"),
    ("docs/PROVIDERS.md", "TRELIX_LLM_LOCAL_CONTEXT_TOKENS"),
    ("docs/CLI_REFERENCE.md", "TRELIX_LLM_LOCAL_CONTEXT_TOKENS"),
    ("README.md", "TRELIX_LLM_LOCAL_CONTEXT_TOKENS"),
    ("docs/PROVIDERS.md", "OLLAMA_CONTEXT_LENGTH"),
    # PR 5: docs/OFFLINE.md and the files that point at it.
    ("docs/OFFLINE.md", "TRELIX_LLM_BASE_URL"),
    ("docs/OFFLINE.md", "TRELIX_LLM_LOCAL_CONTEXT_TOKENS"),
    ("docs/OFFLINE.md", "0.85"),
    ("docs/OFFLINE.md", "20B"),
    ("docs/OFFLINE.md", "TIKTOKEN_CACHE_DIR"),
    ("docs/OFFLINE.md", "OLLAMA_CONTEXT_LENGTH"),
    ("docs/OFFLINE.md", "0.31.0"),
    ("docs/OFFLINE.md", "max_tokens"),
    ("docs/OFFLINE.md", "prompt_truncated"),
    ("docs/OFFLINE.md", "trelix-local"),
    ("docs/OFFLINE.md", "TRELIX_RETRIEVAL_PLAN_CACHE_FILE"),
    ("docs/OFFLINE.md", "HF_HUB_OFFLINE"),
    ("docs/OFFLINE.md", "prefetch_all"),
    ("docs/OFFLINE.md", "## Prefetch"),
    ("docs/OFFLINE.md", "## Troubleshooting"),
    ("docs/OFFLINE.md", "exception:APIConnectionError"),
    ("docs/OFFLINE.md", "#openai-with-a-local-openai-compatible-server"),
    ("docs/README.md", "[OFFLINE.md](OFFLINE.md)"),
    ("docs/USER_GUIDE.md", "OFFLINE.md"),
    ("docs/FAQ.md", "OFFLINE.md"),
    ("docs/TROUBLESHOOTING.md", "TIKTOKEN_CACHE_DIR"),
    ("docs/TROUBLESHOOTING.md", "OFFLINE.md#troubleshooting"),
    ("CHANGELOG.md", "docs/OFFLINE.md"),
]


@pytest.mark.parametrize(("relative_path", "needle"), _PINS)
def test_the_doc_names_it(relative_path: str, needle: str) -> None:
    text = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert needle in text, f"{relative_path} does not mention {needle!r}"


def test_offline_md_carries_no_version_stamp() -> None:
    """The guide carries no release stamp, so no release PR has to touch it.

    test_docs_version_claims.py scans exactly these positions (an H1 and a `Version:` line in
    the first five lines, `trelix vX.Y.Z` in the last three) and would bind the guide to a
    release the moment one appeared; today it would pass (the shipping version matches), so
    this pin is what keeps the stamp out.
    """
    lines = (_REPO_ROOT / "docs/OFFLINE.md").read_text(encoding="utf-8").splitlines()

    assert lines[0].startswith("# ")
    for line in lines[:5]:
        assert re.search(r"\d+\.\d+\.\d+", line) is None, f"version triple in masthead: {line!r}"
    for line in lines[-3:]:
        assert "trelix v" not in line, f"signature stamp in footer: {line!r}"


def test_the_warning_pointers_resolve() -> None:
    """The three warnings in trelix.llm.offline name `docs/OFFLINE.md`; the file and the heading
    the prefetch pointer names must exist. Read as text, not imported: the pin is about the
    source and the guide, not about what the module does at runtime."""
    source = (_REPO_ROOT / "src/trelix/llm/offline.py").read_text(encoding="utf-8")

    assert "docs/OFFLINE.md" in source
    assert "docs/OFFLINE.md: prefetch" in source
    assert (_REPO_ROOT / "docs/OFFLINE.md").is_file()
    guide_lines = (_REPO_ROOT / "docs/OFFLINE.md").read_text(encoding="utf-8").splitlines()
    assert "## Prefetch" in guide_lines


def test_every_plan_cache_offer_names_the_refusal() -> None:
    """A plan file that already holds a plan is replayed and nothing else: a query not in it
    raises `PlanCacheMissError` (`_FrozenPlanCache.plan`, kept outside `route()`'s fallback), and
    `search`, `query` and `ask` exit 1 on it. Every paragraph or table that offers
    `TRELIX_RETRIEVAL_PLAN_CACHE_FILE` must say so, or the Troubleshooting row turns a slow ad-hoc
    `ask` into a failing one. Blocks are blank-line separated, so the table counts once."""
    text = (_REPO_ROOT / "docs/OFFLINE.md").read_text(encoding="utf-8")
    offers = [block for block in text.split("\n\n") if "TRELIX_RETRIEVAL_PLAN_CACHE_FILE" in block]

    assert offers, "the guide no longer offers TRELIX_RETRIEVAL_PLAN_CACHE_FILE"
    for block in offers:
        assert "PlanCacheMissError" in block, f"offer without the refusal: {block[:80]!r}"


def _module_string_constant(source: str, name: str) -> str:
    """The value of the module-level string constant `name` in `source`, read from the AST so an
    implicitly concatenated literal comes back as one string."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str), name
            return value
    raise AssertionError(f"{name} is not a module-level constant")


def test_the_quoted_backend_warnings_match_the_source() -> None:
    """The guide quotes W3 and W4 from `trelix.llm.providers.openai_backend` (numbers in place of
    the `%d` placeholders); a later rewording of either constant must fail here instead of leaving
    the guide quoting an old line. Text is whitespace-normalised because the guide wraps at 100
    columns. MUTATION that must make this fail: reword `it reports` in the guide's W3 quote."""
    source = (_REPO_ROOT / "src/trelix/llm/providers/openai_backend.py").read_text(encoding="utf-8")
    guide = " ".join((_REPO_ROOT / "docs/OFFLINE.md").read_text(encoding="utf-8").split())
    for name in ("_PROMPT_TRUNCATED_WARNING", "_NO_USAGE_WARNING"):
        for fragment in _module_string_constant(source, name).split("%d"):
            fragment = " ".join(fragment.split())
            if len(fragment) > 8:
                assert fragment in guide, (
                    f"{name} fragment missing from docs/OFFLINE.md: {fragment!r}"
                )
