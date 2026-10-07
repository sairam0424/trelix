"""Doc pins for the local-server feature (PRs 1 and 2 of roadmap item C-7).

Every user-facing file that lists the LLM variables names `TRELIX_LLM_BASE_URL`, the two
that explain it name the bearer and the token-limit field, and every file that promised zero
LLM calls "with no chat credential" now names the variable as the second condition (a keyless
local-server client is a usable client; the planner calls it). The files that document the
review statuses name the `prompt_truncated` detail. Literal (file, needle) pairs, the shape of
test_llm_thinking_mode.py's Opus-boundary pins. MUTATION: delete any row here and the matching
doc line; the row for that file fails.
"""

from __future__ import annotations

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
]


@pytest.mark.parametrize(("relative_path", "needle"), _PINS)
def test_the_doc_names_it(relative_path: str, needle: str) -> None:
    text = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert needle in text, f"{relative_path} does not mention {needle!r}"
