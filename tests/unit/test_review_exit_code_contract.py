"""The exit codes of ``trelix review`` are written down in three languages; they must agree.

``src/trelix/cli/main.py`` decides them: ``REVIEW_NOT_RUN_EXIT_CODE`` (nothing came of the review)
and ``REVIEW_INCOMPLETE_EXIT_CODE`` (it covered only part of the diff, after printing the findings
it has). Two consumers hard-code the same numbers, because neither can import a Python module:

* the GitHub App (``infra/github-app/src/review-outcome.ts``) tells "did not run" from "incomplete"
  by ``err.code`` of the rejected ``execFile``;
* the Actions workflow's inline script (``.github/workflows/trelix-review.yml``) compares the step
  output ``TRELIX_REVIEW_EXIT_CODE`` with two string constants.

Neither language's own suite ties its copy to the other. If one drifted, a partial review would be
published as "did not run" (or, worse, the code of a review that did not run would be read as one
whose findings should be posted), and both sides' tests would still pass. This reads the three
sources as text, the way ``test_github_app_child_env_contract.py`` reads ``child-env.ts``, and
compares the numbers; the Python side is pinned as literals in ``test_review_not_run_exit.py`` too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

import trelix.cli.main as cli_main

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAIN_PY = _REPO_ROOT / "src" / "trelix" / "cli" / "main.py"
_REVIEW_OUTCOME_TS = _REPO_ROOT / "infra" / "github-app" / "src" / "review-outcome.ts"
_REVIEW_RUNNER_TS = _REPO_ROOT / "infra" / "github-app" / "src" / "review-runner.ts"
_CHECK_POSTING_TS = _REPO_ROOT / "infra" / "github-app" / "src" / "check-posting.ts"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "trelix-review.yml"

_NAMES = ("REVIEW_NOT_RUN_EXIT_CODE", "REVIEW_INCOMPLETE_EXIT_CODE")

_PYTHON_CONSTANT = re.compile(r"^(REVIEW_[A-Z_]+_EXIT_CODE)\s*=\s*(\d+)\s*$", re.MULTILINE)
_TS_CONSTANT = re.compile(r"\bexport\s+const\s+(REVIEW_[A-Z_]+_EXIT_CODE)\s*=\s*(\d+)\s*;")
_WORKFLOW_CONSTANTS = {
    "REVIEW_NOT_RUN_EXIT_CODE": re.compile(r"\bconst\s+reviewNotRunExitCode\s*=\s*'(\d+)'\s*;"),
    "REVIEW_INCOMPLETE_EXIT_CODE": re.compile(
        r"\bconst\s+reviewIncompleteExitCode\s*=\s*'(\d+)'\s*;"
    ),
}
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
# A comparison with the bare number 3 or 4 (or '3' / '4'): the constants exist so that nobody
# writes one. 0 is not a review-specific code and stays a literal.
_BARE_COMPARISON = re.compile(r"[=!]==?\s*['\"]?[34]['\"]?(?![\w.])")


def _without_comments(source: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", source))


def _python_constants() -> dict[str, int]:
    text = _MAIN_PY.read_text(encoding="utf-8")
    return {name: int(number) for name, number in _PYTHON_CONSTANT.findall(text)}


def _ts_constants() -> dict[str, int]:
    text = _without_comments(_REVIEW_OUTCOME_TS.read_text(encoding="utf-8"))
    return {name: int(number) for name, number in _TS_CONSTANT.findall(text)}


def _publish_script() -> str:
    steps = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["trelix-review"]["steps"]
    [step] = [s for s in steps if s.get("name") == "Post review as Check annotations"]
    return str(step["with"]["script"])


def _workflow_constants() -> dict[str, int]:
    script = _without_comments(_publish_script())
    found: dict[str, int] = {}
    for name, pattern in _WORKFLOW_CONSTANTS.items():
        matches = pattern.findall(script)
        assert len(matches) == 1, f"{name} must be defined exactly once in the workflow script"
        found[name] = int(matches[0])
    return found


class TestTheNumbers:
    def test_main_py_defines_the_two_codes_as_plain_integer_literals(self) -> None:
        # Precondition for every comparison below: the extraction found both, so an equality
        # check cannot pass because two empty dicts were compared.
        assert _python_constants() == {
            "REVIEW_NOT_RUN_EXIT_CODE": 3,
            "REVIEW_INCOMPLETE_EXIT_CODE": 4,
        }

    @pytest.mark.parametrize("name", _NAMES)
    def test_the_text_of_main_py_is_what_the_module_really_holds(self, name: str) -> None:
        assert getattr(cli_main, name) == _python_constants()[name]

    @pytest.mark.parametrize("name", _NAMES)
    def test_the_app_uses_the_same_number(self, name: str) -> None:
        assert _ts_constants()[name] == _python_constants()[name]

    def test_the_app_defines_exactly_these_two_review_codes(self) -> None:
        assert sorted(_ts_constants()) == sorted(_NAMES)

    @pytest.mark.parametrize("name", _NAMES)
    def test_the_workflow_uses_the_same_number(self, name: str) -> None:
        assert _workflow_constants()[name] == _python_constants()[name]


class TestNobodyBypassesTheConstants:
    """A bare ``=== 3`` next to the constant would let a copy drift without this file noticing."""

    @pytest.mark.parametrize("path", [_REVIEW_OUTCOME_TS, _REVIEW_RUNNER_TS, _CHECK_POSTING_TS])
    def test_the_app_compares_exit_codes_only_through_the_constants(self, path: Path) -> None:
        source = _without_comments(path.read_text(encoding="utf-8"))

        assert _BARE_COMPARISON.findall(source) == []

    def test_the_workflow_compares_exit_codes_only_through_the_constants(self) -> None:
        source = _without_comments(_publish_script())

        assert _BARE_COMPARISON.findall(source) == []

    def test_the_matcher_sees_a_bare_comparison(self) -> None:
        """Control: the scan above is not green because the pattern matches nothing."""
        assert _BARE_COMPARISON.findall("if (code === 3) {}") != []
        assert _BARE_COMPARISON.findall("if (code !== 4) {}") != []
        assert _BARE_COMPARISON.findall("if (code == '4') {}") != []
        assert _BARE_COMPARISON.findall("if (code === REVIEW_NOT_RUN_EXIT_CODE) {}") == []
        assert _BARE_COMPARISON.findall("if (code === 0) {}") == []
        assert _BARE_COMPARISON.findall("if (n === 30) {}") == []
