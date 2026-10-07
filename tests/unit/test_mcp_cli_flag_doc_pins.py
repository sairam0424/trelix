"""The five documentation sentences that list every flag `trelix-mcp` accepts, pinned to the parser.

docs/MCP_GUIDE.md (twice), docs/TROUBLESHOOTING.md, docs/USER_GUIDE.md and
docs/integrations/vscode-plugin.md each tell a reader that `trelix-mcp` "accepts only" a fixed list
of flags, so that nobody hunts for a `--cache-dir` or a log-level option that does not exist. When
`--root PATH` arrived, all five kept naming `--help`, `--version` and `--tools core|full`, and
MCP_GUIDE.md contradicted its own section 8, which documents `--root`. The first pin holds each
sentence to the complete list, in that document's own spelling (code spans, a table cell with an
escaped pipe, a shell comment wrapped onto two lines). The second holds the list to the flags
`cli.py` declares, read as text because this suite's environment has no fastmcp to import the
module with: a new flag fails here until every sentence names it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CLI = _ROOT / "packages" / "trelix-mcp" / "src" / "trelix_mcp" / "cli.py"
_SENTENCES = [
    pytest.param(
        "docs/MCP_GUIDE.md",
        "`trelix-mcp` accepts only `--help`, `--version`, `--tools core|full` and `--root PATH` "
        "(section 8)",
        id="mcp-guide-install",
    ),
    pytest.param(
        "docs/MCP_GUIDE.md",
        "`trelix-mcp` accepts only `--help`, `--version`, `--tools core|full` and `--root PATH`, "
        "and has no",
        id="mcp-guide-logging",
    ),
    pytest.param(
        "docs/TROUBLESHOOTING.md",
        "# Verify the binary is on PATH. trelix-mcp accepts only --help, --version,\n"
        "# --tools core|full and --root PATH",
        id="troubleshooting",
    ),
    pytest.param(
        "docs/USER_GUIDE.md",
        "nothing — `trelix-mcp` accepts only `--help`, `--version`, `--tools core\\|full` and "
        "`--root PATH` |",
        id="user-guide",
    ),
    pytest.param(
        "docs/integrations/vscode-plugin.md",
        "`trelix-mcp` accepts only `--help`, `--version`, `--tools core|full` and `--root PATH`; "
        "invoking",
        id="vscode-plugin",
    ),
]
# A declared flag is every `--name` written inside a `parser.add_argument(...)` call, wherever it
# stands (`add_argument("-t", "--transport", ...)` counts); `-h`/`--help` is argparse's own.
_ADD_ARGUMENT_CALL = re.compile(r"add_argument\(([^)]*)")
_LONG_FLAG = re.compile(r'"(--[\w-]+)"')


def _declared_flags(source: str) -> set[str]:
    return {
        flag for call in _ADD_ARGUMENT_CALL.findall(source) for flag in _LONG_FLAG.findall(call)
    }


@pytest.mark.parametrize(("document", "sentence"), _SENTENCES)
def test_each_document_names_every_flag(document: str, sentence: str) -> None:
    assert sentence in (_ROOT / document).read_text(encoding="utf-8")


def test_the_documented_list_is_the_parser_flag_list() -> None:
    """`--version`, `--tools` and `--root`, and nothing else: a fourth flag must reach all five."""
    declared = _declared_flags(_CLI.read_text(encoding="utf-8"))
    assert declared == {"--version", "--tools", "--root"}
