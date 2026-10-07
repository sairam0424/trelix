"""The plugin skill names only tools the pinned server has, and grants only the read-only ones.

WHY. `plugins/trelix/skills/use-trelix-index/SKILL.md` is what makes Claude reach for trelix
instead of grep: the published `trelix-mcp==3.4.3` the plugin pins sends no server
instructions, so the skill carries every rule. A skill that named a tool the server does not
register (`repo_map`, `exact_search`), a parameter or flag only `develop` has (`detail`, `limit`,
`max_body_chars`, `--tools`), or one of the eight tools `--tools core` hides would read as
authoritative and fail at the first call. And `allowed-tools` is a permission grant: it must list
exactly the three tools `packages/trelix-mcp/tests/test_tool_readonly.py` measures as read-only,
nothing that writes the index or spends money.

Every expected value is a literal. `test_skill_names_only_tools_the_server_registers` compares
the skill against an OBSERVATION of the server (`mcp.list_tools()`), which needs `trelix_mcp`
and its dependency `fastmcp`; CI's unit job installs `packages/trelix-mcp` (ci.yml `pip install
-e packages/trelix-mcp`), so the test runs there, and `pytest.importorskip("trelix_mcp.server")`
inside the function (the repository's pattern, tests/unit/test_dotenv_anchoring.py) skips it in a
venv that lacks either one.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
SKILL = _ROOT / "plugins" / "trelix" / "skills" / "use-trelix-index" / "SKILL.md"

# `--tools core` on develop (packages/trelix-mcp/src/trelix_mcp/tool_metadata.py CORE_TOOLS),
# written out so that a change there has to be reconciled here.
CORE_TOOLS = frozenset(
    {
        "index_codebase",
        "search_code",
        "get_symbol",
        "blast_radius",
        "build_knowledge_graph",
        "graph_search_mcp",
        "ask_agent",
    }
)
# The eight the core profile hides. The skill must not name them in any form.
HIDDEN_TOOLS = (
    "agent_list_sessions",
    "agent_clear_session",
    "federation_list_repos",
    "federation_add_repo",
    "federation_remove_repo",
    "federation_search_all",
    "subscribe_resource",
    "unsubscribe_resource",
)
# Parameters that exist on develop (server.py `detail=`, `max_body_chars=`, `limit=`) and not
# on the pinned 3.4.3, and the `--tools` flag (3.4.3's argparse exits 2 on it). The pin-bump PR
# that moves past 3.4.3 deletes this guard.
DEVELOP_ONLY_PARAMETERS = ("detail", "limit", "max_body_chars")
DEVELOP_ONLY_FLAG = "--tools"
READ_ONLY_GRANT = frozenset(
    {
        "mcp__plugin_trelix_trelix__search_code",
        "mcp__plugin_trelix_trelix__get_symbol",
        "mcp__plugin_trelix_trelix__blast_radius",
    }
)
_TOOL_REF = re.compile(r"mcp__plugin_trelix_trelix__([a-z_]+)")
MAX_BODY_LINES = 150


def _frontmatter_and_body() -> tuple[dict[str, object], str]:
    text = SKILL.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines and lines[0] == "---", "SKILL.md must start with a `---` frontmatter line"
    closing = next(i for i, line in enumerate(lines[1:], start=1) if line == "---")
    frontmatter = yaml.safe_load("\n".join(lines[1:closing]))
    assert isinstance(frontmatter, dict), f"frontmatter did not parse as a mapping: {frontmatter!r}"
    return frontmatter, "\n".join(lines[closing + 1 :])


def _granted_tools(value: object) -> set[str]:
    """`allowed-tools` as a set, from the space- or comma-separated string or the YAML list."""
    if isinstance(value, list):
        return {str(item) for item in value}
    assert isinstance(value, str), f"allowed-tools must be a string or a list, got {value!r}"
    return {part for part in re.split(r"[\s,]+", value.strip()) if part}


def _words(name: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def test_skill_frontmatter_names_its_directory_and_grants_only_read_only_tools() -> None:
    """`name` is the directory (the command is `/trelix:use-trelix-index`), the description is
    non-empty and short enough to survive the listing, and the grant is exactly the three
    tools measured read-only.

    MUTATIONS that must make this fail: add `mcp__plugin_trelix_trelix__index_codebase` to
    `allowed-tools`; rename the skill directory; empty the description.
    """
    frontmatter, _ = _frontmatter_and_body()

    assert frontmatter.get("name") == "use-trelix-index"
    assert frontmatter["name"] == SKILL.parent.name, "frontmatter name must equal the directory"

    description = frontmatter.get("description")
    assert isinstance(description, str)
    # 1024 is this project's ceiling; Claude Code truncates the listing at 1,536 characters.
    assert 1 <= len(description) <= 1024, f"description is {len(description)} characters"

    assert _granted_tools(frontmatter.get("allowed-tools")) == READ_ONLY_GRANT, (
        "allowed-tools must pre-approve exactly search_code, get_symbol and blast_radius: the "
        "tools packages/trelix-mcp/tests/test_tool_readonly.py measures as read-only. Every "
        "other tool writes the index or spends money and must keep its prompt."
    )


def test_skill_body_names_only_core_tools_and_no_develop_only_parameters() -> None:
    """Every `mcp__plugin_trelix_trelix__<tool>` reference anywhere in the file (frontmatter
    included) is a core tool, the body names `search_code` in that full form, no hidden tool
    appears anywhere in the file, no develop-only parameter appears in the body, and `--tools`
    appears nowhere in the file.

    MUTATIONS that must make this fail: write `federation_search_all` in the body; reference
    `mcp__plugin_trelix_trelix__repo_map`; put `mcp__plugin_trelix_trelix__federation_search_all`
    in the frontmatter description; write `detail="concise"`; write `--tools core` in the body or
    in the description; shorten the body's `mcp__plugin_trelix_trelix__search_code(` to
    `search_code(`.
    """
    _, body = _frontmatter_and_body()
    whole = SKILL.read_text(encoding="utf-8")

    referenced = set(_TOOL_REF.findall(whole))
    assert referenced <= CORE_TOOLS, f"not core tools: {sorted(referenced - CORE_TOOLS)}"
    assert "search_code" in set(_TOOL_REF.findall(body)), (
        "the body must name search_code with its full tool name; the frontmatter grant alone "
        "does not teach it"
    )

    hidden = [name for name in HIDDEN_TOOLS if _words(name, whole)]
    assert hidden == [], f"the skill names tools the core profile hides: {hidden}"

    # `\b` does not bound `--`, so the flag is a plain substring check. Unlike `detail` and
    # `limit` it has no plain-English collision, so the whole file is checked (the description is
    # what the model reads at listing time).
    develop_only = [name for name in DEVELOP_ONLY_PARAMETERS if _words(name, body)] + (
        [DEVELOP_ONLY_FLAG] if DEVELOP_ONLY_FLAG in whole else []
    )
    assert develop_only == [], (
        f"the skill names parameters the pinned trelix-mcp 3.4.3 does not have: {develop_only}"
    )


def test_skill_body_is_at_most_150_lines() -> None:
    """Once loaded, the body stays in context for the rest of the session.

    MUTATION that must make this fail: append 151 lines to the body.
    """
    _, body = _frontmatter_and_body()
    count = len(body.splitlines())
    assert count <= MAX_BODY_LINES, f"skill body is {count} lines"


def test_skill_names_only_tools_the_server_registers() -> None:
    """The literal core set, and every tool the skill names, are tools the server registers.

    `mcp.list_tools()` is an observation of the server in this checkout, not an imported
    expected value. Needs `trelix_mcp` and `fastmcp`, which CI's unit job installs.

    MUTATION that must make this fail: reference `mcp__plugin_trelix_trelix__repo_map`.
    """
    server_module = pytest.importorskip("trelix_mcp.server")

    registered = {tool.name for tool in asyncio.run(server_module.mcp.list_tools())}
    assert len(registered) >= 7, f"the server lists only {sorted(registered)}"

    referenced = set(_TOOL_REF.findall(SKILL.read_text(encoding="utf-8")))
    assert CORE_TOOLS <= registered, f"core tools absent: {sorted(CORE_TOOLS - registered)}"
    assert referenced <= registered, f"not registered: {sorted(referenced - registered)}"
