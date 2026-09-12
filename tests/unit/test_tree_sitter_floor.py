"""tree-sitter floor-bump regression guard (py-tree-sitter 0.23 -> 0.26).

WHY THIS EXISTS. py-tree-sitter 0.25.0 moved Query.captures/.matches/.timeout_micros
onto a new QueryCursor class, and 0.26.0 removed Language.version, Language.query(source),
and Parser/QueryCursor.timeout_micros outright (each with a named replacement). trelix's
own grammar loading is fully delegated to tree_sitter_language_pack.get_language()/
get_parser() (src/trelix/indexing/parser/_grammar.py) and every extractor does manual AST
walking via `tree_sitter.Node` only -- confirmed by grep to touch none of the five
removed/relocated APIs anywhere in src/trelix. This test pins that absence as a standing
guard: if a future contributor ever reaches for the old Query API, this fails with a
specific, actionable message instead of a cryptic AttributeError against tree-sitter>=0.26.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src" / "trelix"

_REMOVED_APIS = (
    r"\.captures\(",
    r"\.matches\(",
    r"\.timeout_micros",
    r"Language\.version",
    r"Language\.query\(",
)


def test_no_removed_query_api_call_sites() -> None:
    """None of py-tree-sitter 0.25.0/0.26.0's removed/relocated Query APIs are used."""
    hits: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in _REMOVED_APIS:
            for m in re.finditer(pattern, text):
                line_no = text.count("\n", 0, m.start()) + 1
                hits.append(f"{path.relative_to(_ROOT)}:{line_no}: {pattern}")
    assert not hits, (
        "found call sites using py-tree-sitter APIs removed/relocated in 0.25.0/0.26.0 "
        f"(Query.captures/.matches/.timeout_micros moved to QueryCursor; Language.version, "
        f"Language.query(source), timeout_micros removed outright): {hits}"
    )


def test_grammar_loading_still_resolves() -> None:
    """The single chokepoint for grammar loading still returns usable Language/Parser objects."""
    from trelix.indexing.parser._grammar import load_language, make_parser

    language = load_language("python")
    parser = make_parser("python")
    assert language is not None
    assert parser is not None
    tree = parser.parse(b"def f():\n    pass\n")
    assert tree.root_node.type == "module"
