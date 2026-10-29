"""A crafted comment must not stall the Go, Java, JavaScript, TypeScript, Rust, JSON and C# parsers.

These extractors cleaned a doc comment with regular expressions that backtrack quadratically on a
long run of whitespace or blank lines (and, for C#, of ``<``), so one source file with such a
comment held the indexer's worker for minutes. Every hostile case goes through the parser's
``parse`` with an input on which the old patterns needed 10 seconds or more, and checks both the
running time and the docstring the old code produced. The golden cases pin the ordinary comment
styles of each language to what the old code returned.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import NamedTuple

import pytest

from trelix.indexing.parser.base import BaseParser, ParseResult
from trelix.indexing.parser.extractors.csharp import CSharpParser
from trelix.indexing.parser.extractors.go import GoParser
from trelix.indexing.parser.extractors.java import JavaParser
from trelix.indexing.parser.extractors.javascript import JavaScriptParser
from trelix.indexing.parser.extractors.json_config import JsonParser
from trelix.indexing.parser.extractors.rust import RustParser
from trelix.indexing.parser.extractors.typescript import TypeScriptParser

FILE_ID = 7
# Where the old patterns took 10 s or more on the slowest languages; the new code needs ms.
RUN_LENGTH = 64_000
LESS_THAN_RUN_LENGTH = 160_000
MAX_SECONDS = 2.0


class HostileCase(NamedTuple):
    parser: type[BaseParser]
    source: Callable[[int], str]
    symbol: str
    docstring: Callable[[int], str]
    run_length: int


def _docstrings(result: ParseResult) -> dict[str, str]:
    return {s.name: s.docstring for s in result.symbols if s.docstring}


def _timed_parse(
    parser_cls: type[BaseParser], warm_up_source: str, source: str
) -> tuple[ParseResult, float]:
    """Parse ``source`` and return the result with the seconds ``parse`` took.

    The parser is built and used once on ``warm_up_source`` first: building it, and the JSON
    parser's lazy grammar load on its first ``parse``, are not what is being timed.
    """
    parser = parser_cls()
    parser.parse(warm_up_source, FILE_ID)
    started = time.perf_counter()
    result = parser.parse(source, FILE_ID)
    return result, time.perf_counter() - started


def _spaces(n: int) -> str:
    return " " * n


def _blank_lines(n: int) -> str:
    return "\n" * n


HOSTILE_CASES = {
    "go-spaces": HostileCase(
        GoParser,
        lambda n: "package p\n\n/* a" + _spaces(n) + "b */\nfunc F(){}",
        "F",
        lambda n: "a" + _spaces(n) + "b",
        RUN_LENGTH,
    ),
    "go-blank-lines": HostileCase(
        GoParser,
        lambda n: "package p\n\n/* a" + _blank_lines(n) + "b */\nfunc F(){}",
        "F",
        lambda n: "a" + _blank_lines(n) + "b",
        RUN_LENGTH,
    ),
    "java-spaces": HostileCase(
        JavaParser,
        lambda n: "/** a" + _spaces(n) + "b */\nclass C {}\n",
        "C",
        lambda n: "a" + _spaces(n) + "b",
        RUN_LENGTH,
    ),
    "java-blank-lines": HostileCase(
        JavaParser,
        lambda n: "/** a" + _blank_lines(n) + "b */\nclass C {}\n",
        "C",
        lambda n: "a" + _blank_lines(n) + "b",
        RUN_LENGTH,
    ),
    "javascript-spaces": HostileCase(
        JavaScriptParser,
        lambda n: "/** a" + _spaces(n) + "b */\nfunction f() {}\n",
        "f",
        lambda n: "a" + _spaces(n) + "b",
        RUN_LENGTH,
    ),
    "javascript-blank-lines": HostileCase(
        JavaScriptParser,
        lambda n: "/** a" + _blank_lines(n) + "b */\nfunction f() {}\n",
        "f",
        lambda n: "a" + _blank_lines(n) + "b",
        RUN_LENGTH,
    ),
    "typescript-spaces": HostileCase(
        TypeScriptParser,
        lambda n: "/** a" + _spaces(n) + "b */\nexport function f(): void {}\n",
        "f",
        lambda n: "a" + _spaces(n) + "b",
        RUN_LENGTH,
    ),
    "typescript-blank-lines": HostileCase(
        TypeScriptParser,
        lambda n: "/** a" + _blank_lines(n) + "b */\nexport function f(): void {}\n",
        "f",
        lambda n: "a" + _blank_lines(n) + "b",
        RUN_LENGTH,
    ),
    "rust-spaces": HostileCase(
        RustParser,
        lambda n: "/** a" + _spaces(n) + "b */\nfn f() {}\n",
        "f",
        lambda n: "a" + _spaces(n) + "b",
        RUN_LENGTH,
    ),
    "rust-blank-lines": HostileCase(
        RustParser,
        lambda n: "/** a" + _blank_lines(n) + "b */\nfn f() {}\n",
        "f",
        lambda n: "a" + _blank_lines(n) + "b",
        RUN_LENGTH,
    ),
    "json-spaces": HostileCase(
        JsonParser,
        lambda n: "{\n/* a" + _spaces(n) + 'b */\n"k": 1\n}',
        "k",
        lambda n: "a" + _spaces(n) + "b",
        RUN_LENGTH,
    ),
    "json-blank-lines": HostileCase(
        JsonParser,
        lambda n: "{\n/* a" + _blank_lines(n) + 'b */\n"k": 1\n}',
        "k",
        lambda n: "a" + _blank_lines(n) + "b",
        RUN_LENGTH,
    ),
}


@pytest.mark.timeout(30)
@pytest.mark.parametrize("case", HOSTILE_CASES.values(), ids=list(HOSTILE_CASES))
def test_hostile_block_comment_parses_fast_and_keeps_its_docstring(case: HostileCase):
    source = case.source(case.run_length)

    result, elapsed = _timed_parse(case.parser, case.source(1), source)

    assert _docstrings(result)[case.symbol] == case.docstring(case.run_length)
    assert elapsed < MAX_SECONDS


@pytest.mark.timeout(30)
def test_csharp_long_run_of_less_than_signs_in_a_doc_comment_parses_fast():
    source = "/// " + "<" * LESS_THAN_RUN_LENGTH + "\nclass C {}"

    result, elapsed = _timed_parse(CSharpParser, "/// <\nclass C {}", source)
    docstrings = _docstrings(result)

    # Both the class doc and the file-level doc run the same tag strip.
    assert docstrings["C"] == "<" * LESS_THAN_RUN_LENGTH
    assert docstrings["<module>"] == "<" * LESS_THAN_RUN_LENGTH
    assert elapsed < MAX_SECONDS


class GoldenCase(NamedTuple):
    parser: type[BaseParser]
    source: str
    symbol: str
    docstring: str


BLOCK = "/*\n * Foo does things.\n *   indented line\n *\n * Last line.\n */\n"
BLOCK_DOC = "/**\n * Foo does things.\n *   indented line\n *\n * Last line.\n */\n"
BLOCK_TEXT = "Foo does things.\n  indented line\nLast line."
SPACED = "/** Spaced   \t \n  *   keeps inner spacing.   \n  */\n"
SPACED_TEXT = "Spaced   \t \n  keeps inner spacing."
STARS_CLOSE = "/**\n * Stars close.\n **/\n"

GOLDEN_CASES = {
    "go-star-prefixed-block": GoldenCase(
        GoParser, "package p\n\n" + BLOCK + "func Foo() {}\n", "Foo", BLOCK_TEXT
    ),
    "go-one-line-block": GoldenCase(
        GoParser, "package p\n\n/** One line. */\nfunc Foo() {}\n", "Foo", "One line."
    ),
    "go-line-comments": GoldenCase(
        GoParser,
        "package p\n\n// First line.\n// Second line.\nfunc Foo() {}\n",
        "Foo",
        "First line.\nSecond line.",
    ),
    "go-spaced-block": GoldenCase(
        GoParser, "package p\n\n" + SPACED + "func Foo() {}\n", "Foo", SPACED_TEXT
    ),
    "java-star-prefixed-block": GoldenCase(JavaParser, BLOCK_DOC + "class C {}\n", "C", BLOCK_TEXT),
    "java-one-line-block": GoldenCase(
        JavaParser, "/** One line. */\nclass C {}\n", "C", "One line."
    ),
    "java-line-comments": GoldenCase(
        JavaParser,
        "// First line.\n// Second line.\nclass C {}\n",
        "C",
        "First line.\nSecond line.",
    ),
    "java-double-star-close": GoldenCase(
        JavaParser, STARS_CLOSE + "class C {}\n", "C", "Stars close."
    ),
    "javascript-star-prefixed-block": GoldenCase(
        JavaScriptParser, BLOCK_DOC + "function foo() {}\n", "foo", BLOCK_TEXT
    ),
    "javascript-triple-slash": GoldenCase(
        JavaScriptParser,
        "/// First line.\n/// Second line.\nfunction foo() {}\n",
        "foo",
        "First line.\nSecond line.",
    ),
    "javascript-double-star-close": GoldenCase(
        JavaScriptParser, STARS_CLOSE + "function foo() {}\n", "foo", "Stars close."
    ),
    "javascript-spaced-block": GoldenCase(
        JavaScriptParser, SPACED + "function foo() {}\n", "foo", SPACED_TEXT
    ),
    "typescript-star-prefixed-block": GoldenCase(
        TypeScriptParser, BLOCK_DOC + "export function foo(): void {}\n", "foo", BLOCK_TEXT
    ),
    "typescript-triple-slash": GoldenCase(
        TypeScriptParser,
        "/// First line.\n/// Second line.\nexport function foo(): void {}\n",
        "foo",
        "First line.\nSecond line.",
    ),
    "typescript-double-star-close": GoldenCase(
        TypeScriptParser, STARS_CLOSE + "export function foo(): void {}\n", "foo", "Stars close."
    ),
    "typescript-spaced-block": GoldenCase(
        TypeScriptParser, SPACED + "export function foo(): void {}\n", "foo", SPACED_TEXT
    ),
    "rust-star-prefixed-block": GoldenCase(
        RustParser, BLOCK_DOC + "fn foo() {}\n", "foo", BLOCK_TEXT
    ),
    "rust-triple-slash": GoldenCase(
        RustParser,
        "/// First line.\n/// Second line.\nfn foo() {}\n",
        "foo",
        "First line.\nSecond line.",
    ),
    "rust-double-star-close": GoldenCase(
        RustParser, STARS_CLOSE + "fn foo() {}\n", "foo", "Stars close."
    ),
    "rust-spaced-block": GoldenCase(RustParser, SPACED + "fn foo() {}\n", "foo", SPACED_TEXT),
    "json-star-prefixed-block": GoldenCase(
        JsonParser, "{\n" + BLOCK + '"k": 1\n}', "k", BLOCK_TEXT
    ),
    "json-one-line-block": GoldenCase(
        JsonParser, '{\n/** One line. */\n"k": 1\n}', "k", "One line."
    ),
    "json-spaced-block": GoldenCase(JsonParser, "{\n" + SPACED + '"k": 1\n}', "k", SPACED_TEXT),
    "csharp-summary": GoldenCase(
        CSharpParser,
        "/// <summary>Does a thing.</summary>\nclass C {}\n",
        "C",
        "Does a thing.",
    ),
    "csharp-param-and-returns": GoldenCase(
        CSharpParser,
        '/// <param name="x">The x.</param>\n/// <returns>Y</returns>\nclass C {}\n',
        "C",
        "The x. Y",
    ),
}


@pytest.mark.parametrize("case", GOLDEN_CASES.values(), ids=list(GOLDEN_CASES))
def test_ordinary_comment_styles_keep_their_docstrings(case: GoldenCase):
    result = case.parser().parse(case.source, FILE_ID)
    assert _docstrings(result)[case.symbol] == case.docstring


def test_csharp_leading_doc_comment_is_also_the_file_doc():
    source = "/// <summary>Does a thing.</summary>\nclass C {}\n"

    result = CSharpParser().parse(source, FILE_ID)

    assert _docstrings(result)["<module>"] == "Does a thing."
