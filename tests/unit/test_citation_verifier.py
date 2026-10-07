"""`verify_citations`: the model's `[C#]` markers checked against the index and the disk.

Third of six changes toward `trelix ask` answers that cite the code they rest on (C-5
design, R3, as amended by the critique's B1, B5(b) and C.3). Nothing calls the verifier
yet; this file is its whole contract. The fixture is the design's: sources S over a
repository F2 built in `tmp_path` (`src/auth/middleware.py` = 80 newline-terminated
lines, `src/auth/jwt.py` = 25 lines with NO trailing newline, no `src/auth/old.py`), and
the answer A1 that cites them well, badly, twice and not at all.

The line convention is the extractors' own, and the reason B1 exists: tree-sitter's root
node ends on the row after the final newline, so the `<module>` symbol of a newline-
terminated 7-line file is `1-8`; `count_lines` therefore returns `count("\\n") + 1` (81 for
middleware.py, 25 for jwt.py, 1 for an empty file) and the rule stays `line_end > count`.
`test_a_whole_file_symbol_from_the_real_python_parser_is_valid` runs the real parser so the
convention is pinned against the code that produces the ranges, not against a belief.

Every expected value is a literal in this file. MUTATIONS, each named by the test it breaks:
`>=` instead of `>` in the range check -> test_a_chunk_ending_on_the_last_line_* ([C5] flips);
`> lines + 1` instead of `> lines` -> test_a_chunk_ending_one_line_past_the_end_* ([C8] flips);
`count("\\n")` without the `+ 1` -> test_a_whole_file_symbol_of_a_newline_terminated_* and
    the real-parser test (1-81 and 1-8 flip to line_out_of_range);
an unknown tag treated as valid -> test_the_design_answer_* ([C9]);
`file_missing` checked before `unknown` -> test_the_design_answer_* ([C9] raises on None.path);
`exists()` instead of `is_file()` -> test_a_cited_path_that_is_now_a_directory_* (raises);
regex `\\d+` instead of `[1-9][0-9]{0,2}` -> test_find_markers_* and test_the_design_answer_*
    ([C0], [C007], [C1000] gain entries);
regex `[Cc]`, `re.IGNORECASE` or `C ?` -> the same two tests ([c4]/[c5] or [C 5]/[C 6] gain
    an entry; their tags are cited nowhere else, so the dedupe cannot hide them);
dedupe by sorted tag instead of first appearance -> test_markers_keep_first_appearance_order;
the per-call line-count cache dropped -> test_one_cited_file_is_opened_once_*;
whole-file `read()` instead of 1 MiB pieces -> test_count_lines_reads_a_large_file_in_*;
the detail literals changed -> test_the_design_answer_* (compared whole);
`frozen=True` dropped from Citation -> test_a_citation_is_immutable.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from trelix.core.models import Language
from trelix.indexing.parser.registry import get_parser
from trelix.retrieval.citations import (
    Citation,
    CitationSource,
    count_lines,
    find_markers,
    verify_citations,
)

MIDDLEWARE = "src/auth/middleware.py"
JWT = "src/auth/jwt.py"
OLD = "src/auth/old.py"

# S: what the assembler recorded for the tags. Tag 4 names a file that is gone; tag 5 ends
# exactly on the last line of the file without a trailing newline.
S = (
    CitationSource(1, 11, MIDDLEWARE, 42, 67, "AuthMiddleware.verify"),
    CitationSource(2, 33, MIDDLEWARE, 70, 80, "AuthMiddleware.bearer"),
    CitationSource(3, 22, JWT, 10, 30, "decode_token"),
    CitationSource(4, 44, OLD, 1, 20, "OldAuth.check"),
    CitationSource(5, 55, JWT, 20, 25, "jwt_helper"),
)

# A1: good, stale, missing, invented and repeated markers, and five things that look like
# markers but are text. The lower-case and spaced ones carry tags A1 does not otherwise cite
# (5 is in S, 6 is not), so a regex that admitted them would add a row instead of being
# hidden by the dedupe.
A1 = (
    "The token is read from the header [C2] and verified by `verify` [C1]. "
    "Decoding happens in `decode_token` [C3][C1]. Legacy checks lived in OldAuth [C4]. "
    "Nothing supports [C9]. Not markers: [C0] [C007] [c5] [C 6] [C1000]."
)

ONE_MIB = 1048576


def _rows(citations: list[Citation]) -> list[tuple[Any, ...]]:
    return [
        (c.marker, c.tag, c.status, c.path, c.line_start, c.line_end, c.symbol, c.detail)
        for c in citations
    ]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """F2 on disk."""
    auth = tmp_path / "src" / "auth"
    auth.mkdir(parents=True)
    (auth / "middleware.py").write_bytes("".join(f"line {i}\n" for i in range(1, 81)).encode())
    (auth / "jwt.py").write_bytes("\n".join(f"line {i}" for i in range(1, 26)).encode())
    return tmp_path


# ---------------------------------------------------------------------------
# verify_citations
# ---------------------------------------------------------------------------


def test_the_design_answer_over_the_design_sources(repo: Path) -> None:
    """R3: five distinct markers in first-appearance order, each with its literal verdict;
    [C1] once although written twice; the five non-markers produce nothing."""
    assert _rows(verify_citations(A1, S, repo)) == [
        ("[C2]", 2, "valid", MIDDLEWARE, 70, 80, "AuthMiddleware.bearer", ""),
        ("[C1]", 1, "valid", MIDDLEWARE, 42, 67, "AuthMiddleware.verify", ""),
        (
            "[C3]",
            3,
            "line_out_of_range",
            JWT,
            10,
            30,
            "decode_token",
            "src/auth/jwt.py has 25 lines, the cited chunk ends at line 30; re-index",
        ),
        (
            "[C4]",
            4,
            "file_missing",
            OLD,
            1,
            20,
            "OldAuth.check",
            "src/auth/old.py is not in the repository; re-index",
        ),
        ("[C9]", 9, "unknown", None, None, None, None, "no retrieved chunk has this tag"),
    ]


def test_an_answer_without_markers_has_no_citations(repo: Path) -> None:
    assert verify_citations("plain answer", S, repo) == []


def test_a_marker_with_no_sources_at_all_is_unknown(repo: Path) -> None:
    """A unit fact only: the CLI never calls the verifier when `citation_sources` is empty."""
    assert _rows(verify_citations("[C1]", (), repo)) == [
        ("[C1]", 1, "unknown", None, None, None, None, "no retrieved chunk has this tag"),
    ]


def test_a_chunk_ending_on_the_last_line_of_an_unterminated_file_is_valid(repo: Path) -> None:
    """jwt.py has 25 lines and no trailing newline; `jwt_helper` ends on line 25.
    MUTATION: `>=` for `>` calls this line_out_of_range."""
    assert _rows(verify_citations("[C5]", S, repo)) == [
        ("[C5]", 5, "valid", JWT, 20, 25, "jwt_helper", ""),
    ]


def test_a_chunk_ending_one_line_past_the_end_of_a_file_is_out_of_range(repo: Path) -> None:
    """jwt.py has 25 lines; a chunk recorded as ending on line 26 (the file lost its last line
    after indexing) is stale by exactly one line. MUTATION: `> lines + 1` for `> lines` calls
    this valid; [C5] above pins the other side of the same boundary."""
    tail = CitationSource(8, 88, JWT, 20, 26, "jwt_tail")
    assert _rows(verify_citations("[C8]", (tail,), repo)) == [
        (
            "[C8]",
            8,
            "line_out_of_range",
            JWT,
            20,
            26,
            "jwt_tail",
            "src/auth/jwt.py has 25 lines, the cited chunk ends at line 26; re-index",
        ),
    ]


def test_a_cited_path_that_is_now_a_directory_is_file_missing(repo: Path) -> None:
    """The index recorded `src/auth/pkg` as a file; the tree now has a directory of that name.
    MUTATION: `exists()` for `is_file()` lets `count_lines` open the directory and raise
    IsADirectoryError out of the verifier."""
    (repo / "src" / "auth" / "pkg").mkdir()
    pkg = CitationSource(9, 99, "src/auth/pkg", 1, 5, "pkg")
    assert _rows(verify_citations("[C9]", (pkg,), repo)) == [
        (
            "[C9]",
            9,
            "file_missing",
            "src/auth/pkg",
            1,
            5,
            "pkg",
            "src/auth/pkg is not in the repository; re-index",
        ),
    ]


def test_a_whole_file_symbol_of_a_newline_terminated_file_is_valid(repo: Path) -> None:
    """The extractors give a whole-file symbol of an 80-line newline-terminated file the
    range 1-81. MUTATION: `count("\\n")` without the `+ 1` counts 80 and flips this."""
    module = CitationSource(6, 66, MIDDLEWARE, 1, 81, "<module>")
    assert _rows(verify_citations("[C6]", (module,), repo)) == [
        ("[C6]", 6, "valid", MIDDLEWARE, 1, 81, "<module>", ""),
    ]


def test_a_whole_file_symbol_from_the_real_python_parser_is_valid(tmp_path: Path) -> None:
    """The bug B1 exists for: on a fresh index, a `<module>` citation must verify."""
    source = '"""Module docstring."""\n\nX = 1\n\n\ndef f():\n    return X\n'
    (tmp_path / "mod.py").write_text(source, encoding="utf-8")
    parser = get_parser(Language.PYTHON)
    assert parser is not None
    module = next(s for s in parser.parse(source, 1).symbols if s.qualified_name == "<module>")
    assert (module.line_start, module.line_end) == (1, 8)  # seven lines, eight rows
    assert count_lines(tmp_path / "mod.py") == 8
    cited = CitationSource(1, 1, "mod.py", module.line_start, module.line_end, "<module>")
    assert _rows(verify_citations("[C1]", (cited,), tmp_path)) == [
        ("[C1]", 1, "valid", "mod.py", 1, 8, "<module>", ""),
    ]


def test_a_file_summary_source_with_a_negative_symbol_id_is_verified_like_any_other(
    repo: Path,
) -> None:
    """`TRELIX_RETRIEVAL_FILE_SUMMARY_LEG` tags a summary block with a synthetic
    `symbol_id < 0` and a real range; the verifier reads the path and lines only."""
    summary = CitationSource(7, -7, MIDDLEWARE, 42, 80, "AuthMiddleware")
    assert _rows(verify_citations("[C7]", (summary,), repo)) == [
        ("[C7]", 7, "valid", MIDDLEWARE, 42, 80, "AuthMiddleware", ""),
    ]


def test_glued_markers_are_distinct_entries_in_order(repo: Path) -> None:
    """C.3: `[C1][C2][C1]` is two citations, first appearance first."""
    assert _rows(verify_citations("[C1][C2][C1]", S, repo)) == [
        ("[C1]", 1, "valid", MIDDLEWARE, 42, 67, "AuthMiddleware.verify", ""),
        ("[C2]", 2, "valid", MIDDLEWARE, 70, 80, "AuthMiddleware.bearer", ""),
    ]


def test_markers_keep_first_appearance_order(repo: Path) -> None:
    """MUTATION: dedupe through `sorted(set(...))` returns 1, 2, 3."""
    assert [c.tag for c in verify_citations("[C3] [C1] [C2] [C1]", S, repo)] == [3, 1, 2]


def test_a_marker_inside_a_code_span_counts(repo: Path) -> None:
    """The regex has no notion of context (docstring of MARKER_RE)."""
    assert _rows(verify_citations("see `[C1]` here", S, repo)) == [
        ("[C1]", 1, "valid", MIDDLEWARE, 42, 67, "AuthMiddleware.verify", ""),
    ]


def test_one_cited_file_is_opened_once_for_two_markers(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Line counts are cached per path within one call. MUTATION: drop the cache and
    middleware.py is opened twice."""
    opened: list[str] = []
    real_open = Path.open

    def spy(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(self.name)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    result = verify_citations("[C1] and [C2]", S, repo)
    assert [c.status for c in result] == ["valid", "valid"]
    assert opened == ["middleware.py"]


def test_a_citation_is_immutable() -> None:
    citation = Citation("[C1]", 1, "valid", MIDDLEWARE, 42, 67, "AuthMiddleware.verify", "")
    with pytest.raises(FrozenInstanceError):
        citation.status = "unknown"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# find_markers
# ---------------------------------------------------------------------------


def test_find_markers_returns_distinct_markers_in_first_appearance_order() -> None:
    """[C1]..[C999] only; [C0], [C007], [c4], [C 5] and [C1000] are text. Tags 4 and 5 appear
    nowhere else in the text, so an admitted non-marker is a new entry, not a duplicate.
    MUTATIONS: `\\d+` in the regex admits [C0], [C007] and [C1000]; `[Cc]` or `re.IGNORECASE`
    admits [c4]; `C ?` admits [C 5]."""
    text = "[C2] x [C1] y [C2] [C0] [C007] [c4] [C 5] [C1000] [C999]"
    assert find_markers(text) == [("[C2]", 2), ("[C1]", 1), ("[C999]", 999)]


def test_find_markers_on_text_without_markers_is_empty() -> None:
    assert find_markers("plain answer") == []


# ---------------------------------------------------------------------------
# count_lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(b"", 1, id="empty-file-counts-one"),
        pytest.param(b"a", 1, id="one-byte-no-newline"),
        pytest.param(b"a\nb", 2, id="two-lines-no-trailing-newline"),
        pytest.param(b"a\n", 2, id="one-line-newline-terminated-counts-two"),
    ],
)
def test_count_lines_is_newlines_plus_one(tmp_path: Path, content: bytes, expected: int) -> None:
    path = tmp_path / "f.txt"
    path.write_bytes(content)
    assert count_lines(path) == expected


def test_count_lines_on_the_fixture_files(repo: Path) -> None:
    assert count_lines(repo / MIDDLEWARE) == 81
    assert count_lines(repo / JWT) == 25


class _ReadSpy:
    """A file handle that records the size of every `read` it is asked for."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.sizes.append(size)
        return bytes(self._handle.read(size))

    def __enter__(self) -> _ReadSpy:
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.close()


def test_count_lines_reads_a_large_file_in_one_mib_pieces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 MiB of 1024-byte lines: three full pieces and the empty read that ends the loop.
    MUTATION: a whole-file `read()` records `[-1, -1]`; a 2 MiB piece records two reads."""
    big = tmp_path / "big.txt"
    big.write_bytes((b"x" * 1023 + b"\n") * 3072)
    spies: list[_ReadSpy] = []
    real_open = Path.open

    def spy_open(self: Path, *args: Any, **kwargs: Any) -> _ReadSpy:
        spy = _ReadSpy(real_open(self, *args, **kwargs))
        spies.append(spy)
        return spy

    monkeypatch.setattr(Path, "open", spy_open)
    assert count_lines(big) == 3073
    assert [spy.sizes for spy in spies] == [[ONE_MIB, ONE_MIB, ONE_MIB, ONE_MIB]]
