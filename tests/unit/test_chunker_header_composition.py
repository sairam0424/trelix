"""Mutation-pinned tests for how Chunker composes a chunk's context header.

Round 1 pinned the token-budget boundary (`>` vs `>=`). These tests pin the
*text composition* around it: the import header's two independent slice bounds,
the exact line structure of chunk_text, the docstring de-duplication guard, and
the parent-class lookup guard. Every one of these decides what the embedding
provider actually sees, so a silent change here corrupts vectors without any
existing test noticing.

Each test's docstring names the mutation in src/trelix/indexing/chunker.py that
must make it fail. All expected values below are written as literals; nothing is
imported from, or computed by, the module under test.
"""

from __future__ import annotations

from trelix.core.config import ChunkerConfig
from trelix.core.models import ImportEdge, Symbol, SymbolKind
from trelix.indexing.chunker import Chunker

_REL_PATH = "src/pkg/mod.py"
_LANGUAGE = "python"


def _symbol(
    *,
    body: str,
    docstring: str | None = None,
    parent_id: int | None = None,
    id: int = 1,
    name: str = "my_func",
    kind: SymbolKind = SymbolKind.FUNCTION,
) -> Symbol:
    return Symbol(
        id=id,
        file_id=1,
        name=name,
        qualified_name=name,
        kind=kind,
        line_start=1,
        line_end=2,
        signature=f"def {name}()",
        body=body,
        docstring=docstring,
        parent_id=parent_id,
    )


def _build_one(
    chunker: Chunker,
    symbol: Symbol,
    imports: list[ImportEdge] | None = None,
    parent_symbols: dict[int, Symbol] | None = None,
) -> str:
    chunks = chunker.build_chunks([symbol], imports or [], _REL_PATH, _LANGUAGE, parent_symbols)
    assert len(chunks) == 1, f"expected exactly one chunk per symbol, got {len(chunks)}"
    return chunks[0].chunk_text


# ---------------------------------------------------------------------------
# Import header: two independent slice bounds
# ---------------------------------------------------------------------------


class TestImportHeaderExactComposition:
    """_build_import_header has TWO independent slice bounds --
    imports[:max_imports_in_header] and imported_names[:3] -- plus a join
    separator. The pre-existing suite only asserts substring presence/absence
    ("module_2" not in text), so shrinking either bound by one silently drops
    real content from the header of every chunk in the index and stays green.
    """

    # max_imports_in_header is set to 3 while FOUR imports are supplied, so the
    # cut is exercised. pkg.alpha carries FOUR names while only three may show.
    _MAX_IMPORTS = 3
    _NAMES_PER_IMPORT = 3

    def _imports(self) -> list[ImportEdge]:
        return [
            ImportEdge(
                file_id=1, imported_from="pkg.alpha", imported_names=["A1", "A2", "A3", "A4"]
            ),
            ImportEdge(file_id=1, imported_from="pkg.beta", imported_names=["*"]),
            ImportEdge(file_id=1, imported_from="pkg.gamma", imported_names=["G1"]),
            # 4th import: beyond max_imports_in_header, must never be rendered.
            ImportEdge(file_id=1, imported_from="pkg.delta", imported_names=["D1"]),
        ]

    def test_import_header_line_is_exactly_this(self) -> None:
        """Fails under, in chunker.py _build_import_header:
        `imports[: self.config.max_imports_in_header]` -> `... - 1]` (drops
        pkg.gamma) or `... + 1]` (adds pkg.delta);
        `imp.imported_names[:3]` -> `[:2]` (drops A3) or `[:4]` (adds A4);
        `", ".join(seen)` -> `" ".join(seen)`;
        and the `imported_names != ["*"]` wildcard branch being inverted or
        dropped (pkg.beta would gain `.{*}`).
        """
        imports = self._imports()

        # Preconditions: the fixture must actually straddle both bounds,
        # otherwise these assertions would hold for a mutated slice too.
        assert len(imports) > self._MAX_IMPORTS, (
            "FIXTURE NO LONGER DISCRIMINATES: _imports() supplies "
            f"{len(imports)} imports but max_imports_in_header is "
            f"{self._MAX_IMPORTS}; supply more imports than the budget or the "
            "max_imports slice bound is never exercised."
        )
        assert len(imports[0].imported_names) > self._NAMES_PER_IMPORT, (
            "FIXTURE NO LONGER DISCRIMINATES: _imports()[0] carries "
            f"{len(imports[0].imported_names)} names but only "
            f"{self._NAMES_PER_IMPORT} may render; give it more names than "
            "that or the imported_names slice bound is never exercised."
        )

        chunker = Chunker(
            ChunkerConfig(
                include_imports_in_header=True,
                max_imports_in_header=self._MAX_IMPORTS,
            )
        )
        text = _build_one(chunker, _symbol(body="def my_func():\n    return 1"), imports)

        import_lines = [ln for ln in text.splitlines() if ln.startswith("# Imports:")]
        assert len(import_lines) == 1, f"expected one # Imports: line, got {import_lines!r}"

        # Exact literal. Written out by hand -- NOT built from the fixture --
        # so that adding, dropping or reordering any rendered name fails here.
        assert import_lines[0] == "# Imports: pkg.alpha.{A1, A2, A3}, pkg.beta, pkg.gamma.{G1}"

    def test_rendered_module_set_is_exactly_the_first_three(self) -> None:
        """Set equality both ways, so an extra module (max_imports slice `+ 1`)
        and a missing one (`- 1`) both fail.

        Fails under: `imports[: self.config.max_imports_in_header]` ->
        `... - 1]` or `... + 1]` in chunker.py _build_import_header.
        """
        chunker = Chunker(
            ChunkerConfig(
                include_imports_in_header=True,
                max_imports_in_header=self._MAX_IMPORTS,
            )
        )
        text = _build_one(chunker, _symbol(body="def my_func():\n    return 1"), self._imports())

        # Explicit expected table -- deliberately NOT derived by iterating the
        # ImportEdge list that produced the header.
        expected_present = {"pkg.alpha", "pkg.beta", "pkg.gamma"}
        expected_absent = {"pkg.delta"}

        actual_present = {
            m for m in ("pkg.alpha", "pkg.beta", "pkg.gamma", "pkg.delta") if m in text
        }
        assert actual_present == expected_present
        assert expected_present == actual_present
        assert actual_present & expected_absent == set()


# ---------------------------------------------------------------------------
# Exact line structure of chunk_text
# ---------------------------------------------------------------------------


class TestChunkTextLineStructure:
    def test_chunk_text_lines_are_exactly_header_blank_body(self) -> None:
        """The pre-existing suite checks `startswith("# File:")` and
        `body in chunk_text`, both of which stay true if extra blank lines are
        injected between every header line -- inflating the token count of every
        chunk in the index.

        Fails under, in chunker.py:
        `_build_chunk_text` `return "\\n".join(lines)` -> `"\\n\\n".join(lines)`;
        deleting or neutralising `lines.append("")  # blank line between header
        and body`; and `{language.capitalize()}` -> `{language}`.
        """
        chunker = Chunker(ChunkerConfig())
        text = _build_one(chunker, _symbol(body="def my_func():\n    return 1"))

        # Exact literal line list, hand-written.
        assert text.splitlines() == [
            "# File: src/pkg/mod.py | Language: Python",
            "",
            "def my_func():",
            "    return 1",
        ]


# ---------------------------------------------------------------------------
# Docstring de-duplication guard
# ---------------------------------------------------------------------------


class TestDocstringDeduplicationGuard:
    """chunker.py skips the `# Doc:` line when the body already opens with a
    string literal, because a Python docstring is part of the body AST node and
    emitting it twice doubles its weight in the embedding (the module says so).

    The guard is `symbol.body.lstrip().startswith(('""\"', "''\'", '"', "'"))`.
    The pre-existing suite only covers ONE shape: an unindented triple-double
    quote. Both the `lstrip()` and the two single-character quote variants are
    therefore unpinned.
    """

    def test_indented_body_opening_with_docstring_is_not_doubled(self) -> None:
        """A method body arrives indented, so the literal is not at column 0.

        Fails under: chunker.py _build_chunk_text
        `symbol.body.lstrip().startswith(...)` -> `symbol.body.startswith(...)`.
        """
        body = '    """Already documented."""\n    return 1'
        symbol = _symbol(body=body, docstring="Already documented.")

        # Precondition: a truthy docstring is what makes the guard load-bearing.
        # With docstring=None the assertion below would hold by construction.
        assert symbol.docstring, (
            "FIXTURE NO LONGER DISCRIMINATES: _symbol(docstring=...) is falsy, "
            "so the `# Doc:` line is skipped by the docstring check rather than "
            "by the body-opens-with-a-literal guard under test."
        )
        assert body != body.lstrip(), (
            "FIXTURE NO LONGER DISCRIMINATES: this body is not indented, so "
            "removing lstrip() from the guard would not change the outcome."
        )

        chunker = Chunker(ChunkerConfig())
        text = _build_one(chunker, symbol)
        assert "# Doc:" not in text

    def test_single_quoted_body_docstring_is_not_doubled(self) -> None:
        """A single-quoted one-line docstring opens with `'`, not `'''`.

        Fails under: chunker.py _build_chunk_text dropping the `'"'` and `"'"`
        entries from the startswith tuple, leaving only the triple-quoted forms.
        """
        body = "'Already documented.'\nreturn 1"
        symbol = _symbol(body=body, docstring="Already documented.")

        assert symbol.docstring, (
            "FIXTURE NO LONGER DISCRIMINATES: _symbol(docstring=...) is falsy, "
            "so the `# Doc:` line is skipped by the docstring check rather than "
            "by the body-opens-with-a-literal guard under test."
        )
        assert not body.startswith(("'''", '"""')), (
            "FIXTURE NO LONGER DISCRIMINATES: this body opens with a TRIPLE "
            "quote, which the pre-existing tuple entries already cover; the "
            "single-character variants would not be exercised."
        )

        chunker = Chunker(ChunkerConfig())
        text = _build_one(chunker, symbol)
        assert "# Doc:" not in text


# ---------------------------------------------------------------------------
# Parent-class lookup guard
# ---------------------------------------------------------------------------


class TestParentLookupMissingKeyGuard:
    def test_parent_id_absent_from_map_is_skipped_not_a_crash(self) -> None:
        """The indexer builds parent_symbols from one file's symbols, so a
        symbol whose parent_id points outside that set reaches
        `parent_symbols[symbol.parent_id]`. The `symbol.parent_id in
        parent_symbols` membership test is the only thing preventing a KeyError
        that would abort indexing for the whole file.

        Fails under: chunker.py _build_chunk_text dropping
        `and symbol.parent_id in parent_symbols` from the header condition
        (raises KeyError instead of omitting the line).
        """
        chunker = Chunker(ChunkerConfig(include_parent_signature=True))

        # Positive control FIRST: with the parent present the line IS emitted,
        # proving the lookup is live and that the negative case below is not
        # passing merely because the feature is off.
        present_parent = _symbol(
            body="class RealParent: ...", id=10, name="RealParent", kind=SymbolKind.CLASS
        )
        child_found = _symbol(body="return 1", id=11, name="method", parent_id=10)
        found_text = _build_one(chunker, child_found, parent_symbols={10: present_parent})
        assert "# Class: RealParent" in found_text, (
            "PRECONDITION FAILED: the parent-class header is not being emitted "
            "even when the parent IS in parent_symbols, so this test can no "
            "longer distinguish 'guard removed' from 'feature disabled'."
        )

        # Now the missing-key case: parent_id 999 is not a key in the map.
        child_missing = _symbol(body="return 1", id=12, name="orphan", parent_id=999)
        parent_symbols = {10: present_parent}
        assert child_missing.parent_id not in parent_symbols, (
            "FIXTURE NO LONGER DISCRIMINATES: parent_id is present in "
            "parent_symbols, so the membership guard is never exercised."
        )

        # Must not raise, and must simply omit the header line.
        missing_text = _build_one(chunker, child_missing, parent_symbols=parent_symbols)
        assert not any(ln.startswith("# Class:") for ln in missing_text.splitlines())
        assert "RealParent" not in missing_text
