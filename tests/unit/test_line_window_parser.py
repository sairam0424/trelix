"""Tests for LineWindowParser and the filename-aware language detector.

Both exist to fix the same failure: a file that is "indexed" but unreachable. Chunks in
this schema hang off `symbol_id`, so a file yielding no symbols yields no chunks, and
every retrieval leg — vector, BM25, grep, summary, sub-chunk — is blind to it.

Measured on this repository before the change: zero shell scripts, zero Dockerfiles and
zero Makefiles were indexed at all, and 12 non-empty indexed files totalling 17 KB had no
symbols.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.models import Language, SymbolKind
from trelix.indexing.parser.extractors.line_window import LineWindowParser
from trelix.indexing.walker import FILENAME_MAP, detect_language


class TestFilenameAwareDetection:
    """An extensionless artifact must resolve to a language.

    `Path("Dockerfile").suffix` is `""`, so no EXTENSION_MAP entry could ever match one.
    That is why a Dockerfile and a Makefile were absent from the index entirely rather
    than merely ranked badly — the walker's language filter dropped them before any
    parser was consulted.
    """

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Dockerfile", Language.DOCKERFILE),
            ("dockerfile", Language.DOCKERFILE),
            ("Containerfile", Language.DOCKERFILE),
            ("Dockerfile.prod", Language.DOCKERFILE),
            ("Makefile", Language.MAKE),
            ("GNUmakefile", Language.MAKE),
            ("Makefile.local", Language.MAKE),
        ],
    )
    def test_filename_only_artifacts_resolve(self, name: str, expected: Language) -> None:
        assert detect_language(Path(name)) is expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("build.sh", Language.SHELL),
            ("hooks.bash", Language.SHELL),
            ("migrate.sql", Language.SQL),
            ("service.proto", Language.PROTO),
            ("rules.mk", Language.MAKE),
        ],
    )
    def test_new_extensions_resolve(self, name: str, expected: Language) -> None:
        assert detect_language(Path(name)) is expected

    def test_existing_detection_is_unchanged(self) -> None:
        """The filename map must not shadow anything that already worked."""
        for name, expected in (
            ("main.py", Language.PYTHON),
            ("app.tsx", Language.TSX),
            ("README.md", Language.MARKDOWN),
            ("config.yaml", Language.YAML),
        ):
            assert detect_language(Path(name)) is expected, name

    def test_an_unknown_artifact_is_still_unknown(self) -> None:
        assert detect_language(Path("mystery.xyz")) is Language.UNKNOWN

    def test_the_filename_map_is_lowercase_keyed(self) -> None:
        """Lookup lowercases the name, so a capitalised key would never match."""
        assert all(key == key.lower() for key in FILENAME_MAP)


class TestLineWindowParser:
    """Windows, not one symbol per file.

    `Chunker` TRUNCATES a chunk over its token budget rather than splitting it, so a
    single symbol spanning a 200-line script would be cut off at the budget and its tail
    would be exactly as unreachable as before.
    """

    def test_splits_into_windows_with_honest_line_numbers(self) -> None:
        parser = LineWindowParser(window_lines=3)
        result = parser.parse("\n".join(f"line {i}" for i in range(1, 8)), file_id=1)

        spans = [(s.line_start, s.line_end) for s in result.symbols]
        assert spans == [(1, 3), (4, 6), (7, 7)], (
            "line numbers must be 1-indexed and inclusive, like every other extractor, "
            f"so a citation points where a reader would look; got {spans}"
        )

    def test_emits_section_symbols(self) -> None:
        result = LineWindowParser().parse("echo hello\n", file_id=1)
        assert all(s.kind is SymbolKind.SECTION for s in result.symbols)

    def test_bodies_reconstruct_the_source(self) -> None:
        """Nothing may be dropped between windows — a gap is unreachable text."""
        source = "\n".join(f"line {i}" for i in range(1, 11))
        result = LineWindowParser(window_lines=4).parse(source, file_id=1)
        assert "\n".join(s.body for s in result.symbols) == source

    def test_a_blank_file_yields_nothing(self) -> None:
        """A whitespace-only window would cost a chunk and an embedding to match nothing."""
        assert LineWindowParser().parse("\n\n   \n\t\n", file_id=1).symbols == []

    def test_an_empty_file_yields_nothing(self) -> None:
        assert LineWindowParser().parse("", file_id=1).symbols == []

    def test_a_long_line_file_shrinks_the_window(self) -> None:
        """A minified or single-line file blows the char ceiling inside the line budget."""
        parser = LineWindowParser(window_lines=10, max_window_chars=100)
        source = "\n".join("x" * 60 for _ in range(10))
        result = parser.parse(source, file_id=1)

        assert len(result.symbols) > 1, "the window did not shrink for long lines"
        assert all(len(s.body) <= 100 or s.line_start == s.line_end for s in result.symbols)

    def test_a_single_unsplittable_line_still_produces_a_symbol(self) -> None:
        """One 5,000-char line cannot be split, and dropping it would lose the file."""
        result = LineWindowParser(max_window_chars=100).parse("x" * 5000, file_id=1)
        assert len(result.symbols) == 1

    def test_it_claims_no_structural_extraction(self) -> None:
        """It reads no syntax, so it must not pretend to produce edges."""
        result = LineWindowParser().parse("FROM python:3.11\nRUN pip install .\n", file_id=1)
        assert result.call_edges == []
        assert result.import_edges == []
        assert result.parse_errors == 0

    def test_the_language_name_says_what_it_is(self) -> None:
        """A caller inspecting the parser should be able to tell this was a line split."""
        assert LineWindowParser().language_name == "line-window"

    def test_file_id_is_threaded_through(self) -> None:
        result = LineWindowParser().parse("echo hi\n", file_id=77)
        assert all(s.file_id == 77 for s in result.symbols)


class TestRegistryRoutesOpsLanguages:
    """The five ops languages must resolve to a parser, or the files stay unreachable."""

    @pytest.mark.parametrize(
        "language",
        [Language.SHELL, Language.DOCKERFILE, Language.MAKE, Language.SQL, Language.PROTO],
    )
    def test_each_ops_language_gets_the_fallback(self, language: Language) -> None:
        from trelix.indexing.parser.registry import get_parser

        assert isinstance(get_parser(language), LineWindowParser)

    def test_a_real_extractor_is_not_replaced(self) -> None:
        """The fallback must not shadow a language that has a structural extractor."""
        from trelix.indexing.parser.registry import get_parser

        parser = get_parser(Language.PYTHON)
        assert parser is not None
        assert not isinstance(parser, LineWindowParser)

    def test_unknown_still_has_no_parser(self) -> None:
        from trelix.indexing.parser.registry import get_parser

        assert get_parser(Language.UNKNOWN) is None


class TestLineWindowLanguagesUseTheFusionFallbackWeight:
    """Tripwire, not a correctness assertion.

    `fusion.py` looks up `weights.get(str(lang), 1.0)`, so a language with no entry in
    `file_type_weights` is silently ranked level with parsed Python. The five line-window
    languages are in exactly that position, and that is a KNOWN open question rather than
    a settled decision — see the NOTE in `RetrievalConfig.file_type_weights`.

    It is left at the fallback because it could not be tuned honestly: down-weighting to
    0.4-0.6 and to 0.8 each dropped ops files out of the top 10 for queries specifically
    about them, and target rank moved non-monotonically with the multiplier (rank 9 at
    1.0, absent at 0.8, rank 2 at 0.4). Four ops queries cannot support fitting a scalar.

    These tests fail the moment someone adds a weight. That is the intent: the failure
    routes them to the analysis above so the value is chosen against a bigger golden set
    rather than re-fitted to noise. Deleting these tests along with the change is a fine
    outcome — silently landing a tuned scalar is not.
    """

    def _fallback_languages(self) -> list[Language]:
        from trelix.indexing.parser.registry import get_parser

        return [
            lang
            for lang in Language
            if lang is not Language.UNKNOWN and isinstance(get_parser(lang), LineWindowParser)
        ]

    def test_the_five_ops_languages_route_to_the_fallback_parser(self) -> None:
        """Guards the tests below: if routing broke they would pass vacuously."""
        assert {
            Language.SHELL,
            Language.DOCKERFILE,
            Language.MAKE,
            Language.SQL,
            Language.PROTO,
        } <= set(self._fallback_languages())

    def test_they_are_left_unweighted_on_purpose(self) -> None:
        from trelix.core.config import RetrievalConfig

        weights = RetrievalConfig().file_type_weights
        weighted = {
            str(lang): weights[str(lang)]
            for lang in self._fallback_languages()
            if str(lang) in weights
        }
        assert not weighted, (
            f"{weighted} now carry explicit weights. Read the NOTE in "
            "RetrievalConfig.file_type_weights before keeping this: the value needs a "
            "~20-query ops golden set to justify, and 0.4-0.8 were each measured to cost "
            "ops-query reachability. If you have that evidence, update it and delete "
            "this test."
        )

    # The other half of this invariant — that an unweighted language actually receives
    # full weight rather than being dropped — is pinned behaviourally in
    # test_fusion.py::test_a_language_absent_from_weights_keeps_full_weight, next to the
    # fusion helpers rather than duplicated here.
