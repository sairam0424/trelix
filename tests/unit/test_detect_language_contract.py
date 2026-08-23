"""Contract tests pinning every language-detection entry in trelix.indexing.walker.

MUTATIONS these tests must fail on (each was a MEASURED survivor of the full suite
before this file existed — deleting `.rb`, `.kt`, `.kts`, `.cs`, `.cpp`, `.cc`, `.cxx`,
`.hpp`, `.jsx`, `.mjs`, `.cjs`, `.scss`, `.sass` or `.less` from `EXTENSION_MAP` broke
NO test, i.e. Ruby, Kotlin, C#, C++, JSX and SCSS could be made invisible to the
indexer in silence):

  * deleting ANY entry from `walker.EXTENSION_MAP`
  * deleting ANY entry from `walker.FILENAME_MAP`
  * repointing an entry at a different Language (e.g. `".scss": Language.SASS`,
    `".cc": Language.C`, `".mjs": Language.TYPESCRIPT`)
  * renaming a `Language` member's VALUE (the wire/DB string), e.g. `RUBY = "rb"`
  * dropping the case-fold in `detect_language` (`path.name.lower()` /
    `path.suffix.lower()`)
  * dropping the stem fallback in `detect_language` (which is what makes
    `Dockerfile.prod` and `Makefile.local` resolve)
  * swapping the FILENAME_MAP lookup after the EXTENSION_MAP lookup
  * removing a Language from `WalkerConfig.languages`, which lets the extension
    resolve and then be dropped by the walk's allow-list (test_walk_yields_*)

Why the tables below are written out by hand instead of being read from the module:
an assertion that loops over `EXTENSION_MAP` checks LESS the moment an entry is
deleted, and still passes. The expected tables here are literals, so a deletion fails
an assertion and an ADDITION fails the set-equality check until someone records the
new extension's expectation. That is what makes this a contract and not a snapshot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.config import IndexConfig, WalkerConfig
from trelix.core.models import Language
from trelix.indexing.walker import EXTENSION_MAP, FILENAME_MAP, FileWalker, detect_language

# ---------------------------------------------------------------------------
# The contract. Hand-written; expected values are LITERAL Language strings, never
# read back out of walker.py or out of the Language enum.
# ---------------------------------------------------------------------------

# (file extension, expected Language value)
EXPECTED_EXTENSIONS: tuple[tuple[str, str], ...] = (
    (".py", "python"),
    (".js", "javascript"),
    (".mjs", "javascript"),
    (".cjs", "javascript"),
    (".jsx", "javascript"),
    (".ts", "typescript"),
    (".tsx", "tsx"),
    (".go", "go"),
    (".rs", "rust"),
    (".java", "java"),
    (".kt", "kotlin"),
    (".kts", "kotlin"),
    (".rb", "ruby"),
    (".cpp", "cpp"),
    (".cc", "cpp"),
    (".cxx", "cpp"),
    (".hpp", "cpp"),
    (".c", "c"),
    (".h", "c"),
    (".cs", "csharp"),
    (".razor", "razor"),
    (".cshtml", "cshtml"),
    (".csproj", "csproj"),
    (".fsproj", "csproj"),
    (".vbproj", "csproj"),
    (".md", "markdown"),
    (".mdx", "markdown"),
    (".json", "json"),
    (".yaml", "yaml"),
    (".yml", "yaml"),
    (".toml", "toml"),
    (".html", "html"),
    (".htm", "html"),
    (".css", "css"),
    (".scss", "css"),
    (".sass", "css"),
    (".less", "css"),
    (".sh", "shell"),
    (".bash", "shell"),
    (".zsh", "shell"),
    (".sql", "sql"),
    (".proto", "proto"),
    (".mk", "make"),
)

# (on-disk filename, expected Language value). These have no usable suffix —
# `Path("Dockerfile").suffix` is "" — so no EXTENSION_MAP entry can ever reach them.
EXPECTED_FILENAMES: tuple[tuple[str, str], ...] = (
    ("Dockerfile", "dockerfile"),
    ("Containerfile", "dockerfile"),
    ("Makefile", "make"),
    ("GNUmakefile", "make"),
)

# An extension deliberately absent from the table above, used as the discrimination
# control: it proves `detect_language` does not answer some non-UNKNOWN value for
# everything handed to it, which is the only way the assertions above could pass by
# construction.
UNMAPPED_EXTENSION = ".trelix-not-a-language"


class TestExtensionMapContract:
    def test_expected_table_is_exactly_the_extension_map(self) -> None:
        """MUTATION: delete or add any EXTENSION_MAP key (e.g. drop `".rb"`).

        Set equality, not just a count: deleting `.rb` while adding `.foo` keeps the
        length identical and must still fail.
        """
        expected_keys = {ext for ext, _ in EXPECTED_EXTENSIONS}
        assert len(expected_keys) == 43, "expected table has duplicate extensions"
        assert set(EXTENSION_MAP) == expected_keys, (
            "EXTENSION_MAP no longer matches the recorded contract. Missing from the "
            f"map: {sorted(expected_keys - set(EXTENSION_MAP))}; present in the map but "
            f"not recorded here: {sorted(set(EXTENSION_MAP) - expected_keys)}. Record the "
            "expectation in EXPECTED_EXTENSIONS (extension -> Language value) rather than "
            "loosening this assertion."
        )

    def test_expected_table_covers_many_distinct_languages(self) -> None:
        """Discrimination precondition, not a behaviour assertion.

        If this table ever collapsed toward one language, the per-extension tests
        below would stop being able to catch a mis-pointed entry.
        """
        assert len({value for _, value in EXPECTED_EXTENSIONS}) == 25

    @pytest.mark.parametrize(("extension", "expected"), EXPECTED_EXTENSIONS)
    def test_extension_detects_as_language(self, extension: str, expected: str) -> None:
        """MUTATION: delete `".rb": Language.RUBY` (or repoint it at another Language).

        Goes through `detect_language`, the single detector every call site uses, so a
        dict entry that exists but is unreachable also fails here.
        """
        resolved = detect_language(Path(f"sample{extension}"))
        assert resolved == expected, f"sample{extension} detected as {resolved!r}"
        assert isinstance(resolved, Language), (
            f"detect_language returned {type(resolved).__name__}, not a Language"
        )

    @pytest.mark.parametrize(("extension", "expected"), EXPECTED_EXTENSIONS)
    def test_uppercase_extension_detects_as_language(self, extension: str, expected: str) -> None:
        """MUTATION: drop `.lower()` from `EXTENSION_MAP.get(path.suffix.lower(), ...)`."""
        assert detect_language(Path(f"SAMPLE{extension.upper()}")) == expected

    def test_unmapped_extension_is_unknown(self) -> None:
        """Control for the two tests above: they would pass by construction if
        `detect_language` returned a non-UNKNOWN value for every input.

        MUTATION: change the `EXTENSION_MAP.get` default from `Language.UNKNOWN`.
        """
        assert UNMAPPED_EXTENSION not in {ext for ext, _ in EXPECTED_EXTENSIONS}
        assert detect_language(Path(f"sample{UNMAPPED_EXTENSION}")) == "unknown"


class TestFilenameMapContract:
    def test_expected_table_is_exactly_the_filename_map(self) -> None:
        """MUTATION: delete or add any FILENAME_MAP key (e.g. drop `"containerfile"`).

        FILENAME_MAP keys are stored lowercased; the table above holds the on-disk
        casing, so the comparison lowercases the table (never the map).
        """
        expected_keys = {name.lower() for name, _ in EXPECTED_FILENAMES}
        assert len(expected_keys) == 4, "expected table has duplicate filenames"
        assert set(FILENAME_MAP) == expected_keys, (
            "FILENAME_MAP no longer matches the recorded contract. Missing from the "
            f"map: {sorted(expected_keys - set(FILENAME_MAP))}; present in the map but "
            f"not recorded here: {sorted(set(FILENAME_MAP) - expected_keys)}."
        )

    @pytest.mark.parametrize(("filename", "expected"), EXPECTED_FILENAMES)
    def test_filename_detects_as_language(self, filename: str, expected: str) -> None:
        """MUTATION: delete `"containerfile": Language.DOCKERFILE`, or drop the
        `path.name.lower()` case-fold that lets `Dockerfile` match the lowercased key.
        """
        resolved = detect_language(Path(filename))
        assert resolved == expected, f"{filename} detected as {resolved!r}"
        assert isinstance(resolved, Language)

    @pytest.mark.parametrize(("filename", "expected"), EXPECTED_FILENAMES)
    def test_filename_with_suffix_detects_via_stem(self, filename: str, expected: str) -> None:
        """MUTATION: delete the `stem = name.split(".", 1)[0]` fallback in
        `detect_language`, which is what resolves `Dockerfile.prod` / `Makefile.local`.
        """
        assert detect_language(Path(f"{filename}.prod")) == expected

    def test_filename_wins_over_extension(self) -> None:
        """MUTATION: move the `EXTENSION_MAP` lookup ahead of the `FILENAME_MAP` lookups.

        `Makefile.json` must be a Makefile, not JSON: `.json` is a real EXTENSION_MAP
        entry, so only lookup ORDER decides this one.
        """
        assert detect_language(Path("Makefile.json")) == "make"


class TestWalkYieldsEveryMappedLanguage:
    """The behaviour the maps exist for: does the file reach the index at all?

    `EXTENSION_MAP` alone is not sufficient — `WalkerConfig.languages` is an
    allow-list the walk filters against, so an extension can resolve correctly and
    still be dropped. These tests assert on what `FileWalker.walk()` actually yields,
    under the SHIPPED default config, rather than on a dict lookup.
    """

    @staticmethod
    def _build_repo(base: Path) -> dict[str, str]:
        """One file per contract entry at the repo root. Returns {filename: language}."""
        expected: dict[str, str] = {}
        for extension, language in EXPECTED_EXTENSIONS:
            name = f"sample{extension}"
            (base / name).write_text(f"content for {name}\n", encoding="utf-8")
            expected[name] = language
        for filename, language in EXPECTED_FILENAMES:
            (base / filename).write_text(f"content for {filename}\n", encoding="utf-8")
            expected[filename] = language
        return expected

    def test_every_mapped_file_is_yielded_with_its_language(self, tmp_path: Path) -> None:
        """MUTATION: delete `".rb": Language.RUBY` from EXTENSION_MAP, or delete
        `Language.RUBY` from `WalkerConfig.languages` — either makes `sample.rb`
        vanish from the walk.
        """
        expected = self._build_repo(tmp_path)
        # Shipped defaults on purpose: this is the config a real `trelix index` uses.
        config = IndexConfig(repo_path=str(tmp_path), walker=WalkerConfig())

        walker = FileWalker(config)
        found = {indexed.rel_path: str(indexed.language) for indexed in walker.walk()}

        assert walker.walk_was_complete, f"walk was truncated: {walker.incomplete_paths}"
        mislabelled = {
            name: (found[name], expected[name])
            for name in sorted(set(found) & set(expected))
            if found[name] != expected[name]
        }
        assert found == expected, (
            "walk() did not yield exactly the contract files. Not indexed at all: "
            f"{sorted(set(expected) - set(found))}; wrong language (got, want): "
            f"{mislabelled}; unexpected extra: {sorted(set(found) - set(expected))}"
        )

    def test_unmapped_file_is_not_yielded(self, tmp_path: Path) -> None:
        """Control for the test above: it would pass by construction if the walk
        yielded every file it saw regardless of detected language.
        """
        (tmp_path / f"sample{UNMAPPED_EXTENSION}").write_text("x\n", encoding="utf-8")
        (tmp_path / "sample.rb").write_text("x\n", encoding="utf-8")
        config = IndexConfig(repo_path=str(tmp_path), walker=WalkerConfig())

        found = {indexed.rel_path for indexed in FileWalker(config).walk()}

        assert found == {"sample.rb"}, f"walk yielded {sorted(found)}"
