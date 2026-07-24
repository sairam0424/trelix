"""
Grammar loading helpers, backed by tree-sitter-language-pack.

Single chokepoint for every language extractor under
src/trelix/indexing/parser/extractors/ — swap the backing package here and
nothing else needs to change.

Unlike the previous tree_sitter_languages dependency (which bundled every
compiled grammar in its wheel), tree-sitter-language-pack ships only its
native loader and fetches each language's compiled grammar from the network
on first use, caching it under a local directory (see PackConfig.cache_dir /
tree_sitter_language_pack.cache_dir()). Call prefetch_all() once during image
build / CI setup / first install to warm that cache so indexing itself never
needs network access.
"""

from __future__ import annotations

from tree_sitter import Language, Parser
from tree_sitter_language_pack import get_language, get_parser, prefetch

# Every language name trelix's extractors pass to load_language()/make_parser(),
# using tree-sitter-language-pack's naming (not tree_sitter_languages' — note
# "csharp", not "c_sharp").
PREFETCH_LANGUAGES: tuple[str, ...] = (
    "c",
    "cpp",
    "csharp",
    "css",
    "go",
    "html",
    "java",
    "javascript",
    "json",
    "kotlin",
    "python",
    "ruby",
    "rust",
    "toml",
    "tsx",
    "typescript",
)


def load_language(name: str) -> Language:
    """Return a tree-sitter Language for *name*."""
    return get_language(name)


def make_parser(name: str) -> Parser:
    """Return a Parser pre-loaded with the named grammar."""
    return get_parser(name)


def prefetch_all() -> None:
    """Download every grammar trelix uses, so later parsing is fully offline."""
    prefetch(list(PREFETCH_LANGUAGES))
