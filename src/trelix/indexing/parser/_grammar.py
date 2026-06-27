"""
Grammar loading helpers for tree-sitter 0.21.x.

tree_sitter_languages.get_language() calls Language(path, name) internally,
which fires a FutureWarning on tree-sitter 0.21.3. We suppress it at the call
site so neither library callers nor the test suite see the noise.

When upgrading to tree-sitter 0.22 this is the single place to update:
  1. Delete this file.
  2. Add per-language grammar packages (tree-sitter-python, etc.).
  3. Update each extractor to: Language(tree_sitter_<lang>.language()).
  4. Remove the filterwarnings entry from pyproject.toml.
"""

from __future__ import annotations

import warnings

from tree_sitter import Language, Parser


def load_language(name: str) -> Language:
    """Return a tree-sitter Language for *name*, suppressing 0.21.x FutureWarning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="tree_sitter",
        )
        from tree_sitter_languages import get_language

        return get_language(name)  # type: ignore[no-any-return]


def make_parser(name: str) -> Parser:
    """Return a Parser pre-loaded with the named grammar."""
    lang = load_language(name)
    parser = Parser()
    parser.set_language(lang)
    return parser
