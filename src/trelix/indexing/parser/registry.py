"""
Parser registry: maps Language enum values to their parser instances.

Design: parsers are instantiated once and reused (Tree-sitter parsers
are stateful but cheap to keep alive per language).

Only PythonParser is registered in Phase 5. Additional parsers (JavaScript,
TypeScript, Go, Rust, etc.) will be added in later phases.
"""

from __future__ import annotations

from functools import lru_cache

from trelix.core.models import Language

from .base import BaseParser


@lru_cache(maxsize=None)
def get_parser(language: Language) -> BaseParser | None:
    """
    Return the parser for a given language, or None if unsupported.
    Lazily imported to avoid loading all tree-sitter grammars at startup.
    """
    match language:
        case Language.PYTHON:
            from .extractors.python import PythonParser
            return PythonParser()
        case Language.JAVASCRIPT:
            try:
                from .extractors.javascript import JavaScriptParser  # type: ignore[import]
                return JavaScriptParser()
            except ImportError:
                return None
        case Language.TYPESCRIPT | Language.TSX:
            try:
                from .extractors.typescript import TypeScriptParser  # type: ignore[import]
                return TypeScriptParser(tsx=(language == Language.TSX))
            except ImportError:
                return None
        case Language.GO:
            try:
                from .extractors.go import GoParser  # type: ignore[import]
                return GoParser()
            except ImportError:
                return None
        case Language.RUST:
            try:
                from .extractors.rust import RustParser  # type: ignore[import]
                return RustParser()
            except ImportError:
                return None
        case Language.JAVA:
            try:
                from .extractors.java import JavaParser  # type: ignore[import]
                return JavaParser()
            except ImportError:
                return None
        case Language.KOTLIN:
            try:
                from .extractors.kotlin import KotlinParser  # type: ignore[import]
                return KotlinParser()
            except ImportError:
                return None
        case Language.CPP:
            try:
                from .extractors.cpp import CppParser  # type: ignore[import]
                return CppParser()
            except ImportError:
                return None
        case Language.C:
            try:
                from .extractors.c import CParser  # type: ignore[import]
                return CParser()
            except ImportError:
                return None
        case Language.CSHARP:
            try:
                from .extractors.csharp import CSharpParser  # type: ignore[import]
                return CSharpParser()
            except ImportError:
                return None
        case _:
            return None
