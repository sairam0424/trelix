"""
Parser registry: maps Language enum values to their parser instances.

Design: parsers are instantiated once and reused (Tree-sitter parsers
are stateful but cheap to keep alive per language).

Each entry uses try/except ImportError so that a missing grammar wheel
(tree-sitter-languages) doesn't break the whole registry — the caller
receives None and the file is silently skipped.
"""

from __future__ import annotations

from functools import cache

from trelix.core.models import Language

from .base import BaseParser


@cache
def get_parser(language: Language) -> BaseParser | None:
    """
    Return the parser for a given language, or None if unsupported.
    Lazily imported to avoid loading all tree-sitter grammars at startup.
    """
    match language:
        case Language.PYTHON:
            try:
                from .extractors.python import PythonParser

                return PythonParser()
            except ImportError:
                return None

        case Language.JAVASCRIPT:
            try:
                from .extractors.javascript import JavaScriptParser

                return JavaScriptParser()
            except ImportError:
                return None

        case Language.TYPESCRIPT | Language.TSX:
            try:
                from .extractors.typescript import TypeScriptParser

                return TypeScriptParser(tsx=(language == Language.TSX))
            except ImportError:
                return None

        case Language.GO:
            try:
                from .extractors.go import GoParser

                return GoParser()
            except ImportError:
                return None

        case Language.RUST:
            try:
                from .extractors.rust import RustParser

                return RustParser()
            except ImportError:
                return None

        case Language.JAVA:
            try:
                from .extractors.java import JavaParser

                return JavaParser()
            except ImportError:
                return None

        case Language.KOTLIN:
            try:
                from .extractors.kotlin import KotlinParser

                return KotlinParser()
            except ImportError:
                return None

        case Language.CPP:
            try:
                from .extractors.cpp import CppParser

                return CppParser()
            except ImportError:
                return None

        case Language.C:
            try:
                from .extractors.c import CParser

                return CParser()
            except ImportError:
                return None

        case Language.CSHARP:
            try:
                from .extractors.csharp import CSharpParser

                return CSharpParser()
            except ImportError:
                return None

        case Language.RAZOR:
            try:
                from .extractors.razor import RazorParser

                return RazorParser()
            except ImportError:
                return None

        case Language.CSHTML:
            try:
                from .extractors.cshtml import CshtmlParser

                return CshtmlParser()
            except ImportError:
                return None

        case Language.CSPROJ:
            try:
                from .extractors.csproj import CsprojParser

                return CsprojParser()
            except ImportError:
                return None

        case Language.MARKDOWN:
            try:
                from .extractors.markdown import MarkdownParser

                return MarkdownParser()
            except ImportError:
                return None

        case Language.JSON:
            try:
                from .extractors.json_config import JsonParser

                return JsonParser()
            except ImportError:
                return None

        case Language.YAML:
            try:
                from .extractors.yaml_config import YamlParser

                return YamlParser()
            except ImportError:
                return None

        case Language.TOML:
            try:
                from .extractors.toml_config import TomlParser

                return TomlParser()
            except ImportError:
                return None

        case Language.HTML:
            try:
                from .extractors.html import HtmlParser

                return HtmlParser()
            except ImportError:
                return None

        case Language.CSS:
            try:
                from .extractors.css import CssParser

                return CssParser()
            except ImportError:
                return None

        case Language.RUBY:
            try:
                from .extractors.ruby import RubyParser

                return RubyParser()
            except ImportError:
                return None

        case _:
            return None
