"""
Base parser interface. Every language-specific parser implements this.

NOTE: This file is inlined in Phase 6a so the extractors can run before
Phase 5 (parser-infra) merges into develop. When Phase 5 merges, this
file will be the authoritative version (they are identical).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from trelix.core.models import CallEdge, ImportEdge, Symbol, TypeEdge


@dataclass
class ParseResult:
    """Everything extracted from a single file."""

    symbols: list[Symbol]
    call_edges: list[CallEdge]      # populated only if config.parser.extract_calls
    import_edges: list[ImportEdge]  # populated only if config.parser.extract_imports
    parse_errors: int               # number of Tree-sitter error nodes (0 = clean parse)
    type_edges: list[TypeEdge] = field(default_factory=list)  # inheritance/impl edges


class BaseParser(ABC):
    """
    Abstract parser. Subclasses implement `parse` for their language.

    Each subclass loads a language-specific grammar via tree_sitter_languages
    and walks the parsed AST to extract symbols, edges, and metadata.
    """

    @abstractmethod
    def parse(self, source: str, file_id: int) -> ParseResult:
        """
        Parse source code and extract symbols, call edges, import edges.

        Args:
            source:  raw source code string
            file_id: DB id of the file (used to populate Symbol.file_id)

        Returns:
            ParseResult with all extracted information
        """
        ...

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Tree-sitter language name, e.g. 'python', 'javascript'."""
        ...
