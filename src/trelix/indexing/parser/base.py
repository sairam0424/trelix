"""
Base parser interface. Every language-specific parser implements this.

Parsers use Tree-sitter AST traversal to extract symbols, call edges,
import edges, and type edges from source files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from trelix.core.models import CallEdge, ImportEdge, Symbol, TypeEdge


@dataclass
class ParseResult:
    """Everything extracted from a single file."""
    symbols: list[Symbol]
    call_edges: list[CallEdge]       # populated when extract_calls is enabled
    import_edges: list[ImportEdge]   # populated when extract_imports is enabled
    parse_errors: int                # number of Tree-sitter ERROR nodes (0 = clean)
    type_edges: list[TypeEdge] = field(default_factory=list)  # inheritance/impl edges


class BaseParser(ABC):
    """
    Abstract parser. Subclasses implement `parse` for their language.

    Each subclass loads a Tree-sitter grammar and uses direct AST traversal
    to extract symbols and relationships.
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
