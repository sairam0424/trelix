"""
Multi-granularity sub-symbol indexing.

Indexes code at three levels:
  - FUNCTION: entire function body (existing, in chunks table)
  - BLOCK: logical code blocks (if/for/while/with bodies, class methods)
  - STATEMENT: individual assignment/return/call statements

Research basis: MGS3 (arXiv:2505.24274, KDD 2025, 2-1 vote) -- multi-granularity
improves recall on block-level queries where function-level retrieval is too coarse.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.models import Symbol

logger = logging.getLogger("trelix.indexing.multi_granularity")


class Granularity(StrEnum):
    FUNCTION = "function"
    BLOCK = "block"
    STATEMENT = "statement"


@dataclass
class SubSymbolChunk:
    """A sub-symbol chunk at block or statement granularity."""

    parent_symbol_id: int
    granularity: Granularity
    chunk_text: str
    line_start: int
    line_end: int
    token_count: int
    id: int | None = field(default=None)


# Node types that delimit logical blocks in Python
_BLOCK_NODE_TYPES: frozenset[str] = frozenset([
    "if_statement", "for_statement", "while_statement",
    "with_statement", "try_statement", "match_statement",
    "function_definition", "class_definition",
])

# Node types for individual statements
_STATEMENT_NODE_TYPES: frozenset[str] = frozenset([
    "expression_statement", "assignment", "augmented_assignment",
    "return_statement", "raise_statement", "assert_statement",
    "delete_statement", "import_statement", "import_from_statement",
])


class MultiGranularityChunker:
    """
    Extracts sub-symbol chunks at block and statement granularity.

    Uses tree-sitter to walk the AST and identify logical code blocks
    and individual statements within a function body.
    """

    def extract_sub_chunks(
        self,
        symbol: Symbol,
        granularities: list[Granularity] | None = None,
    ) -> list[SubSymbolChunk]:
        """
        Extract sub-symbol chunks from a symbol's body.

        Args:
            symbol: The symbol to extract from
            granularities: Which granularities to extract (default: [BLOCK, STATEMENT])

        Returns:
            list[SubSymbolChunk], empty list on any failure
        """
        if not symbol.body or not symbol.id:
            return []

        targets = set(granularities or [Granularity.BLOCK, Granularity.STATEMENT])

        try:
            return self._extract_tree_sitter(symbol, targets)
        except Exception as exc:
            logger.debug(
                "MultiGranularityChunker failed for %s: %s",
                symbol.qualified_name, exc,
            )
            return []

    def _extract_tree_sitter(
        self,
        symbol: Symbol,
        targets: set[Granularity],
    ) -> list[SubSymbolChunk]:
        """Walk tree-sitter AST to find blocks and statements."""
        try:
            from tree_sitter_languages import get_parser as get_ts_parser
        except ImportError:
            return []

        body = symbol.body.encode("utf-8")
        try:
            parser = get_ts_parser("python")
            tree = parser.parse(body)
        except Exception:
            return []

        chunks: list[SubSymbolChunk] = []
        lines = symbol.body.splitlines()
        base_line = symbol.line_start

        def walk(node: object) -> None:
            node_type = getattr(node, "type", "")

            if Granularity.BLOCK in targets and node_type in _BLOCK_NODE_TYPES:
                start_line = node.start_point[0]  # type: ignore[union-attr]
                end_line = node.end_point[0]  # type: ignore[union-attr]
                # Skip trivially small blocks (<=2 lines)
                if end_line - start_line >= 2:
                    block_lines = lines[start_line:end_line + 1]
                    text = "\n".join(block_lines).strip()
                    if text:
                        chunks.append(SubSymbolChunk(
                            parent_symbol_id=int(symbol.id),  # type: ignore[arg-type]
                            granularity=Granularity.BLOCK,
                            chunk_text=text,
                            line_start=base_line + start_line,
                            line_end=base_line + end_line,
                            token_count=len(text.split()),
                        ))

            elif Granularity.STATEMENT in targets and node_type in _STATEMENT_NODE_TYPES:
                start_line = node.start_point[0]  # type: ignore[union-attr]
                end_line = node.end_point[0]  # type: ignore[union-attr]
                stmt_lines = lines[start_line:end_line + 1]
                text = "\n".join(stmt_lines).strip()
                # Skip trivially short statements (<=2 tokens)
                if text and len(text.split()) > 2:
                    chunks.append(SubSymbolChunk(
                        parent_symbol_id=int(symbol.id),  # type: ignore[arg-type]
                        granularity=Granularity.STATEMENT,
                        chunk_text=text,
                        line_start=base_line + start_line,
                        line_end=base_line + end_line,
                        token_count=len(text.split()),
                    ))

            for child in getattr(node, "children", []):
                walk(child)

        walk(tree.root_node)
        return chunks
