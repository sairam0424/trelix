"""
Chunker: converts Symbol objects into embeddable Chunk objects.

The key insight stolen from LlamaIndex's CodeHierarchyNodeParser:
each chunk gets a "context header" that includes the file path, language,
relevant imports, and parent class — so the embedding model understands
the symbol WITHOUT needing to see the whole file.

Example chunk_text:
    # File: src/auth/login.py | Language: Python
    # Imports: django.contrib.auth, .models.User
    # Class: LoginView

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        \"\"\"Authenticate user credentials.\"\"\"
        ...
"""

from __future__ import annotations

import tiktoken

from trelix.core.config import ChunkerConfig
from trelix.core.models import Chunk, ImportEdge, Symbol, SymbolKind


class Chunker:
    """
    Converts symbols into Chunk objects with context headers.

    Usage:
        chunker = Chunker(config.chunker)
        chunks = chunker.build_chunks(symbols, imports, file_rel_path, language)
    """

    def __init__(self, config: ChunkerConfig) -> None:
        self.config = config
        # cl100k_base works for most modern models (GPT-4, Claude via approximation)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def build_chunks(
        self,
        symbols: list[Symbol],
        imports: list[ImportEdge],
        file_rel_path: str,
        language: str,
        parent_symbols: dict[int, Symbol] | None = None,  # id → Symbol map for class lookup
    ) -> list[Chunk]:
        """
        Build one Chunk per symbol. Large symbols are split if over token budget.
        """
        if parent_symbols is None:
            parent_symbols = {}

        # Build a short import header string (top N imports)
        import_header = self._build_import_header(imports)

        chunks: list[Chunk] = []
        for symbol in symbols:
            chunk_text = self._build_chunk_text(
                symbol=symbol,
                file_rel_path=file_rel_path,
                language=language,
                import_header=import_header,
                parent_symbols=parent_symbols,
            )
            token_count = len(self._tokenizer.encode(chunk_text))

            # If chunk exceeds budget, truncate body (keep header + signature + docstring)
            if token_count > self.config.max_tokens_per_chunk:
                chunk_text = self._truncate_chunk(chunk_text, self.config.max_tokens_per_chunk)
                token_count = self.config.max_tokens_per_chunk

            chunks.append(Chunk(
                symbol_id=symbol.id or 0,
                chunk_text=chunk_text,
                token_count=token_count,
            ))

        return chunks

    def _build_chunk_text(
        self,
        symbol: Symbol,
        file_rel_path: str,
        language: str,
        import_header: str,
        parent_symbols: dict[int, Symbol],
    ) -> str:
        lines: list[str] = []

        # --- Context header ---
        lines.append(f"# File: {file_rel_path} | Language: {language.capitalize()}")

        if import_header and self.config.include_imports_in_header:
            lines.append(f"# Imports: {import_header}")

        # Parent class context (for methods)
        if self.config.include_parent_signature and symbol.parent_id and symbol.parent_id in parent_symbols:
            parent = parent_symbols[symbol.parent_id]
            lines.append(f"# {parent.kind.value.capitalize()}: {parent.name}")

        lines.append("")   # blank line between header and body

        # Docstring — surfaced before body so it survives truncation.
        # Skip if the body already starts with a string literal (Python docstrings
        # are part of the body AST node, so emitting twice doubles their embedding weight).
        if symbol.docstring and not symbol.body.lstrip().startswith(('"""', "'''", '"', "'")):
            lines.append(f"# Doc: {symbol.docstring}")
            lines.append("")

        # --- Symbol body ---
        lines.append(symbol.body)

        return "\n".join(lines)

    def _build_import_header(self, imports: list[ImportEdge]) -> str:
        """Summarise top-N imports as a compact comma-separated string."""
        if not imports:
            return ""
        seen: list[str] = []
        for imp in imports[: self.config.max_imports_in_header]:
            if imp.imported_names and imp.imported_names != ["*"]:
                names = ", ".join(imp.imported_names[:3])
                seen.append(f"{imp.imported_from}.{{{names}}}")
            else:
                seen.append(imp.imported_from)
        return ", ".join(seen)

    def _truncate_chunk(self, chunk_text: str, max_tokens: int) -> str:
        """Truncate chunk to max_tokens by cutting body lines from the bottom."""
        tokens = self._tokenizer.encode(chunk_text)
        truncated = self._tokenizer.decode(tokens[:max_tokens])
        return truncated + "\n# ... (truncated)"

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))
