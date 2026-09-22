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

Markdown sections get the same header treatment, plus a diagram tag when
the section's body contains a fenced mermaid/plantuml block (only the
first such fence per section is tagged):
    # File: docs/architecture.md | Language: Markdown
    # Diagram: mermaid

    ## Architecture Overview

    ```mermaid
    graph TD
        A --> B
    ```
"""

from __future__ import annotations

import logging
import re
from typing import Any

import tiktoken

from trelix.core.config import ChunkerConfig
from trelix.core.models import Chunk, ImportEdge, Symbol
from trelix.llm.client import ChatMessage, TrelixChatClient

logger = logging.getLogger("trelix.indexing.chunker")

_DIAGRAM_FENCE_RE = re.compile(r"^```(mermaid|plantuml)\b", re.MULTILINE)


class Chunker:
    """
    Converts symbols into Chunk objects with context headers.

    Usage:
        chunker = Chunker(config.chunker)
        chunks = chunker.build_chunks(symbols, imports, file_rel_path, language)
    """

    # Marker inserted into every piece of a split chunk, right after the
    # structural header. Measured to cost the same 9 cl100k_base tokens
    # regardless of digit count (tiktoken tokenizes 1-2 digit numbers as single
    # tokens), so sizing the per-piece body budget off a 2-digit placeholder
    # ("99 of 99") is a safe, if slightly conservative, upper bound for the
    # realistic range of split counts.
    _SPLIT_MARKER_TEMPLATE = "# Split: chunk {i} of {n}"

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

            # If the chunk exceeds budget, split the body across multiple
            # sequential chunks instead of truncating (and losing) the tail.
            if token_count > self.config.max_tokens_per_chunk:
                for piece_text in self._split_chunk_text(
                    symbol, chunk_text, self.config.max_tokens_per_chunk
                ):
                    chunks.append(
                        Chunk(
                            symbol_id=symbol.id or 0,
                            chunk_text=piece_text,
                            token_count=len(self._tokenizer.encode(piece_text)),
                        )
                    )
            else:
                chunks.append(
                    Chunk(
                        symbol_id=symbol.id or 0,
                        chunk_text=chunk_text,
                        token_count=token_count,
                    )
                )

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
        if (
            self.config.include_parent_signature
            and symbol.parent_id
            and symbol.parent_id in parent_symbols
        ):
            parent = parent_symbols[symbol.parent_id]
            lines.append(f"# {parent.kind.value.capitalize()}: {parent.name}")

        # Diagram-block tagging: mirrors the "# File:"/"# Class:" header
        # convention. Only the first fenced mermaid/plantuml block in the
        # symbol body is tagged (v1 limitation, not a bug).
        if self.config.include_diagram_tags:
            diagram_match = _DIAGRAM_FENCE_RE.search(symbol.body)
            if diagram_match:
                lines.append(f"# Diagram: {diagram_match.group(1)}")

        lines.append("")  # blank line between header and body

        # Docstring — surfaced before body so it is part of the header and
        # therefore repeated on every split piece if the symbol is split.
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

    def _split_chunk_text(
        self,
        symbol: Symbol,
        base_chunk_text: str,
        max_tokens: int,
        prefix: str | None = None,
    ) -> list[str]:
        """
        Split an over-budget chunk into multiple sequential pieces instead of
        truncating it and discarding the tail.

        `base_chunk_text` is the header+body text as built by
        `_build_chunk_text` (i.e. it must end with exactly `symbol.body`, with
        nothing appended after it). Every returned piece repeats the
        structural context header (file/imports/parent/diagram/doc) plus a
        "# Split: chunk i of n" marker, so each piece stays independently
        embeddable and is never mistaken for the whole symbol — the same
        design goal this module states for the header itself.

        `prefix` (an LLM-generated context summary, from ContextualChunker) is
        prepended ONLY to the first piece: it describes the whole symbol once,
        so repeating it on every later piece would just eat into that piece's
        body-token budget to duplicate a summary the reader already saw in
        chunk 1 — a worse trade for embedding/retrieval quality than any
        single piece missing it.

        Always returns at least one piece, even when `symbol.body` is empty
        (or tiny) and the header alone already exceeds `max_tokens` — e.g. a
        very long docstring or import list with a near-empty function body.
        Returning zero pieces there would make the symbol vanish from every
        index (vector, BM25, everything) with no record it ever existed,
        which is strictly worse than the truncate-and-discard behavior this
        split path replaced. A single over-budget "header-only" chunk is far
        preferable to a silently missing symbol.
        """
        body = symbol.body
        assert base_chunk_text.endswith(body), (
            "base_chunk_text must end with symbol.body exactly — "
            "_build_chunk_text appends it as the final joined line"
        )
        header = base_chunk_text[: len(base_chunk_text) - len(body)]
        header_stripped = header.rstrip("\n")
        # Placeholder marker for sizing only: per the class-level comment on
        # _SPLIT_MARKER_TEMPLATE, every "chunk {i} of {n}" with i, n <= 99
        # encodes to the same token count, so measuring against "99 of 99"
        # is a safe, if slightly conservative, stand-in for the real marker
        # that gets substituted in once the final piece count `n` is known.
        marker_placeholder = self._SPLIT_MARKER_TEMPLATE.format(i=99, n=99)

        def assemble(piece_body: str, *, is_first: bool) -> str:
            text = f"{header_stripped}\n{marker_placeholder}\n\n{piece_body}"
            if is_first and prefix:
                text = f"{prefix}\n\n{text}"
            return text

        body_tokens = self._tokenizer.encode(body)

        def fit_end(start: int, *, is_first: bool) -> int:
            """
            Largest `end` such that `body_tokens[start:end]` fits `assemble()`
            within `max_tokens`, always advancing by at least one token when
            tokens remain (never silently drops body content).

            The initial estimate comes from encoding the literal skeleton
            (header + marker + blank line, with the prefix for the first
            piece) as ONE string, rather than summing independently-encoded
            header/marker/prefix token counts: tiktoken's BPE merges tokens
            across concatenation boundaries, so separately-encoded lengths
            don't reliably add up to the length of the joined string — this
            was the source of split pieces measuring 1 token over budget
            (e.g. 41 vs. a 40-token budget) despite the old per-piece math
            "adding up." Encoding the true skeleton removes that drift.

            That estimate still isn't a hard guarantee (the boundary between
            the skeleton's trailing blank line and the real body's first
            token can itself merge unpredictably), so the loop below verifies
            by encoding the actual assembled candidate and shrinks by one
            token at a time until it fits — capped at leaving exactly one
            body token in this piece, so a pathologically small `max_tokens`
            (smaller than the header+marker overhead alone) still makes
            forward progress instead of looping forever or dropping tokens.
            """
            if start >= len(body_tokens):
                return start
            overhead = len(self._tokenizer.encode(assemble("", is_first=is_first)))
            budget = max(max_tokens - overhead, 1)
            end = min(start + budget, len(body_tokens))
            while end > start + 1:
                candidate = str(self._tokenizer.decode(body_tokens[start:end]))
                if (
                    len(self._tokenizer.encode(assemble(candidate, is_first=is_first)))
                    <= max_tokens
                ):
                    break
                end -= 1
            return end

        piece_bodies: list[str] = []
        idx = 0
        is_first = True
        while idx < len(body_tokens) or is_first:
            end = fit_end(idx, is_first=is_first)
            piece_bodies.append(str(self._tokenizer.decode(body_tokens[idx:end])))
            idx = max(end, idx + 1)
            is_first = False

        n = len(piece_bodies)
        pieces: list[str] = []
        for i, piece_body in enumerate(piece_bodies, start=1):
            marker = self._SPLIT_MARKER_TEMPLATE.format(i=i, n=n)
            piece_text = f"{header_stripped}\n{marker}\n\n{piece_body}"
            if i == 1 and prefix:
                piece_text = f"{prefix}\n\n{piece_text}"
            pieces.append(piece_text)
        return pieces

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))


class ContextualChunker(Chunker):
    """
    Extends Chunker with LLM-generated context summaries (contextual chunking).

    When contextual=True and an llm_client is provided, each chunk gets an
    LLM-generated 2-3 sentence summary prepended to its chunk_text before
    embedding and BM25 indexing.  This reduces retrieval failure ~67% per
    Anthropic's contextual retrieval research.

    If contextual=False or llm_client=None, behaviour is identical to base Chunker.

    Usage:
        from openai import OpenAI
        client = OpenAI()
        chunker = ContextualChunker(config.chunker, llm_client=client)
        chunks = chunker.build_chunks(symbols, imports, file_rel_path, language)
    """

    _CONTEXT_PROMPT = (
        "In 2-3 sentences, describe what this code does. "
        "File: {rel_path}, Language: {language}. Code:\n{body}"
    )

    def __init__(
        self,
        config: ChunkerConfig,
        llm_client: Any | None = None,  # openai.OpenAI, TrelixChatClient, or None
    ) -> None:
        super().__init__(config)
        self._llm_client = llm_client

    @property
    def _contextual_enabled(self) -> bool:
        return self.config.contextual and self._llm_client is not None

    def build_chunks(
        self,
        symbols: list[Symbol],
        imports: list[ImportEdge],
        file_rel_path: str,
        language: str,
        parent_symbols: dict[int, Symbol] | None = None,
    ) -> list[Chunk]:
        """
        Build chunks, optionally prepending LLM-generated context summaries.

        If contextual mode is off or no LLM client was provided, delegates
        entirely to the base Chunker — zero overhead.
        """
        if not self._contextual_enabled:
            return super().build_chunks(symbols, imports, file_rel_path, language, parent_symbols)

        if parent_symbols is None:
            parent_symbols = {}

        import_header = self._build_import_header(imports)
        chunks: list[Chunk] = []

        for symbol in symbols:
            # Generate base chunk text (same as base Chunker)
            base_chunk_text = self._build_chunk_text(
                symbol=symbol,
                file_rel_path=file_rel_path,
                language=language,
                import_header=import_header,
                parent_symbols=parent_symbols,
            )

            # Generate LLM context summary
            context_summary = self._generate_summary(symbol, file_rel_path, language)

            if context_summary:
                # Store on the Symbol so it is persisted to the DB
                symbol.context_summary = context_summary
                # Prepend summary to chunk_text for richer embeddings + BM25
                chunk_text = f"{context_summary}\n\n{base_chunk_text}"
            else:
                chunk_text = base_chunk_text

            token_count = len(self._tokenizer.encode(chunk_text))

            if token_count > self.config.max_tokens_per_chunk:
                # Split on `base_chunk_text` (no summary), not `chunk_text`
                # (summary + base): _split_chunk_text prepends `prefix` to the
                # first piece itself, so passing the already-prefixed text
                # here would duplicate the summary onto chunk 1.
                for piece_text in self._split_chunk_text(
                    symbol,
                    base_chunk_text,
                    self.config.max_tokens_per_chunk,
                    prefix=context_summary,
                ):
                    chunks.append(
                        Chunk(
                            symbol_id=symbol.id or 0,
                            chunk_text=piece_text,
                            token_count=len(self._tokenizer.encode(piece_text)),
                        )
                    )
            else:
                chunks.append(
                    Chunk(
                        symbol_id=symbol.id or 0,
                        chunk_text=chunk_text,
                        token_count=token_count,
                    )
                )

        return chunks

    def _generate_summary(
        self,
        symbol: Symbol,
        rel_path: str,
        language: str,
    ) -> str | None:
        """
        Call the LLM to produce a 2-3 sentence summary of the symbol.
        Returns None on any failure so the pipeline degrades gracefully.
        """
        prompt = self._CONTEXT_PROMPT.format(
            rel_path=rel_path,
            language=language,
            body=symbol.body[:800],
        )
        try:
            assert self._llm_client is not None  # guaranteed by _contextual_enabled check
            # New path: TrelixChatClient interface
            if isinstance(self._llm_client, TrelixChatClient):
                response = self._llm_client.complete(
                    messages=[ChatMessage(role="user", content=prompt)],
                    max_tokens=self.config.contextual_max_tokens,
                    temperature=0,
                )
                return response.content.strip() or None
            # Legacy path: raw openai client (backward compat)
            response = self._llm_client.chat.completions.create(
                model=self.config.contextual_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=self.config.contextual_max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content.strip()  # type: ignore[no-any-return]
        except Exception as exc:  # noqa: BLE001
            logger.warning("ContextualChunker LLM call failed: %s", exc)
            return None
