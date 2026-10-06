"""Fixture F1 for the citation-tag tests: three results over two files.

Score order is verify (0.9) > decode_token (0.8) > bearer (0.7), while the assembler
RENDERS middleware.py first (verify, then bearer by line) and jwt.py second, so the
rendered order differs from the score order. Shared by tests/unit/test_citation_tags.py,
test_citation_tags_compressed.py and test_citation_prompts.py; in-memory only.

The three ``*_BLOCK`` literals are what the assembler rendered for each result before
citation tags existed (the design's observed value); the tests prefix them.
"""

from __future__ import annotations

from datetime import datetime

from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    RetrievedContext,
    SearchResult,
    Symbol,
    SymbolKind,
)

QUERY = "how is a jwt verified"
MIDDLEWARE = "src/auth/middleware.py"
JWT = "src/auth/jwt.py"


def make_file(file_id: int, rel_path: str) -> IndexedFile:
    return IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash=f"sha-{file_id}",
        size_bytes=100,
        id=file_id,
        indexed_at=datetime(2024, 1, 1),
    )


def make_result(
    *,
    symbol_id: int,
    rel_path: str,
    file_id: int,
    name: str,
    qualified_name: str,
    kind: SymbolKind,
    line_start: int,
    line_end: int,
    body: str,
    score: float,
    rank: int,
    source: str = "vector",
) -> SearchResult:
    """A result whose chunk text is the chunker's ``# File: ... | Language: ...`` header
    plus ``body``, with a fixed token_count of 20."""
    symbol = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        line_start=line_start,
        line_end=line_end,
        signature="",
        body=body,
        id=symbol_id,
    )
    chunk_text = f"# File: {rel_path} | Language: Python\n\n{body}"
    return SearchResult(
        chunk=Chunk(symbol_id=symbol_id, chunk_text=chunk_text, token_count=20, id=symbol_id),
        symbol=symbol,
        file=make_file(file_id, rel_path),
        score=score,
        rank=rank,
        source=source,
    )


def verify_result(*, score: float = 0.9, rank: int = 1, source: str = "vector") -> SearchResult:
    """AuthMiddleware.verify, symbol 11, src/auth/middleware.py lines 42-67."""
    return make_result(
        symbol_id=11,
        rel_path=MIDDLEWARE,
        file_id=1,
        name="verify",
        qualified_name="AuthMiddleware.verify",
        kind=SymbolKind.METHOD,
        line_start=42,
        line_end=67,
        body="def verify(self, token):\n    return decode_token(token)",
        score=score,
        rank=rank,
        source=source,
    )


def decode_token_result() -> SearchResult:
    """decode_token, symbol 22, src/auth/jwt.py lines 10-30 (a top-level function)."""
    return make_result(
        symbol_id=22,
        rel_path=JWT,
        file_id=2,
        name="decode_token",
        qualified_name="decode_token",
        kind=SymbolKind.FUNCTION,
        line_start=10,
        line_end=30,
        body="def decode_token(token):\n    return jwt.decode(token, SECRET)",
        score=0.8,
        rank=2,
    )


def bearer_result(rank: int = 3) -> SearchResult:
    """AuthMiddleware.bearer, symbol 33, src/auth/middleware.py lines 70-80."""
    return make_result(
        symbol_id=33,
        rel_path=MIDDLEWARE,
        file_id=1,
        name="bearer",
        qualified_name="AuthMiddleware.bearer",
        kind=SymbolKind.METHOD,
        line_start=70,
        line_end=80,
        body="def bearer(self, header):\n    return header.split()[1]",
        score=0.7,
        rank=rank,
    )


def f1_results() -> list[SearchResult]:
    """The three results in score order."""
    return [verify_result(), decode_token_result(), bearer_result()]


def sources_of(context: RetrievedContext) -> list[tuple[int, int, str, int, int, str]]:
    """``citation_sources`` as plain tuples, for literal comparison."""
    return [
        (s.tag, s.symbol_id, s.path, s.line_start, s.line_end, s.symbol)
        for s in context.citation_sources
    ]


VERIFY_BLOCK = (
    "[Lines 42-67] AuthMiddleware.verify\n"
    "# File: src/auth/middleware.py | Language: Python\n"
    "\n"
    "def verify(self, token):\n"
    "    return decode_token(token)\n"
)
BEARER_BLOCK = (
    "[Lines 70-80] AuthMiddleware.bearer\n"
    "# File: src/auth/middleware.py | Language: Python\n"
    "\n"
    "def bearer(self, header):\n"
    "    return header.split()[1]\n"
)
DECODE_BLOCK = (
    "[Lines 10-30] decode_token\n"
    "# File: src/auth/jwt.py | Language: Python\n"
    "\n"
    "def decode_token(token):\n"
    "    return jwt.decode(token, SECRET)\n"
)
