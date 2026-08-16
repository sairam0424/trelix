"""
Core data models for trelix.

These dataclasses are the single source of truth that flows through every
stage of the pipeline: walker → parser → chunker → store → retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SymbolKind(StrEnum):
    """Coarse-grained kind of a parsed symbol."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    CONSTANT = "constant"
    VARIABLE = "variable"
    MODULE = "module"  # file-level module symbol
    SECTION = "section"  # markdown heading section
    UNKNOWN = "unknown"


class Language(StrEnum):
    """Languages with Tree-sitter grammars and extractors."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    RAZOR = "razor"
    CSHTML = "cshtml"
    CSPROJ = "csproj"
    KOTLIN = "kotlin"
    RUBY = "ruby"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    HTML = "html"
    CSS = "css"
    # Operations and contract artifacts. Shell entrypoints, container definitions,
    # build files, migrations and protobuf schemas are among the things people search a
    # codebase for most often, and none of them were reachable: EXTENSION_MAP had no
    # entry, and `Path("Dockerfile").suffix` is "" so no extension entry could ever have
    # matched one.
    SHELL = "shell"
    DOCKERFILE = "dockerfile"
    MAKE = "make"
    SQL = "sql"
    PROTO = "proto"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------


@dataclass
class IndexedFile:
    """
    A file that has been discovered and (optionally) parsed.
    `hash` is SHA-256 of file content — used for incremental re-indexing.
    """

    path: str  # absolute path on disk
    rel_path: str  # path relative to repo root — stable key
    language: Language
    hash: str  # SHA-256 of file content
    size_bytes: int
    id: int | None = None
    indexed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Symbol  (output of Tree-sitter parsing)
# ---------------------------------------------------------------------------


@dataclass
class Symbol:
    """
    A single named construct extracted from a file by Tree-sitter.

    `qualified_name` follows the pattern: ClassName.method_name
    `body` is the full source text of the symbol (verbatim from the file).
    `parent_id` links methods back to their enclosing class symbol.
    """

    file_id: int
    name: str
    qualified_name: str  # e.g. "LoginView.authenticate_user"
    kind: SymbolKind
    line_start: int  # 1-indexed
    line_end: int  # 1-indexed, inclusive
    signature: str  # e.g. "def authenticate_user(self, username: str) -> User"
    body: str  # full source text
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    is_public: bool = True
    parent_id: int | None = None  # enclosing class/struct symbol id
    id: int | None = None
    context_summary: str | None = None  # LLM-generated summary (contextual chunking)


# ---------------------------------------------------------------------------
# Call edge
# ---------------------------------------------------------------------------


@dataclass
class CallEdge:
    """
    A directed edge in the call graph: caller_id calls callee_name at `line`.
    `callee_id` is resolved after all symbols are stored (may remain None for
    external / stdlib calls).

    `callee_type_hint` carries the static type of the receiver object when the
    call is a method call on a typed variable.  For example, given:

        user_service: UserService
        user_service.login(username)

    the extractor stores callee_type_hint="UserService" so that resolution can
    prefer symbols whose qualified_name starts with "UserService." over bare
    name-only matches — dramatically reducing false-positive edges.
    """

    caller_id: int
    callee_name: str
    line: int
    callee_id: int | None = None
    callee_type_hint: str | None = None


# ---------------------------------------------------------------------------
# Type edge
# ---------------------------------------------------------------------------


@dataclass
class TypeEdge:
    """
    A directed edge in the type hierarchy: from_symbol extends/implements to_type_name.

    edge_kind values:
      "extends"    — class inheritance
      "implements" — interface implementation
      "trait_impl" — Rust impl Trait for Type
      "embedded"   — Go struct embedding
    """

    from_symbol_id: int  # local idx during parse, remapped by Indexer to DB id
    to_type_name: str
    edge_kind: str
    to_symbol_id: int | None = None


# ---------------------------------------------------------------------------
# Generic (cross-source) edge
# ---------------------------------------------------------------------------


@dataclass
class GenericEdge:
    """
    A typed edge from a code symbol to a non-code artefact (ticket, test,
    doc, ...) that has no row of its own in this DB.

    Exclusive arc: `from_symbol_id` is a real symbol id (every writer today,
    e.g. the git-log ticket linker, always attaches from real code); the
    other side is `source_ref`, a free-form "<type>:<ref>" string (e.g.
    "ticket:PROJ-123") rather than a foreign key, since a generic external
    artefact has no id in this DB to reference.

    edge_kind values:
      "references_ticket" — code touched by a commit that mentions a ticket ID
      "references_artifact" — a symbol whose name/qualified_name (or, absent
        a match, embedding similarity) was found in a connector-fetched
        artifact's title/body (see ArtifactLinker)
    """

    from_symbol_id: int
    source_ref: str
    edge_kind: str
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Artifact  (non-code content fetched by a source connector)
# ---------------------------------------------------------------------------


@dataclass
class Artifact:
    """
    Content fetched from an external system (Jira, TestRail, ...) by a
    source connector — a ticket, a test case, etc.

    Deliberately NOT a Chunk: Chunk.symbol_id is a hard, NOT NULL foreign
    key to symbols.id, read through multiple retrieval-leg hydration paths
    (grep, bm25, vector) that all assume it resolves to a real symbol with a
    real file/line-span. Relaxing that FK to fit non-code content would
    require defensive branching sprinkled across every one of those call
    sites — a much larger blast radius than a new, isolated model. Artifacts
    join to code purely through generic_edges.source_ref, matching that
    table's own exclusive-arc design rather than introducing a new pattern
    on top of it.

    `source_ref` is the same string an edge's source_ref would use for this
    artefact (e.g. "ticket:PROJ-123") — the join key back to generic_edges.
    """

    source_ref: str
    artifact_kind: str  # "ticket" | "test_case"
    title: str
    body: str
    url: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    id: int | None = None


# ---------------------------------------------------------------------------
# Import edge
# ---------------------------------------------------------------------------


@dataclass
class ImportEdge:
    """
    Tracks what a file imports and from where.
    `imported_names` is ["*"] for wildcard imports.
    """

    file_id: int
    imported_from: str  # module path, e.g. "django.contrib.auth"
    imported_names: list[str]  # ["authenticate", "logout"]


# ---------------------------------------------------------------------------
# Chunk  (output of Chunker — what gets embedded)
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """
    The unit that gets embedded and stored in the vector store.

    `chunk_text` includes a context header (file path, language, imports,
    parent class) so the embedding model has full context without seeing
    the whole file.

    Example chunk_text:
        # File: src/auth/login.py | Language: Python
        # Imports: django.contrib.auth, .models
        # Class: LoginView

        def authenticate_user(self, username, password):
            ...
    """

    symbol_id: int
    chunk_text: str  # context header + symbol body
    token_count: int  # pre-computed via tiktoken
    embedding: list[float] | None = None
    id: int | None = None


# ---------------------------------------------------------------------------
# Search result  (output of Retrieval)
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """
    A single result from any retrieval method, before fusion/reranking.
    `source` tracks which retrieval method produced it for RRF fusion.
    """

    chunk: Chunk
    symbol: Symbol
    file: IndexedFile
    score: float
    rank: int
    source: str  # "vector" | "bm25" | "graph_expansion"


# ---------------------------------------------------------------------------
# Retrieved context  (final output sent to LLM)
# ---------------------------------------------------------------------------


@dataclass
class RetrievedContext:
    """
    The assembled context block ready to be injected into an LLM prompt.
    `context_text` is the final formatted string within the token budget.
    """

    query: str
    results: list[SearchResult]
    context_text: str
    total_tokens: int
    retrieval_sources: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    intent: str = ""  # planner intent — used by synthesizer for per-intent prompts
