"""Keep a tool result inside a client's context budget.

A client pays for every character of a tool result, about twice: FastMCP sends a dict result as a
text block and again as `structuredContent`, and the text block holds the JSON as a string, so each
quote and backslash in it gains an escape (measured: `search_code` at k=100 is about 97,000
characters of text and 197,000 on the wire, past Claude Code's 25,000-token cap). Three things
bound it here, all reading the environment on each call:

* `TRELIX_MCP_MAX_K` (default 50) clamps `k` and `limit` to 1..that value.
* `TRELIX_MCP_MAX_RESULT_CHARS` (default 15,000; 0 disables) is the budget for the text of a list
  result. The tail is dropped and the response says so. Both copies are counted, so the whole
  response stays within twice the budget (30,000 characters by default), not just its text.
* `detail="concise"` swaps each result's `body` for a one-line `signature`.

Kept out of server.py on purpose: server.py is already very large and its tools only call in
here. Every helper returns today's shapes (an envelope dict or a bare array) plus additive keys.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import pydantic_core
from fastmcp.exceptions import ToolError
from fastmcp.tools.base import ToolResult
from mcp.types import TextContent

MAX_K_ENV = "TRELIX_MCP_MAX_K"
MAX_RESULT_CHARS_ENV = "TRELIX_MCP_MAX_RESULT_CHARS"
DEFAULT_MAX_K = 50
DEFAULT_MAX_RESULT_CHARS = 15_000
DEFAULT_MAX_BODY_CHARS = 20_000
DEFAULT_BLAST_LIMIT = 100
MAX_BLAST_LIMIT = 500
SIGNATURE_MAX_CHARS = 200
# A session's most recent prompt has no bound of its own, and the fit always keeps one row, so the
# listing shows only its start.
SESSION_QUERY_MAX_CHARS = 300
# A client is sent the JSON of a result twice (text block and `structuredContent`).
COPIES_SENT = 2
# What the CallToolResult adds around those two copies: its keys, `_meta`, and a cut array's
# wrapper. Measured at 213 to 358 characters with model_dump_json(by_alias=True).
ENVELOPE_CHARS = 400

Detail = Literal["concise", "detailed"]


class BudgetConfigError(ValueError):
    """An output-limit environment variable holds a value the server cannot use."""


@dataclass(frozen=True)
class Limits:
    max_k: int
    max_result_chars: int


def _read_int(environ: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    message = f"{name} must be an integer of at least {minimum}, got {raw!r}"
    try:
        value = int(raw)
    except ValueError:
        raise BudgetConfigError(message) from None
    if value < minimum:
        raise BudgetConfigError(message)
    return value


def limits_from_env(environ: Mapping[str, str] | None = None) -> Limits:
    """The limits in force. A blank variable means its default; an unusable one raises."""
    env = os.environ if environ is None else environ
    return Limits(
        max_k=_read_int(env, MAX_K_ENV, DEFAULT_MAX_K, 1),
        max_result_chars=_read_int(env, MAX_RESULT_CHARS_ENV, DEFAULT_MAX_RESULT_CHARS, 0),
    )


def clamp_page_size(value: int, maximum: int) -> int:
    return max(1, min(value, maximum))


def check_cursor(cursor: int) -> None:
    if cursor < 0:
        raise ToolError(
            f"cursor must be 0 or greater, got {cursor}. Use 0 for the first page, then pass "
            "the next_cursor value from the previous response."
        )


def _first_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[0][:SIGNATURE_MAX_CHARS] if lines else ""


def body_or_signature(symbol: Any, detail: Detail, body_chars: int) -> dict[str, str]:
    """`{"body": ...}` for detail=detailed (today's text), `{"signature": ...}` for concise."""
    if detail == "concise":
        return {"signature": _first_line(symbol.signature) or _first_line(symbol.body)}
    return {"body": symbol.body[:body_chars]}


def cap_session_query(session: dict[str, Any]) -> dict[str, Any]:
    """`session` with a `query` over SESSION_QUERY_MAX_CHARS cut, and `query_truncated` true.

    A session whose query fits is returned as it is, without the extra key.
    """
    query = session["query"]
    if len(query) <= SESSION_QUERY_MAX_CHARS:
        return session
    return {**session, "query": query[:SESSION_QUERY_MAX_CHARS], "query_truncated": True}


def _text(value: Any) -> str:
    """The text FastMCP sends for a dict or list result: compact JSON, non-ASCII kept."""
    return pydantic_core.to_json(value, fallback=str).decode()


def _sent_chars(text: str, note: str = "") -> int:
    """Characters a client is sent for a result whose JSON is `text`, and an optional `note` block.

    The JSON travels twice: in the text block as a JSON string, where each quote and backslash
    gains an escape, and in `structuredContent` as it is. ENVELOPE_CHARS covers the rest.
    """
    sent = len(text) + len(_text(text)) + ENVELOPE_CHARS
    if note:
        sent += len(_text(note))
    return sent


def _largest_fit(count: int, size_of: Callable[[int], int], max_chars: int) -> int:
    """The most items (at least one) whose response is sent within twice `max_chars`.

    `size_of(n)` is `_sent_chars` with the first n items kept. It grows with n (an item adds far
    more characters than a changed digit in a cursor or a count removes), so a binary search is
    exact. One item is always kept so that paging advances, even under a tiny budget.
    """
    budget = COPIES_SENT * max_chars
    if budget <= 0 or count <= 1 or size_of(count) <= budget:
        return count
    low, high = 1, count - 1
    while low < high:
        middle = (low + high + 1) // 2
        if size_of(middle) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def fit_rows(
    rows: Sequence[dict[str, Any]],
    build: Callable[[list[dict[str, Any]], int], dict[str, Any]],
    max_chars: int,
) -> dict[str, Any]:
    """`build(kept_rows, omitted_count)` for the most rows whose response fits twice `max_chars`."""
    items = list(rows)

    def size_of(kept: int) -> int:
        return _sent_chars(_text(build(items[:kept], len(items) - kept)))

    kept = _largest_fit(len(items), size_of, max_chars)
    return build(items[:kept], len(items) - kept)


def fit_page(
    rows: Sequence[dict[str, Any]],
    *,
    cursor: int,
    page_size: int,
    total: int,
    max_chars: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A cursor-paged envelope. `rows` is the page that starts at `cursor`.

    When the tail is dropped, `next_cursor` points at the first dropped row, so the next page
    repeats nothing and skips nothing. `page_size` is the clamped k the server applied.
    """

    def build(kept: list[dict[str, Any]], omitted: int) -> dict[str, Any]:
        position = cursor + len(kept)
        return {
            "results": kept,
            "next_cursor": position if position < total else None,
            "total_available": total,
            "page_size": page_size,
            "truncated": omitted > 0,
            "omitted": omitted,
            **(extra or {}),
        }

    return fit_rows(rows, build, max_chars)


def bare_array_result(
    rows: Sequence[dict[str, Any]],
    *,
    total_available: int,
    max_chars: int,
    noun: str,
    remedy: str,
) -> list[dict[str, Any]]:
    """A bare-array result, cut to the budget. Untouched (a plain list) when nothing is omitted.

    When rows are omitted, `content[0]` is still a JSON array (a client that reads only the
    first text block keeps working), `content[1]` is a note that ends with `remedy` (what the
    caller can change to see more), and `_meta.trelix` carries `total_available` and `omitted`,
    so a cut can never go unnoticed.

    The declared type is the list the tool's output schema is built from: FastMCP builds that
    schema from the return annotation and passes a `ToolResult` through unchanged, while a
    `list | ToolResult` annotation would drop the schema (and structuredContent) for every call.
    """
    items = list(rows)

    def note(kept: int) -> str:
        return (
            f"Truncated: {kept} of {total_available} {noun} returned, "
            f"{total_available - kept} omitted. {remedy}"
        )

    def size_of(kept: int) -> int:
        return _sent_chars(_text(items[:kept]), note(kept) if kept < total_available else "")

    kept = _largest_fit(len(items), size_of, max_chars)
    omitted = total_available - kept
    if omitted == 0:
        return items
    result = ToolResult(
        content=[
            TextContent(type="text", text=_text(items[:kept])),
            TextContent(type="text", text=note(kept)),
        ],
        structured_content={"result": items[:kept]},
        meta={
            "fastmcp": {"wrap_result": True},
            "trelix": {"total_available": total_available, "omitted": omitted},
        },
    )
    return cast("list[dict[str, Any]]", result)


def check_body_limit(max_chars: int) -> None:
    if max_chars < 0:
        raise ToolError(
            f"max_body_chars must be 0 (no limit) or greater, got {max_chars}. "
            "Use a positive number to cut the body."
        )


def truncate_body(body: str, max_chars: int) -> tuple[str, bool]:
    """`body` cut to `max_chars` (0 = no limit) and whether it was cut."""
    if max_chars == 0 or len(body) <= max_chars:
        return body, False
    return body[:max_chars], True
