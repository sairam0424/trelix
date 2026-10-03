"""What became of each reviewed hunk, and the parsing that decides it.

`DiffReviewer.review()` used to return `[]` both when the model looked at a hunk and found
nothing and when the reply was cut off, refused, filtered or not JSON at all. A hunk now gets
one of five statuses, and only a parsed JSON array that followed a clean stop counts as
reviewed.

The statuses and their `detail` strings are a fixed vocabulary on purpose. They end up in the
outcome file and in a published GitHub Check, so they never carry refusal text or model prose:
the only free-form part is a provider stop token, which `safe_token` restricts to
`[A-Za-z0-9_.-]{1,40}`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

_TOKEN = re.compile(r"[A-Za-z0-9_.-]{1,40}")
# "No issues" said as an empty array: the reply starts with it (and the line ends there), or a
# code fence holds only it. An empty array that merely appears inside prose ("I would return []
# if ...", "[] (but I found bugs)") is neither. Fences are found line by line, never with a regex
# that scans the whole reply: model output can be tens of thousands of backticks or spaces, and a
# scanning pattern over that is quadratic.
_LEADING_EMPTY = re.compile(r"\[\s*\][ \t]*(?:\n|\Z)")
_EMPTY_ARRAY = re.compile(r"\[\s*\]")
_MAX_FENCE_INFO = 30
# Where a review array could start inside prose: "[" and then an object ("x[i]" is not one).
_ARRAY_OF_OBJECTS = re.compile(r"\[\s*\{")
_MAX_ARRAY_CANDIDATES = 20


class HunkStatus(StrEnum):
    REVIEWED = "reviewed"  # a parsed JSON array after a clean stop (it may be empty)
    TRUNCATED = "truncated"  # cut off, even after a retry at a larger limit
    REFUSED = "refused"  # the model declined, or a provider filter withheld the reply
    PARSE_FAILED = "parse_failed"  # a clean stop, but the reply holds no usable review array
    ERROR = "error"  # the call raised, or the provider reported an error or an unknown stop


@dataclass(frozen=True)
class HunkResult:
    """One hunk's status. `line` is the first line of the hunk in the new file.

    `kept_comments` is how many findings the hunk still produced; a truncated hunk can keep
    the complete ones it salvaged. It is for callers deciding what to show: it is not part of
    the outcome file and not part of equality.
    """

    file_path: str
    line: int
    status: HunkStatus
    detail: str = ""
    kept_comments: int = field(default=0, compare=False)

    @property
    def reviewed(self) -> bool:
        return self.status is HunkStatus.REVIEWED

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file_path,
            "line": self.line,
            "status": self.status.value,
            "detail": self.detail,
        }


def safe_token(value: object, *, default: str = "none") -> str:
    """A provider stop token if it is short and plain, else `default`."""
    if isinstance(value, str) and _TOKEN.fullmatch(value):
        return value
    return default


def _normalise(content: str) -> str:
    text = content[1:] if content.startswith("\ufeff") else content
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _is_fence(line: str) -> bool:
    """True for a code-fence line: three or more backticks or tildes and an optional info word."""
    stripped = line.strip()
    if not stripped or stripped[0] not in "`~":
        return False
    marks = len(stripped) - len(stripped.lstrip(stripped[0]))
    if marks < 3:
        return False
    info = stripped[marks:].strip()
    return not info or (
        len(info) <= _MAX_FENCE_INFO
        and info.isascii()
        and all(c.isalnum() or c in "_-" for c in info)
    )


def _unfence(text: str) -> str:
    """The body of a reply that is a single code fence from first line to last, else the reply."""
    lines = text.split("\n")
    if len(lines) >= 3 and _is_fence(lines[0]) and _is_fence(lines[-1]):
        return "\n".join(lines[1:-1]).strip()
    return text


def _has_fenced_empty_array(text: str) -> bool:
    """True when some code fence in the reply holds nothing but an empty array."""
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        if not _is_fence(lines[index]):
            index += 1
            continue
        closing = index + 1
        while closing < len(lines) and not _is_fence(lines[closing]):
            closing += 1
        if closing >= len(lines):
            return False
        if _EMPTY_ARRAY.fullmatch("\n".join(lines[index + 1 : closing]).strip()):
            return True
        index = closing + 1
    return False


def _usable(items: list[Any]) -> list[dict[str, Any]] | None:
    """The review objects in a parsed array, or None when a non-empty array holds none.

    An object counts only if it has a non-empty text `comment`. `[]` is a review with no
    findings; `[{}]` or objects under another key is not a review at all.
    """
    if not items:
        return []
    usable = [
        item
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("comment"), str)
        and item["comment"].strip()
    ]
    return usable or None


def extract_review_items(content: str) -> list[dict[str, Any]] | None:
    """The review objects in a reply, or None when the reply holds no review array.

    Accepted: the reply is the array (optionally in a code fence); or it holds an array of
    objects among prose, at least one with a text `comment`. `[]` means "no issues" only as
    the whole reply, at its start, or alone in a code fence. A plausible array of objects
    that does not parse is a cut-off review, not a cue to look for another array inside it.
    """
    text = _normalise(content)
    if not text:
        return None
    try:
        whole = json.loads(_unfence(text))
    except (ValueError, RecursionError):  # JSONDecodeError is a ValueError; so is a 5000-digit int
        whole = None
    if isinstance(whole, list):
        return _usable(whole)

    decoder = json.JSONDecoder()
    unusable_array = False
    for _, match in zip(
        range(_MAX_ARRAY_CANDIDATES), _ARRAY_OF_OBJECTS.finditer(text), strict=False
    ):
        try:
            parsed, _end = decoder.raw_decode(text, match.start())
        except (ValueError, RecursionError):
            return None
        if isinstance(parsed, list):
            usable = _usable(parsed)
            if usable:
                return usable
            unusable_array = True
    if unusable_array:
        # A reply that also shows an array of objects with no comment in it is not a clean
        # "no issues": an empty array next to it must not be read as the review.
        return None
    if _LEADING_EMPTY.match(text) or _has_fenced_empty_array(text):
        return []
    return None


def salvage_review_items(content: str) -> list[dict[str, Any]]:
    """The complete objects at the start of a review array that was cut off mid-way.

    Stops at the first thing that is not a whole JSON object, so a half-written final object is
    dropped and nothing after it is guessed at. Only objects with a text `comment` are kept.
    """
    text = _normalise(content)
    opening = _ARRAY_OF_OBJECTS.search(text)
    if opening is None:
        return []
    decoder = json.JSONDecoder()
    position = opening.start() + 1
    objects: list[dict[str, Any]] = []
    while True:
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        if position >= len(text) or text[position] != "{":
            break
        try:
            parsed, position = decoder.raw_decode(text, position)
        except (ValueError, RecursionError):
            break
        if isinstance(parsed, dict):
            objects.append(parsed)
    return _usable(objects) or []
