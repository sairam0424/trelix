"""What the eval file loaders (`prereg`, `results`) share: a bounded read and an error type.

Both loaders report every problem they find in one go, like `_parse_golden`, so a file with
three mistakes is fixed in one edit, not three.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

_CLIP_CHARS = 60
_MAX_PROBLEMS = 20


class ProblemsError(ValueError):
    """A file that cannot be used, with the reasons found (`problems`), not just the first.

    The first 20 are kept and a last entry counts the rest, so a file full of mistakes
    cannot flood a terminal.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        listed = list(problems[:_MAX_PROBLEMS])
        if len(problems) > _MAX_PROBLEMS:
            listed.append(f"... and {len(problems) - _MAX_PROBLEMS} more problems")
        self.problems: tuple[str, ...] = tuple(listed)
        super().__init__("; ".join(self.problems))


def read_text_capped(path: Path, max_bytes: int) -> str:
    """The UTF-8 text of `path`, reading at most `max_bytes + 1` bytes of it.

    Raises OSError when the file cannot be opened or read and ValueError when it is over
    `max_bytes` or is not UTF-8 (`UnicodeDecodeError` is a ValueError). A file over the cap
    is refused without being read into memory whole.
    """
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"is larger than {max_bytes} bytes")
    return data.decode("utf-8")


def clip(text: str, limit: int = _CLIP_CHARS) -> str:
    """`text` on one line and cut to `limit` characters with "..." at the cut.

    For a value read from a file. Runs of whitespace, newlines included, collapse to one space,
    so the value cannot start a line of its own and pass for a `verdict:` or `reason:` line.
    """
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def show(value: object) -> str:
    """`repr(value)`, clipped: how a message quotes a value a file supplied."""
    return clip(repr(value))
