"""Shared Rich Progress factory for trelix's CLI long-running operations.

Every indexing phase (parse, write, summarize, embed) and the migrate-vectors
command render progress through this single factory, so the column
configuration -- and any future change to it -- lives in one place instead
of being copy-pasted at each of the 7 call sites this replaces.
"""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)


def make_progress(console: Console) -> Progress:
    """Build a Progress bar showing both a literal 'X of Y' count and a
    percentage -- clig.dev/Evil Martians recommend the count as the default
    for measurable step-by-step work; the percentage is kept alongside it
    for anyone who was relying on the prior percentage-only display."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        console=console,
    )
