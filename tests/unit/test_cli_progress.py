"""Regression test for the shared progress-bar helper (make_progress)."""

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TaskProgressColumn,
)

from trelix.cli.progress import make_progress


def test_make_progress_includes_both_count_and_percentage():
    """The literal 'X of Y' count and the percentage must both be visible --
    clig.dev/Evil Martians recommend the count as the default for
    measurable step-by-step work; the percentage stays for users who
    prefer it. Neither should be silently dropped by a future refactor."""
    console = Console()
    progress = make_progress(console)
    column_types = [type(c) for c in progress.columns]

    assert SpinnerColumn in column_types
    assert BarColumn in column_types
    assert MofNCompleteColumn in column_types
    assert TaskProgressColumn in column_types
    assert progress.console is console
