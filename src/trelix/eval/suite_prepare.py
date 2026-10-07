"""Prepare a suite: prove its files, then clone and prove its repository.

The order is cheapest first, and nothing is cloned, and no cache directory is created, until
the inputs are proven: `suite.json` and both hashes, then the golden file and the plans file
together (every golden query has a recorded plan). Only then does the clone happen, followed by
the gold-path check against the clone. This is everything `trelix eval-suite --prepare-only`
does, and the first step of a run (`trelix.eval.suite_run`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trelix.eval.suite import SuiteError, SuiteSpec, input_problems, load_suite
from trelix.eval.suite_git import default_cache_root, ensure_clone
from trelix.eval.suite_gold import check_gold


@dataclass(frozen=True)
class PreparedSuite:
    """A suite whose inputs and clone have been verified."""

    spec: SuiteSpec
    clone: Path
    queries: int
    gold_files: int


def cache_root(cache_dir: Path | str | None) -> Path:
    """`cache_dir` as an absolute path, or `default_cache_root()` when it is None or empty.

    An empty string would otherwise mean the current directory.
    """
    root = Path(cache_dir) if cache_dir else default_cache_root()
    return root.resolve()


def prepare_suite(
    suite_file: Path | str, cache_dir: Path | str | None = None, *, allow_local: bool = False
) -> PreparedSuite:
    """Verify `suite_file` and its clone, cloning when needed. Raises SuiteError with all reasons.

    `cache_dir` is the cache root (see `cache_root`). `allow_local` lets `repo.url` be a local
    repository, for the tests; the command line never sets it.
    """
    spec = load_suite(suite_file, allow_local=allow_local)
    return prepare_spec(spec, cache_root(cache_dir), allow_local=allow_local)


def prepare_spec(spec: SuiteSpec, root: Path, *, allow_local: bool = False) -> PreparedSuite:
    """`prepare_suite` for a loaded `spec` under the cache root `root`."""
    problems = input_problems(spec)
    if problems:
        raise SuiteError(problems)
    clone = ensure_clone(spec, root, allow_local=allow_local)
    counts = check_gold(spec, clone)
    return PreparedSuite(spec=spec, clone=clone, queries=counts.queries, gold_files=counts.files)
