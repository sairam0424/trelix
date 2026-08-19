"""Index provenance and worktree drift.

An index is a snapshot, but nothing recorded *of what*. `index_metadata` held exactly one
row — `embedding_dimension` — so a user querying an index had no way to answer the first
question they should ask: does this reflect the code in front of me? A stale index does
not fail loudly. It returns confident, well-ranked answers about code that has since
changed, which is worse than returning nothing.

Two separate things are recorded here, and the distinction matters:

* **Provenance** — which commit the index was built from, when, and with which embedder.
  This is context for reproducibility. It is cheap to read (a few key-value rows) and is
  never authoritative about staleness: a commit can match while the worktree is dirty,
  and a commit can differ while every indexed file is byte-identical.

* **Drift** — how many indexed files no longer match what is on disk. This is measured
  from content hashes and *is* authoritative, but it costs a walk plus a hash of every
  file, so it is opt-in rather than part of every `trelix stats`.

Drift deliberately reuses `FileWalker` rather than reimplementing discovery. The indexer's
own staleness rule is `db.get_file_hash(rel_path) != file.hash`, where the hash comes
from `FileWalker._compute_hash`. Reimplementing either would make the two disagree — a
different hash function would report every file stale, and a different ignore-filter
would invent files that were never indexable in the first place.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

logger = logging.getLogger("trelix.store.provenance")

# Namespaced so provenance keys can never collide with `embedding_dimension`, which
# predates them and is read by the dimension guard on a fixed key name.
_PREFIX = "provenance."

# Long enough for a cold-cache `git status` on a large worktree, short enough that a
# hung git cannot wedge an index run. Matches the 60s used in git_linker.
_GIT_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class IndexProvenance:
    """What the index was built from. Every field is optional by design.

    A non-git directory, a missing `git` binary and a shallow checkout are all normal,
    and none of them should stop an index from being written — so each field degrades to
    None independently rather than the whole record being absent.
    """

    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    indexed_at: str | None = None
    trelix_version: str | None = None
    embedder_provider: str | None = None
    embedder_model: str | None = None
    # Canonical JSON of everything that decides WHICH files get indexed: the walker
    # settings from `_WALK_FIELDS` plus a digest of the `.gitignore` chain. Stored as
    # named fields rather than one hash so a mismatch can name the offending setting
    # instead of only announcing that one exists — only the gitignore chain is reduced to
    # a digest, because its contents are unbounded.
    #
    # Describes the run that WROTE it, and nothing about the rows earlier runs wrote.
    walk_config: str | None = None
    # Every distinct `walk_config` that has ever written this index, as digests. Accumulated
    # by `write_provenance` rather than captured, because it is a property of the index on
    # disk and not of the current process.
    #
    # This is the field `walk_config` cannot be: a prune deletes PER-FILE, and one global
    # row cannot certify a per-file property. A re-index deletes no rows, so rows written
    # under a wide walk survive a run that narrows it — while the record is overwritten with
    # the narrow one. Snapshot and current then agree and the surviving rows read as
    # deletions. More than one member here means the index cannot certify that, and while it
    # holds `--prune` refuses.
    walk_config_history: tuple[str, ...] = ()

    _FIELDS = (
        "git_commit",
        "git_branch",
        "git_dirty",
        "indexed_at",
        "trelix_version",
        "embedder_provider",
        "embedder_model",
        "walk_config",
    )

    @property
    def is_empty(self) -> bool:
        """True when nothing was recorded — i.e. an index predating provenance."""
        return all(getattr(self, f) is None for f in self._FIELDS)


@dataclass(frozen=True)
class DriftReport:
    """How far the index has diverged from the worktree.

    `missing` is derived from "indexed paths the walk did not yield", which has exactly
    two innocent explanations besides deletion, and both are carried here so no caller
    can render the count without them:

    * `walk_was_complete` is False — the walk skipped a directory it could not read, so
      files under it read as missing while being present.
    * `walk_config_changed` is True — the walk was run with different ignore rules than
      the index was built with, so files it no longer reaches read as deleted. Measured
      at 35 phantom deletions on this repository from `extra_ignore_dirs` alone. Since
      `_gitignore_digest` this also covers edits to the `.gitignore` chain's *contents*,
      not merely to the config that switches it on.
    * `rows_share_one_walk_config` is False — more than one walk config has written this
      index, so a row may have come from a walk this one does not reproduce even when the
      recorded config matches the current one exactly. The two flags above compare ONE
      recorded config against now; this one is about the rows.

    Acting on `missing` in any of those cases deletes embeddings that cost money to
    recompute, so `actionable_drifted_count` — not `drifted_count` — is what a renderer
    should lead with.
    """

    stale: tuple[str, ...] = ()
    new: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unchanged_count: int = 0
    walk_was_complete: bool = True
    incomplete_paths: tuple[str, ...] = ()
    # {setting: (recorded_at_index_time, current)} — empty when identical OR when the
    # index predates walk-config recording. `walk_config_comparable` separates those.
    walk_config_diff: tuple[tuple[str, str, str], ...] = ()
    walk_config_comparable: bool = True
    # False when the index records more than one distinct walk config, or records no
    # history of them at all. Defaults True like the two flags above: a report built by
    # hand asserts its own preconditions, and only `compute_drift` can know this one.
    rows_share_one_walk_config: bool = True

    @property
    def walk_config_changed(self) -> bool:
        return bool(self.walk_config_diff)

    @property
    def missing_is_trustworthy(self) -> bool:
        """False when `missing` has an innocent explanation that was not ruled out."""
        return (
            self.walk_was_complete
            and self.walk_config_comparable
            and not self.walk_config_changed
            and self.rows_share_one_walk_config
        )

    @property
    def is_clean(self) -> bool:
        """True when every indexed file matches disk and nothing indexable is absent."""
        return not self.stale and not self.new and not self.missing

    @property
    def drifted_count(self) -> int:
        return len(self.stale) + len(self.new) + len(self.missing)

    @property
    def actionable_drifted_count(self) -> int:
        """`drifted_count` minus the part of it nothing has verified.

        This is the number a human can act on. `drifted_count` folds in `missing`
        unconditionally, and a renderer that leads with it contradicts the trustworthiness
        warning printed directly above: measured on this repository, `trelix stats --drift`
        announced "99 file(s) have drifted" where 35 were every file under `packages/`,
        all present on disk and simply unreachable by that walk.

        `stale` and `new` need no such gate — both come from files the walk actually
        yielded and hashed, so an incomplete or differently-configured walk can only make
        them *undercount*, never invent entries.
        """
        actionable = len(self.stale) + len(self.new)
        return actionable + len(self.missing) if self.missing_is_trustworthy else actionable


def _git(repo_path: Path, *args: str) -> str | None:
    """Run a read-only git command, returning stripped stdout or None on any failure.

    Never raises. A non-git directory, an absent `git`, a timeout and a non-zero exit
    all mean the same thing to the caller — this field is unknown — and provenance is
    strictly additive context, so none of them warrants failing an index.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=str(repo_path),
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git %s failed in %s: %s", " ".join(args), repo_path, exc)
        return None

    if result.returncode != 0:
        logger.debug(
            "git %s exited %d in %s: %s",
            " ".join(args),
            result.returncode,
            repo_path,
            result.stderr.strip()[:200],
        )
        return None
    return result.stdout.strip()


# The walker settings that decide WHICH files are yielded. Anything here differing
# between an index run and a later drift check makes `new` and `missing` meaningless:
# files the walk no longer reaches are indistinguishable from files that were deleted.
#
# This is not hypothetical. `extra_ignore_dirs` REPLACES the default 30-entry list rather
# than extending it, and `TRELIX_WALKER_*` is process-env-only (no `.env`), so the config
# that built an index is routinely not the config a later command reconstructs. Measured
# on this repository: a drift check run without `scripts/self-index.sh`'s environment
# reported 35 files under `packages/` as deleted when every one was present — the default
# ignore list contains "packages" for .NET NuGet output and hides this repo's own
# monorepo packages.
_WALK_FIELDS = (
    "languages",
    "extra_ignore_dirs",
    "extra_ignore_extensions",
    "extra_ignore_filenames",
    "max_file_size_bytes",
    # The bool alone is not enough — it says the chain was consulted, not what it said.
    # `_gitignore_digest` fingerprints the contents under the `gitignore_chain` key.
    "respect_gitignore",
    # Omitting this was a false-TRUE in `missing_is_trustworthy`: index with
    # follow_symlinks=True, then run a drift check with it False, and every file reachable
    # only through a symlink reports as `missing` while the report still claims the count
    # is trustworthy. A future `--prune` reading that would delete present files.
    # `test_walk_fields_covers_every_walker_setting` fails if a new field is added here
    # without being recorded.
    "follow_symlinks",
)


# Key under which the `.gitignore` chain digest is recorded. Deliberately NOT in
# `_WALK_FIELDS`: that tuple is pinned in both directions against
# `WalkerConfig.model_fields` (`test_recorded_fields_all_exist_on_the_config`), and this
# is derived from the worktree rather than from a config field.
_GITIGNORE_KEY = "gitignore_chain"

# Version of the `.gitignore` digest's OWN arithmetic, carried in the recorded value.
#
# Without it, any change to what the digest hashes reads as "the ignore rules moved" on
# every index already on disk — a false `walk_config_changed` for every existing user,
# announcing a narrowed walk when nothing narrowed but trelix's own hashing. That is a
# false positive on the one warning that must be believed, and it is not hypothetical:
# v3.1.2 started reading nested `.gitignore` chains and moved this project's own index by
# 74% without a single config field changing.
#
# A tagged digest lets `is_walk_config_comparable` say "recorded under an older scheme,
# cannot compare" instead. Both verdicts keep `missing_is_trustworthy` False and both keep
# `--prune` refusing, but only one of them is true. Bump this whenever the digest's inputs
# or format change.
_DIGEST_SCHEME = "v2"

# How many `.gitignore` directories the digest names before summarising the rest. Membership
# is the part a human can act on: `cp -a .trelix .trelix.bak-<name>` — this project's own
# documented backup step — copies `.trelix/.gitignore` into the walk, and a value reading
# only "4 file(s), sha256:…" cannot distinguish that from an edited ignore rule. Worse, the
# count can read IDENTICALLY on both sides while membership changed completely.
_GITIGNORE_MEMBERS_SHOWN = 8


def _gitignore_digest(config: IndexConfig) -> str | None:
    """Fingerprint the WHOLE `.gitignore` chain — which files exist and what is in them.

    `_WALK_FIELDS` records `respect_gitignore` as a bool, which says nothing about what
    was actually ignored. Editing ignore rules alone therefore left
    `missing_is_trustworthy` at True while files legitimately dropped out of the walk, and
    a future `--prune` reading that would delete embeddings for files still on disk. The
    scale is not hypothetical: one line in `workspace-vscode/.gitignore` excludes a 2.6 GB
    `.vscode-test/` bundle that was 74% of this project's own index until v3.1.2 read
    nested chains (walker.py:9-13). The ignore chain is the largest single lever on which
    files are indexable, and it was the one thing the fingerprint did not look at.

    Discovery and parsing are both borrowed from `FileWalker` rather than re-globbing.
    `_iter_files` already prunes ignored directories, enforces `follow_symlinks`
    containment and breaks symlink cycles, so this sees exactly the `.gitignore` files
    that `_gitignore_chain` can give authority to — a `rglob(".gitignore")` would descend
    into `node_modules` and fingerprint files the walk never consults. `_spec_for_dir`
    supplies the parse (and its cache).

    Stability matters as much as sensitivity: a digest that varies between runs reports
    every drift check as untrustworthy, which disables the feature as effectively as the
    false TRUE it replaces. Hence entries are sorted, anchored on repo-RELATIVE directory
    paths (so a checkout and its copy agree), and built from pathspec's parsed patterns
    rather than raw bytes, which drops blank lines and normalises line endings.
    """
    walker_config = getattr(config, "walker", None)
    if walker_config is None or not getattr(walker_config, "respect_gitignore", False):
        # With the chain switched off its contents cannot change the walk, so digesting
        # them would raise an untrustworthy verdict with no consequence behind it — and a
        # warning that cries wolf trains users past the one that matters.
        return None

    from trelix.indexing.walker import FileWalker

    root = Path(config.repo_path)
    entries: list[str] = []
    anchors: list[str] = []
    try:
        walker = FileWalker(config)
        for path in walker._iter_files(root):
            if path.name != ".gitignore":
                continue
            spec = walker._spec_for_dir(path.parent)
            if spec is None:
                # No spec means unreadable (`_spec_for_dir` swallows OSError), and the
                # walk treats it as absent, so the digest must too or the two disagree.
                continue
            anchor = path.parent.relative_to(root).as_posix()
            patterns = "\n".join(str(getattr(p, "pattern", p)) for p in spec.patterns)
            entries.append(f"{anchor}\n{patterns}")
            anchors.append(anchor)
    except Exception as exc:
        # Runs inside `capture_provenance`, which must never fail an index — the
        # embeddings are the expensive artefact. Returning None costs one stats line and
        # errs toward "cannot verify", never toward false confidence.
        logger.debug("Could not digest the .gitignore chain under %s: %s", root, exc)
        return None

    entries.sort()
    digest = hashlib.sha256("\0".join(entries).encode("utf-8")).hexdigest()
    # Count, MEMBERSHIP and scheme are carried in the value, not just the hash, so the diff
    # table names what moved instead of showing two opaque hex strings. The count alone was
    # not enough: a `.trelix.bak-*` backup adds a member without changing anything a user
    # edited, and a swap of one chain file for another leaves the count untouched.
    anchors.sort()
    shown = ", ".join(anchors[:_GITIGNORE_MEMBERS_SHOWN])
    if len(anchors) > _GITIGNORE_MEMBERS_SHOWN:
        shown += f", +{len(anchors) - _GITIGNORE_MEMBERS_SHOWN} more"
    return f"{_DIGEST_SCHEME}:{len(entries)} file(s) [{shown}], sha256:{digest[:16]}"


def _walk_config_json(config: IndexConfig) -> str | None:
    """Serialise the walk-determining settings, order-insensitively.

    Collections are sorted because the walker turns each into a set — two configs that
    list the same ignores in a different order produce identical walks and must not be
    reported as a mismatch.
    """
    import json

    walker = getattr(config, "walker", None)
    if walker is None:
        return None

    payload: dict[str, object] = {}
    for name in _WALK_FIELDS:
        if not hasattr(walker, name):
            continue
        value = getattr(walker, name)
        payload[name] = (
            sorted(str(v) for v in value) if isinstance(value, list | set | tuple) else value
        )

    payload[_GITIGNORE_KEY] = _gitignore_digest(config)

    try:
        return json.dumps(payload, sort_keys=True)
    except (TypeError, ValueError) as exc:
        logger.debug("Could not serialise walk config: %s", exc)
        return None


def is_walk_config_comparable(provenance: IndexProvenance) -> bool:
    """Whether the recorded walk config can be compared against one captured now.

    False for an index that recorded none, and false for one whose `.gitignore` digest was
    computed by an older scheme — those two digests are different arithmetic over the same
    tree, so comparing them reports a change no user made. "Cannot compare" is the honest
    verdict and it fails the same way (`missing_is_trustworthy` False, `--prune` refuses);
    the difference is that it does not train users past the warning.

    A record with no digest at all (`respect_gitignore=False`) stays comparable: with the
    chain switched off, every scheme records the same nothing.
    """
    import json

    if not provenance.walk_config:
        return False

    try:
        recorded = json.loads(provenance.walk_config)
    except (TypeError, ValueError) as exc:
        logger.debug("Recorded walk config is not readable JSON: %s", exc)
        return False

    chain = recorded.get(_GITIGNORE_KEY) if isinstance(recorded, dict) else None
    if isinstance(chain, str) and not chain.startswith(f"{_DIGEST_SCHEME}:"):
        return False
    return True


def walk_config_differences(
    provenance: IndexProvenance, config: IndexConfig
) -> dict[str, tuple[object, object]]:
    """Settings that differ between index time and now, as {name: (recorded, current)}.

    An empty dict means either "identical" or "cannot tell" — the caller distinguishes
    those via `is_walk_config_comparable`, because an index predating this field, or one
    whose digest scheme predates this build, cannot be compared and must not be reported
    as either matching or changed.
    """
    import json

    if not is_walk_config_comparable(provenance) or not provenance.walk_config:
        return {}

    current_raw = _walk_config_json(config)
    if current_raw is None:
        return {}

    try:
        recorded = json.loads(provenance.walk_config)
        current = json.loads(current_raw)
    except (TypeError, ValueError) as exc:
        logger.debug("Could not compare walk configs: %s", exc)
        return {}

    return {
        key: (recorded.get(key), current.get(key))
        for key in set(recorded) | set(current)
        if recorded.get(key) != current.get(key)
    }


def capture_provenance(config: IndexConfig) -> IndexProvenance:
    """Read the current git and embedder state. Never raises."""
    from trelix import __version__

    repo_path = Path(config.repo_path)

    commit = _git(repo_path, "rev-parse", "HEAD")
    branch = _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")

    # `git status --porcelain` already honours .gitignore, so an ignored build directory
    # does not read as a dirty worktree. Distinguished from "unknown": an empty string is
    # a clean tree, None is a failed or non-git call.
    status = _git(repo_path, "status", "--porcelain")
    dirty = None if status is None else bool(status)

    return IndexProvenance(
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        indexed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        trelix_version=__version__,
        embedder_provider=str(getattr(config.embedder, "provider", "") or "") or None,
        embedder_model=str(getattr(config.embedder, "model", "") or "") or None,
        walk_config=_walk_config_json(config),
    )


# Key under which the accumulated set of walk-config digests lives. Namespaced under
# `_PREFIX` with the rest, but not one of `IndexProvenance`'s written fields: it is derived
# from what is already in the database plus the incoming record, so it is computed here.
_HISTORY_KEY = "walk_config_digests"

# Stands in for a walk config that cannot be identified — a run that recorded none, or rows
# written before history was recorded at all. A distinct member rather than an omission,
# because "unknown" must not be silently unified with the config that happens to be current:
# that is exactly how a single re-index would launder a pre-fix index into a prunable one.
_UNKNOWN_DIGEST = "unrecorded"


def _walk_config_digest(walk_config: str | None) -> str:
    """Short, stable identity for one walk config. Full text is already stored separately."""
    if walk_config is None:
        return _UNKNOWN_DIGEST
    return hashlib.sha256(walk_config.encode("utf-8")).hexdigest()[:16]


def _extend_walk_config_history(db: Database, provenance: IndexProvenance) -> None:
    """Add this run's walk config to the set of configs that have written this index.

    Monotonic: members are only ever added. The set is a claim about ROWS, and a row written
    under a walk config that is no longer current does not stop existing because a later run
    used a different one — which is precisely the mistake that let `--prune` propose deleting
    12 files that were present on disk with zero refusals.

    MUST be called before the fields are overwritten, because it reads the previous
    `walk_config` row to detect an index that predates history recording.
    """
    import json

    stored = db.get_index_metadata(_PREFIX + _HISTORY_KEY)
    known: set[str] = set()
    if stored:
        try:
            known = {str(d) for d in json.loads(stored)}
        except (TypeError, ValueError):
            # A truncated or hand-edited row is not evidence of anything, and treating it as
            # an empty history would read as "one config wrote everything".
            known = {_UNKNOWN_DIGEST}
    elif db.get_index_metadata(_PREFIX + "walk_config") is not None:
        # An index written by a trelix that recorded no history. How many walks wrote its
        # rows is unknowable from here, so record the ignorance instead of asserting "one".
        known.add(_UNKNOWN_DIGEST)

    known.add(_walk_config_digest(provenance.walk_config))
    db.set_index_metadata(_PREFIX + _HISTORY_KEY, json.dumps(sorted(known)))


def write_provenance(db: Database, provenance: IndexProvenance) -> None:
    """Persist provenance to `index_metadata`. Never raises.

    Called at the end of an index run. A failure here must not fail an index that has
    otherwise succeeded — the embeddings are the expensive artefact and they are already
    committed by this point; losing the provenance row costs a `trelix stats` line.
    """
    try:
        # First, and reading the row this call is about to replace.
        _extend_walk_config_history(db, provenance)
    except Exception as exc:
        logger.warning(
            "Could not record the walk-config history (--prune will refuse until an index "
            "run records it): %s",
            exc,
        )

    fields = {
        "git_commit": provenance.git_commit,
        "git_branch": provenance.git_branch,
        "git_dirty": None if provenance.git_dirty is None else str(provenance.git_dirty).lower(),
        "indexed_at": provenance.indexed_at,
        "trelix_version": provenance.trelix_version,
        "embedder_provider": provenance.embedder_provider,
        "embedder_model": provenance.embedder_model,
        "walk_config": provenance.walk_config,
    }
    try:
        for name, value in fields.items():
            if value is None:
                # Deleted rather than stored as "None": a re-index from a directory that
                # is no longer a git repo must not leave the previous run's commit behind
                # to be read as current.
                db.delete_index_metadata(_PREFIX + name)
            else:
                db.set_index_metadata(_PREFIX + name, value)
    except Exception as exc:
        logger.warning(
            "Could not record index provenance (the index itself is unaffected): %s", exc
        )


def read_provenance(db: Database) -> IndexProvenance:
    """Load provenance. Returns an all-None record for an index that predates it."""
    try:
        stored = db.get_index_metadata_with_prefix(_PREFIX)
    except Exception as exc:
        logger.warning("Could not read index provenance: %s", exc)
        return IndexProvenance()

    dirty_raw = stored.get("git_dirty")
    return IndexProvenance(
        git_commit=stored.get("git_commit"),
        git_branch=stored.get("git_branch"),
        git_dirty=None if dirty_raw is None else dirty_raw == "true",
        indexed_at=stored.get("indexed_at"),
        trelix_version=stored.get("trelix_version"),
        embedder_provider=stored.get("embedder_provider"),
        embedder_model=stored.get("embedder_model"),
        walk_config=stored.get("walk_config"),
        walk_config_history=_parse_history(stored.get(_HISTORY_KEY)),
    )


def _parse_history(raw: str | None) -> tuple[str, ...]:
    """Read the accumulated digest set. Degrades to "unknown", never to "one".

    An unreadable row is not evidence that a single walk config wrote every row, and this
    runs on the path of every `trelix stats`, so it must not raise either.
    """
    import json

    if not raw:
        return ()
    try:
        return tuple(sorted(str(d) for d in json.loads(raw)))
    except (TypeError, ValueError) as exc:
        logger.debug("Walk-config history is not readable JSON: %s", exc)
        return (_UNKNOWN_DIGEST,)


def commits_since(config: IndexConfig, indexed_commit: str | None) -> int | None:
    """How many commits HEAD is ahead of `indexed_commit`.

    None when it cannot be determined, which is not the same as 0 and must not be
    rendered as such: the recorded commit may have been rebased away or garbage
    collected, or the directory may no longer be a git repo.
    """
    if not indexed_commit:
        return None
    count = _git(Path(config.repo_path), "rev-list", "--count", f"{indexed_commit}..HEAD")
    if count is None:
        return None
    try:
        return int(count)
    except ValueError:
        return None


def compute_drift(
    config: IndexConfig, db: Database, *, provenance: IndexProvenance | None = None
) -> DriftReport:
    """Compare every indexable file on disk against its stored hash.

    Costs a full walk plus a SHA-256 of every file — on this repository ~470 files, well
    under a second, but it is proportional to repo size and so is opt-in at the CLI.

    `provenance` overrides the record this reads from `db`, and a caller that runs an INDEX
    before a drift check must supply it. `Indexer.index()` writes provenance at the end of
    the run, so a record read here afterwards describes the run that just finished — and
    `walk_config_comparable` / `walk_config_diff` then compare the current walk config
    against a copy of itself and can only ever come back "unchanged".

    That is not hypothetical: it disarmed the `--prune` guard. Reproduced with a real
    Indexer over a temp repo — index three files, re-run with one file edited and
    `vendored/` newly ignored, and `walk_config_differences(pre_run_record, config)`
    correctly returns `['extra_ignore_dirs']` while this function returned
    `walk_config_diff = ()`. Two of the six prune guards read the report rather than the
    provenance, so threading a pre-run snapshot into `plan_prune` alone was not enough.
    """
    from trelix.indexing.walker import FileWalker

    walker = FileWalker(config)

    stale: list[str] = []
    new: list[str] = []
    seen: set[str] = set()
    unchanged = 0

    for file in walker.walk():
        seen.add(file.rel_path)
        stored_hash = db.get_file_hash(file.rel_path)
        if stored_hash is None:
            new.append(file.rel_path)
        elif stored_hash != file.hash:
            stale.append(file.rel_path)
        else:
            unchanged += 1

    # Indexed paths the walk did not yield. Genuinely deleted files land here, but so do
    # files inside a directory the walk could not read, and files a changed ignore rule
    # now excludes. The two flags below are what let a caller tell those apart.
    missing = sorted(set(db.get_all_file_rel_paths()) - seen)

    # `provenance if provenance is not None` rather than `provenance or ...`: an all-None
    # IndexProvenance is falsy-looking but meaningful — it says "this index recorded
    # nothing", which must stay distinguishable from "the caller did not pass one".
    record = provenance if provenance is not None else read_provenance(db)
    diff = walk_config_differences(record, config)

    return DriftReport(
        walk_config_comparable=is_walk_config_comparable(record),
        # One member means every row in this index was written under the same walk config.
        # Zero means nothing recorded it, which is not evidence that they were.
        rows_share_one_walk_config=len(record.walk_config_history) == 1,
        walk_config_diff=tuple(
            (name, repr(recorded), repr(current))
            for name, (recorded, current) in sorted(diff.items())
        ),
        stale=tuple(sorted(stale)),
        new=tuple(sorted(new)),
        missing=tuple(missing),
        unchanged_count=unchanged,
        walk_was_complete=walker.walk_was_complete,
        incomplete_paths=tuple(walker.incomplete_paths),
    )


# ---------------------------------------------------------------------------
# Prune planning
# ---------------------------------------------------------------------------

# Share of the index one prune may remove before the size of the request is itself read as
# evidence that the walk is wrong, rather than that the repository lost that many files.
#
# Chosen against the two measured shapes on this repository, which sit on either side of
# it. The false-deletion event the guards above exist for was 35 of 467 indexed files
# (7.5%) from `extra_ignore_dirs` alone — under this cap, and deliberately so: that case is
# refused by `walk_config_changed`, and tuning the cap down to catch it would make the cap
# fire on ordinary refactors instead. What the cap is for is the catastrophic shape, where
# a whole subtree stops being reachable at once: one line in `workspace-vscode/.gitignore`
# governs 74% of this index (see `_gitignore_digest`), and a walk rooted one directory too
# deep reaches nothing at all and so proposes deleting 100%. Those are an order of
# magnitude above 10%, and a legitimate prune is normally a handful of files.
_PRUNE_MAX_FRACTION_DEFAULT = 0.10

# A percentage alone makes a small index unprunable: 10% of a 20-file repo is 2 files, and
# deleting 2 files is an ordinary commit. Below this many candidates the percentage does
# not apply — at this size the dry-run has already printed the complete list, and a human
# can read ten paths before typing --yes, which is a better check than any ratio.
_PRUNE_MIN_CANDIDATES_FOR_CAP = 10


@dataclass(frozen=True)
class PrunePlan:
    """The files a prune would delete, and every reason it must not run.

    Refusals are sentences rather than an enum because each one carries its own remedy:
    "re-index so provenance exists" and "re-run with the environment that built the index"
    are different actions, and a caller holding only a boolean would have to reinvent the
    mapping to say anything a user could act on. A prune that reports only "refused" gets
    re-run with more force.

    An empty `refusals` is a licence to delete paid-for embeddings, so nothing but
    `plan_prune` should ever construct one of these.
    """

    candidates: tuple[str, ...] = ()
    indexed_count: int = 0
    refusals: tuple[str, ...] = ()
    max_fraction: float = _PRUNE_MAX_FRACTION_DEFAULT

    @property
    def is_refused(self) -> bool:
        return bool(self.refusals)

    @property
    def fraction_of_index(self) -> float:
        """Share of the index `candidates` represents; 0.0 when nothing is indexed."""
        if self.indexed_count <= 0:
            return 0.0
        return len(self.candidates) / self.indexed_count


def plan_prune(
    report: DriftReport,
    provenance: IndexProvenance,
    indexed_count: int,
    *,
    max_fraction: float = _PRUNE_MAX_FRACTION_DEFAULT,
) -> PrunePlan:
    """Decide whether `report.missing` may be deleted, and refuse in the user's words if not.

    Pure by design: it takes an already-computed `DriftReport` instead of computing one, so
    a prune costs exactly one walk and every refusal is testable without a repository.

    The licence to delete is a conjunction of five independent facts, each of which has
    made `missing` lie in production:

    * `walk_was_complete` — an unreadable directory drops its whole subtree, and every
      file under it then reads as deleted while being present.
    * `walk_config_comparable` — an index with no recorded walk config, or one recorded
      under an older digest scheme, cannot be compared at all, and "cannot tell" must not
      resolve to "same".
    * not `walk_config_changed` — measured at 35 phantom deletions here from
      `extra_ignore_dirs`, which REPLACES rather than extends the default list.
    * `provenance.trelix_version == __version__` — the version is checked separately from
      the other three because an older trelix can have walked differently in ways no
      `_WALK_FIELDS` entry describes: v3.1.2 started reading nested `.gitignore` chains and
      moved this project's own index by 74% without a single config field changing.
    * exactly one member in `provenance.walk_config_history` — the four above compare the
      walk config of ONE run against now, and a prune acts per FILE. Rows written by an
      earlier run under a wider walk survive a re-index that narrows it (a re-index deletes
      nothing), while the record is overwritten with the narrow config. The three
      walk-config conditions then all pass on a comparison of the narrow config with itself,
      and the surviving rows are proposed for deletion. Measured by
      `scratch-pad/audit_prune_guard_probe.py`: 12 candidates, all 12 on disk, 9.1% of the
      index, and ZERO refusals — the scenario the refusals themselves prescribe
      ("re-index, then prune").

    The first three are exactly `DriftReport.missing_is_trustworthy`; it is re-derived
    field by field here so a refusal can name which one failed.
    """
    from trelix import __version__

    refusals: list[str] = []

    if not report.walk_was_complete:
        shown = ", ".join(report.incomplete_paths[:5])
        refusals.append(
            f"{len(report.incomplete_paths)} path(s) could not be read during the walk "
            f"({shown}{' …' if len(report.incomplete_paths) > 5 else ''}), so every file "
            "under them looks deleted while being present. Fix the permission (or the "
            "broken symlink) and re-run."
        )

    if not report.walk_config_comparable:
        refusals.append(
            "this index records no walk config, so whether this walk used the same ignore "
            "rules that built it cannot be checked — and 'cannot tell' is not 'the same'. "
            "A full `trelix index <repo>` writes provenance; prune after that."
        )
    elif report.walk_config_changed:
        names = ", ".join(name for name, _recorded, _current in report.walk_config_diff)
        message = (
            f"the walk settings changed since this index was built ({names}), so files the "
            "walk no longer reaches are indistinguishable from files that were deleted. "
            "Re-run with the environment that built the index — `scripts/self-index.sh` is "
            "the reference — then prune."
        )
        # Naming the environment is not enough when the chain itself moved: the walk can
        # narrow with no env change at all, because `cp -a .trelix .trelix.bak-<name>` — this
        # project's own documented backup step — puts a copy of `.trelix/.gitignore` into the
        # walk. Both values name their members, so the offending directory is readable here
        # instead of the user being sent to reproduce an environment that was never wrong.
        chain = [(rec, cur) for name, rec, cur in report.walk_config_diff if name == _GITIGNORE_KEY]
        if chain:
            recorded, current = chain[0]
            message += (
                f" The `.gitignore` chain itself differs (recorded {recorded}, now "
                f"{current}) — a copy of `.trelix`, e.g. a `.trelix.bak-*` backup, adds a "
                "member to it. Delete or rename it, or re-index."
            )
        refusals.append(message)

    # `walk_config` describes one run; `walk_config_history` describes the rows. The three
    # conditions above cannot see a row written by a run whose record has since been
    # overwritten, and that row is exactly what a prune would delete.
    if len(provenance.walk_config_history) > 1:
        digests = ", ".join(provenance.walk_config_history)
        refusals.append(
            f"this index's rows were written under {len(provenance.walk_config_history)} "
            f"different walk configurations (digests {digests}), so rows written under a "
            "wider walk are indistinguishable from files that were deleted — the recorded "
            "walk config describes only the run that wrote it, and a re-index deletes no "
            "rows. Rebuild the index from one walk config (delete "
            "`<repo>/.trelix/index.db`, then `trelix index <repo>`), then prune."
        )
    elif not provenance.walk_config_history:
        refusals.append(
            "this index does not record which walk configurations wrote its rows, so "
            "whether every row was written under the walk config being compared cannot be "
            "established — one recorded walk config describes the LAST run only. A full "
            "`trelix index <repo>` with this version records it; an index built before it "
            "did must be rebuilt (delete `<repo>/.trelix/index.db`) before its rows can be "
            "certified."
        )

    if provenance.trelix_version != __version__:
        refusals.append(
            f"this index was built by trelix {provenance.trelix_version or '(unrecorded)'} "
            f"and this is {__version__}; a different version can walk differently in ways "
            "the recorded settings do not describe. Re-index with this version, then prune."
        )

    # Last, and independent of the four above: even a fully-verified walk does not license
    # a prune of this shape without someone having read the list.
    if (
        len(report.missing) > _PRUNE_MIN_CANDIDATES_FOR_CAP
        and indexed_count > 0
        and len(report.missing) > max_fraction * indexed_count
    ):
        share = len(report.missing) / indexed_count
        refusals.append(
            f"{len(report.missing)} of {indexed_count} indexed files ({share:.1%}) would be "
            f"removed, over the {max_fraction:.0%} cap. That is the shape of a walk that "
            "lost a subtree, not of a repository that lost that many files. Read the list "
            "above; if those files are genuinely gone, re-run with an explicit higher cap "
            "(`--prune-max-percent`)."
        )

    return PrunePlan(
        candidates=report.missing,
        indexed_count=indexed_count,
        refusals=tuple(refusals),
        max_fraction=max_fraction,
    )
