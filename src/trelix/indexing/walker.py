"""
FileWalker: discovers files in a repo, detects language, filters noise.

Key decisions stolen from continue.dev:
- pathspec for .gitignore-aware walking
- Comprehensive default ignore list for dirs and extensions
- File hash (SHA-256) computed here so the indexer can skip unchanged files

Nested `.gitignore` files are honoured (see `_is_gitignored`): each one's patterns are
matched relative to its own directory, and the file closest to a path decides its fate.
Until v3.1.2 only `repo_root/.gitignore` was read despite this docstring claiming
otherwise, which is how a 2.6 GB `.vscode-test/` bundle — excluded by
`workspace-vscode/.gitignore` — ended up as 74% of this project's own index.

Directory exclusion has two tiers (see `_CONDITIONAL_IGNORE_DIRS`): most
`extra_ignore_dirs` entries are unconditional, while `packages` and `bin` are ignored
only when the directory beside them fails to prove it holds first-party source.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import pathspec

from trelix.core.config import IndexConfig
from trelix.core.models import IndexedFile, Language

logger = logging.getLogger("trelix.indexing.walker")

# Names in `extra_ignore_dirs` that are only *sometimes* build output.
#
# `bin`, `obj` and `packages` were added together for .NET (NuGet restores into
# `packages/`, MSBuild writes `bin/` and `obj/`). Two of those three names are also
# first-party SOURCE directories elsewhere: every pnpm/npm/yarn/lerna/turborepo monorepo
# keeps its own code under `packages/`, and a Node CLI keeps its real executables in
# `bin/`. Because `extra_ignore_dirs` is enforced during traversal, the walk never descends
# and the index simply contains none of it — while the run still reports `errors: 0`.
#
# Measured across six repositories in one workspace. Every figure here is a WALK-UNIT count —
# files a bare `FileWalker.walk()` yields, i.e. what trelix would actually index after the
# language, size, filename and `.gitignore` filters — because that is the unit the controls
# beside it are in, and the only unit a Stage-2 plan can be sized from:
#
#   repo          dir       hidden   default walk -> Stage-2 walk
#   Graph-Forge   packages      36     598 -> 634   (+6%)
#   CommandVault  packages     168      31 -> 199
#   ContextOS     packages     137     200 -> 337
#   Tombstone     packages     104     510 -> 614
#   MindForge     bin          189    2765 -> 2954  (183 of the 189 are `.js`)
#
# Graph-Forge was previously published here as "584 source files", an on-disk count in a
# different unit from the controls printed beside it — and 16x the number a plan needs. Its
# `packages/` holds 909 files on disk, but 830 of those are a vendored `.venv` or
# `node_modules` that trelix excludes twice over (both are in `extra_ignore_dirs` AND
# gitignored, via the nested `.gitignore` support added in v3.1.2); 558 of its 569 `.py` files
# are one virtualenv. The controls ARE walk-units and do reproduce exactly: `services/` 265,
# `apps/` 125, `proto/` 34 and `docs/` 51 are indexed, so the exclusion is the only
# difference. CommandVault finishes at 31 of 212 tracked files with exactly ONE file in a code
# language (typescript — the other 30 are 13 yaml, 10 markdown, 7 json) and 0 call edges.
#
# `obj` is NOT here: it is genuinely .NET-only. Dropping it from the list re-admits 0 files in
# all six repos — but only because none of them HAS a reachable .NET `obj/`. The workspace's
# only two `obj` directories are a gitignored Go toolchain cache and one inside
# `node_modules`, so that 0 means "untested here", not "measured harmless".
#
# The sixth repository is trelix itself, and it is the honest limit of requiring proof: its own
# `packages/` (four SDK packages, 41 files on disk) has no root workspace manifest, so the
# probe finds no evidence, nothing is reported, and Stage-2 would not widen into it either.
_CONDITIONAL_IGNORE_DIRS = frozenset({"packages", "bin"})

# Files whose mere PRESENCE beside a `packages/` directory proves a JS workspace, read for
# free out of the parent's already-materialised `iterdir()` listing.
#
# Both this and the `workspaces` key below are load-bearing, measured: CommandVault has no
# `workspaces` key (its `pnpm-workspace.yaml` is the only signal) while Graph-Forge, ContextOS
# and Tombstone — three of the four, not two — have no marker file at all and are carried
# entirely by the key. A single-branch probe misses most of the measured population.
# `turbo.json` is deliberately absent — it appears in non-workspace repos too.
_WORKSPACE_MARKER_FILES = frozenset(
    {
        "pnpm-workspace.yaml",
        "pnpm-workspace.yml",
        "lerna.json",
        "nx.json",
        "rush.json",
    }
)

# .NET evidence. Present beside a conditional directory, it wins any tie and the directory
# stays ignored — the cheap, safe direction, since re-including a NuGet `packages/` tree is
# thousands of files of third-party code priced per token.
#
# Stored LOWERCASED and compared against `name.lower()`, exactly as the suffix tuple already
# was. MSBuild and NuGet resolve these names case-insensitively, and .NET is developed on
# Windows and macOS where the filesystem does too — so `Directory.build.props` and
# `Packages.config` are the same files to the toolchain, and an exact-case comparison let a
# real solution through the veto. Measured on a 25-package restore beside an npm-workspace
# `package.json` (the ASP.NET-plus-SPA shape, where the veto is the only thing holding the
# restore out): `Directory.Build.props` warn=0 junk=+0, but `Directory.build.props`,
# `directory.packages.props` and `Packages.config` each warn=1 junk=+50.
#
# `NuGet.config` is here because it is the file that declares `repositoryPath` and therefore
# MAKES a root `packages/` a restore directory — the single most direct proof of the shape
# these entries exist for, and it was not a marker at any casing (warn=1 junk=+50).
# `packages.lock.json` and `Directory.Build.targets` measured the same way, as did `.slnf`
# (a solution filter) and `.vcxproj` (C++ MSBuild) against the suffix tuple.
_DOTNET_MARKER_FILES = frozenset(
    {
        "directory.build.props",
        "directory.build.targets",
        "directory.packages.props",
        "packages.config",
        "packages.lock.json",
        "nuget.config",
    }
)
_DOTNET_MARKER_SUFFIXES = (
    ".sln",
    ".slnx",
    ".slnf",
    ".csproj",
    ".fsproj",
    ".vbproj",
    ".vcxproj",
)

# A conditional directory is NEVER reclassified when one of these is an ancestor segment.
# Package stores contain complete copies of other projects — including their workspace
# manifests — so evidence found inside one proves nothing about first-party source and
# would re-admit the exact duplicate-content shape that nested `.gitignore` support was
# added to remove in v3.1.2. `.pnpm-store` in particular is NOT in `extra_ignore_dirs`, so
# the walk really does reach inside it; this rule is the only thing stopping it.
_STORE_PATH_SEGMENTS = frozenset({"node_modules", ".pnpm-store", ".yarn", ".npm", ".pnp", ".git"})

# Map file extension → Language
EXTENSION_MAP: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TSX,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".c": Language.C,
    ".h": Language.C,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".razor": Language.RAZOR,
    ".cshtml": Language.CSHTML,
    ".csproj": Language.CSPROJ,
    ".fsproj": Language.CSPROJ,
    ".vbproj": Language.CSPROJ,
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
    ".rb": Language.RUBY,
    ".md": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
    ".json": Language.JSON,
    ".yaml": Language.YAML,
    ".yml": Language.YAML,
    ".toml": Language.TOML,
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".jsx": Language.JAVASCRIPT,
    ".css": Language.CSS,
    ".scss": Language.CSS,
    ".sass": Language.CSS,
    ".less": Language.CSS,
    # Ops and contract artifacts
    ".sh": Language.SHELL,
    ".bash": Language.SHELL,
    ".zsh": Language.SHELL,
    ".sql": Language.SQL,
    ".proto": Language.PROTO,
    ".mk": Language.MAKE,
}

# Artifacts identified by FILENAME, not extension. `Path("Dockerfile").suffix` is "", so
# no EXTENSION_MAP entry can ever reach one — which is why a Dockerfile, a Makefile and
# every shell entrypoint were absent from the index entirely rather than merely ranked
# badly. Matched case-insensitively on the full name first, then on the stem, so
# `Dockerfile.prod` and `Makefile.local` resolve too.
FILENAME_MAP: dict[str, Language] = {
    "dockerfile": Language.DOCKERFILE,
    "containerfile": Language.DOCKERFILE,
    "makefile": Language.MAKE,
    "gnumakefile": Language.MAKE,
}


def detect_language(path: Path) -> Language:
    """Resolve a path to a Language, filename first and extension second.

    The single detector for every call site. There were four independent copies of
    `EXTENSION_MAP.get(suffix)` — in the walker, the watcher (twice), the REST /parse
    route and the indexer's single-file path — so a new artifact type had to be added in
    five places or the surfaces would silently disagree about what is indexable.
    """
    name = path.name.lower()
    if name in FILENAME_MAP:
        return FILENAME_MAP[name]
    # "Dockerfile.prod" -> stem "Dockerfile"; also catches "Makefile.local".
    stem = name.split(".", 1)[0]
    if stem in FILENAME_MAP:
        return FILENAME_MAP[stem]
    return EXTENSION_MAP.get(path.suffix.lower(), Language.UNKNOWN)


class FileWalker:
    """
    Walks a repository directory and yields IndexedFile objects for every
    file that passes language + size + ignore filters.

    Usage:
        walker = FileWalker(config)
        for file in walker.walk():
            ...
    """

    def __init__(self, config: IndexConfig, *, index_conditional_dirs: bool = False) -> None:
        """`index_conditional_dirs=False` walks exactly what every existing index walked.

        With it False the conditional tier is REPORT-ONLY: the probe runs, a reclassified
        directory is named at WARNING, and the walk yields the same files it always did. So
        this release costs nothing — no extra embedding, and no change to the walk-config
        fingerprint that `--prune` compares (`_WALK_FIELDS` in store/provenance.py), which
        is why it needs no migration.

        With it True a conditional directory that PROVES it holds first-party source is
        walked. It is deliberately not reachable from the CLI or the environment yet, and
        that is the point: it is not a `WalkerConfig` field, so provenance cannot record it,
        and an index built with it True would contain rows a later default walk cannot reach
        — which `compute_drift` reads as deletions and `--prune` reads as candidates. It
        becomes the DEFAULT in the release that also splits the field and ships the history
        collapse, so the widening is recorded, comparable and refusable in one move.
        """
        self.config = config
        self.repo_root = Path(config.repo_path)
        # Resolved once, and only consulted when containment is enabled.
        #
        # IndexConfig's repo_path validator (core/config.py) already returns a
        # resolved path, so for a normally-constructed config this is a no-op. It
        # is load-bearing for the two call sites that build the config with
        # `IndexConfig.model_construct(repo_path=...)` — federation/retriever.py
        # and indexing/multi_watcher.py — because model_construct bypasses
        # validators entirely, leaving repo_path exactly as the caller wrote it.
        # A containment check against an unresolved root there would resolve every
        # entry to its real location, compare it against a symlinked root, and
        # reject the whole repository.
        self._resolved_root = self.repo_root.resolve()
        # directory -> the .gitignore living *directly* in it, parsed once.
        # Populated lazily: a repo's .gitignore files are only read if the walk (or a
        # watcher event) actually reaches the directory holding them.
        self._spec_cache: dict[Path, pathspec.PathSpec | None] = {}  # type: ignore[type-arg]
        # Paths the traversal could not read. An unreadable directory drops its whole
        # subtree, and an unreadable file drops itself; both used to happen in total
        # silence, so the walk simply returned fewer files and every caller treated the
        # result as the complete contents of the repository.
        self._incomplete_paths: list[str] = []

        self._index_conditional_dirs = index_conditional_dirs
        # Only names the effective list actually ignores are conditional. A user who
        # dropped `packages` from their override (as `scripts/self-index.sh` does) has
        # already said "index it" — the probe must not put it back under a rule.
        self._conditional_names = _CONDITIONAL_IGNORE_DIRS & {
            entry.lower() for entry in config.walker.extra_ignore_dirs
        }
        # parent directory -> {conditional dir name (lowercased): evidence filename}.
        # Shaped exactly like `_spec_cache` above and populated the same way: once per
        # parent, from the entry listing `_iter_files` already has. A directory absent from
        # the inner dict has no evidence and stays ignored.
        self._conditional_cache: dict[Path, dict[str, str]] = {}
        # Reclassified directories already reported, so the warning fires once per
        # directory rather than once per verdict (`_is_ignored_dir` is also called by both
        # watchers' ancestor loops, once per event, for the same path).
        self._reported_conditional: set[str] = set()

    @property
    def incomplete_paths(self) -> list[str]:
        """Paths the most recent walk could not read, as repo-relative strings."""
        return list(self._incomplete_paths)

    @property
    def walk_was_complete(self) -> bool:
        """False when the last walk skipped anything it could not read.

        Anything that DELETES index rows for files the walk did not yield must check
        this first: a truncated walk is indistinguishable from a repository whose files
        were removed, and acting on it destroys embeddings that cost money to compute.
        """
        return not self._incomplete_paths

    def _record_incomplete(self, path: Path, exc: OSError | RuntimeError) -> None:
        """Note a path the traversal had to skip, and say so once at WARNING."""
        try:
            rel = str(path.relative_to(self.repo_root))
        except ValueError:
            rel = str(path)
        self._incomplete_paths.append(rel)
        logger.warning(
            "Skipped %s while walking the repository (%s) — the index will be "
            "incomplete for this path",
            rel,
            exc.__class__.__name__,
        )

    def _spec_for_dir(self, directory: Path) -> pathspec.PathSpec | None:  # type: ignore[type-arg]
        """Parse and cache the `.gitignore` sitting directly inside `directory`."""
        if directory in self._spec_cache:
            return self._spec_cache[directory]

        spec: pathspec.PathSpec | None = None  # type: ignore[type-arg]
        gitignore_path = directory / ".gitignore"
        try:
            if gitignore_path.is_file():
                patterns = gitignore_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                spec = pathspec.PathSpec.from_lines("gitignore", patterns)
        except OSError:
            # Unreadable .gitignore (permissions, races during a watch). Treating it as
            # absent keeps the walk going; the alternative is aborting an entire index
            # over one file we were not able to read.
            spec = None

        self._spec_cache[directory] = spec
        return spec

    def _gitignore_chain(self, path: Path) -> list[tuple[Path, pathspec.PathSpec]]:  # type: ignore[type-arg]
        """`(anchor_dir, spec)` pairs governing `path`, ordered shallowest → deepest.

        Every `.gitignore` from the repo root down to `path`'s own directory, skipping
        directories that do not contain one.
        """
        try:
            rel = path.relative_to(self.repo_root)
        except ValueError:
            # Outside the repo entirely — no .gitignore of ours has authority over it.
            return []

        # repo_root first, then each intermediate directory down to path's parent.
        directories = [self.repo_root]
        current = self.repo_root
        for part in rel.parts[:-1]:
            current = current / part
            directories.append(current)

        chain: list[tuple[Path, pathspec.PathSpec]] = []  # type: ignore[type-arg]
        for directory in directories:
            spec = self._spec_for_dir(directory)
            if spec is not None:
                chain.append((directory, spec))
        return chain

    def _is_gitignored(self, path: Path, *, is_dir: bool) -> bool:
        """Apply the full nested-`.gitignore` chain to `path`.

        Two details make this match git rather than merely approximate it:

        1. Each `.gitignore`'s patterns are matched against the path *relative to that
           file's own directory*. Anchored patterns (`/rooted.py`) and directory patterns
           (`harness/`) are meaningless otherwise — a repo-root-relative path would make
           `/rooted.py` in `sub/.gitignore` silently match nothing.
        2. Proximity wins. Walking shallowest → deepest and letting each *explicit*
           verdict overwrite the previous one means a deeper `!keep.log` re-includes a
           file its parent excluded, while an unmentioned path (`include is None`) leaves
           the inherited verdict untouched.
        """
        if not self.config.walker.respect_gitignore:
            return False

        ignored = False
        for anchor, spec in self._gitignore_chain(path):
            rel = path.relative_to(anchor).as_posix()
            if is_dir:
                rel += "/"
            verdict = spec.check_file(rel).include
            if verdict is not None:
                ignored = verdict
        return ignored

    # ------------------------------------------------------------------
    # Conditional ignore tier (`packages`, `bin`)
    # ------------------------------------------------------------------

    def _under_store_path(self, path: Path) -> bool:
        """True if any segment below the repo root is a package store.

        See `_STORE_PATH_SEGMENTS` for why evidence found inside one proves nothing.
        """
        try:
            rel = path.relative_to(self.repo_root)
        except ValueError:
            # Outside the repo: judged by the same rule on whatever segments it has, rather
            # than trusted, because a walk can only get here through a symlink.
            rel = path
        return any(part.lower() in _STORE_PATH_SEGMENTS for part in rel.parts)

    @staticmethod
    def _read_package_json(path: Path) -> dict[str, object] | None:
        """Parse `package.json`, or None if it cannot be read or is not a JSON object.

        A malformed manifest degrades to "no key evidence" — the marker-file branch can
        still fire — and never propagates. Failing an index because one `package.json` in
        one directory has a trailing comma would be a far worse outcome than not indexing
        that directory.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, ValueError) as exc:
            logger.debug("Could not read %s for workspace detection: %s", path, exc)
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _bin_points_into(declared: object, dir_name: str) -> bool:
        """True if a `package.json` `bin` declaration targets something inside `dir_name`.

        npm allows both `"bin": "bin/cli.js"` and `"bin": {"tool": "bin/cli.js"}`, so both
        are accepted. The target must point INTO the directory: `{"tool": "dist/cli.js"}`
        beside a `bin/` says nothing about `bin/` and must not admit it.
        """
        targets: list[object]
        if isinstance(declared, str):
            targets = [declared]
        elif isinstance(declared, dict):
            targets = list(declared.values())
        else:
            return False

        for target in targets:
            if not isinstance(target, str):
                continue
            # Only a leading "./" is stripped, never with `lstrip("./")`: that strips any
            # run of both characters, turning "../bin/x" and "/abs/bin/x" into "bin/x" and
            # admitting a directory the declaration never pointed at.
            normalized = target[2:] if target.startswith("./") else target
            parts = PurePosixPath(normalized).parts
            if len(parts) > 1 and parts[0].lower() == dir_name.lower():
                return True
        return False

    def _classify_conditional_dirs(self, parent: Path, entries: list[Path]) -> dict[str, str]:
        """Decide, once per parent, which conditional dirs inside it hold first-party source.

        Returns `{lowercased dir name: evidence filename}`; a name absent from the result
        has no positive evidence and stays ignored.

        Positive evidence is REQUIRED, which inverts the tempting rule "ignore `bin/` only
        when .NET markers are present". The absence of .NET evidence is not evidence of
        source: absent-means-index would silently expand the walk into every virtualenv
        `bin/`, every Go `bin/` and every compiled-output `bin/` in the wild, priced at
        whatever that repo happens to contain. Requiring proof can only UNDER-include, and
        under-inclusion is now loud (the directory is named at WARNING) while
        over-inclusion is silent money. That asymmetry is the entire defect being fixed
        here, so it is not repeated in the fix.
        """
        candidates = {
            entry.name.lower(): entry.name
            for entry in entries
            if entry.name.lower() in _CONDITIONAL_IGNORE_DIRS and entry.is_dir()
        }
        if not candidates:
            return {}
        if self._under_store_path(parent):
            return {}

        names = sorted(entry.name for entry in entries)
        # Both branches case-fold. See `_DOTNET_MARKER_FILES` for the measurement that put
        # the marker-file branch here rather than leaving it exact-case.
        if any(name.lower() in _DOTNET_MARKER_FILES for name in names) or any(
            name.lower().endswith(_DOTNET_MARKER_SUFFIXES) for name in names
        ):
            # Applied to `bin` as well as `packages`, even though a .NET `bin/` could never
            # produce the `package.json` evidence below anyway. Restating it costs one
            # comparison and makes the .NET regression these three entries exist to prevent
            # independent of how the positive rules later change.
            return {}

        manifest = (
            self._read_package_json(parent / "package.json") if "package.json" in names else None
        )

        verdicts: dict[str, str] = {}
        if "packages" in candidates:
            marker = next((name for name in names if name.lower() in _WORKSPACE_MARKER_FILES), None)
            if marker is not None:
                verdicts["packages"] = marker
            elif manifest is not None and "workspaces" in manifest:
                verdicts["packages"] = "package.json"
        if "bin" in candidates and manifest is not None:
            if self._bin_points_into(manifest.get("bin"), candidates["bin"]):
                verdicts["bin"] = "package.json"
        return verdicts

    def _prime_conditional_cache(self, parent: Path, entries: list[Path]) -> None:
        """Classify `parent`'s conditional dirs from a listing the caller already has.

        Called once per directory from `_iter_files`, so the walk pays zero extra syscalls
        for the probe: marker names come out of `entries`, and the single `package.json`
        read only happens in a directory that actually contains a `packages/` or `bin/`.
        """
        if parent not in self._conditional_cache:
            self._conditional_cache[parent] = self._classify_conditional_dirs(parent, entries)

    def _conditional_evidence(self, dir_path: Path) -> str | None:
        """The evidence file admitting `dir_path`, or None if it stays ignored.

        On the watcher paths there is no preceding `_iter_files`, so the memo is cold and
        the parent is listed here — the one place the probe costs a real syscall, bounded to
        once per parent for the walker's lifetime (`MultiRepoWatcher` builds its walkers
        once per repo, not once per event).
        """
        parent = dir_path.parent
        memo = self._conditional_cache.get(parent)
        if memo is None:
            try:
                entries = sorted(parent.iterdir())
            except OSError:
                # Unreadable parent: no evidence, so the directory keeps today's verdict.
                # NOT recorded as incomplete — the traversal itself records that when it
                # fails to list the same directory, and it is not a gap in the index here.
                entries = []
            memo = self._classify_conditional_dirs(parent, entries)
            self._conditional_cache[parent] = memo
        return memo.get(dir_path.name.lower())

    def _report_conditional(self, dir_path: Path, evidence: str) -> None:
        """Say once, at WARNING, that a directory holding source is not being indexed.

        Reported for any effective `extra_ignore_dirs` that STILL LISTS the name — not only
        for a list byte-identical to trelix's shipped default. The old gate meant that
        following docs/CONFIGURATION.md's own instruction ("to add one entry you must restate
        the whole list") silenced the fix: default + `".cache"` still excluded
        `packages/core/index.ts`, now with zero signal. Worse, it ate its own tail — a repo
        with both a workspace `packages/` and a declared `bin/` warned about both on the
        defaults, and warned about NEITHER once the user removed `packages` as instructed,
        hiding `bin/cli.js` in silence.

        The "a hand-written override is a choice" rationale survives without a gate, because
        it never needed one: dropping a name leaves it out of `_conditional_names`, so no probe
        runs, and out of `exact`, so `_is_ignored_dir` never calls this. Removing an entry is
        therefore still the way to make the report stop, and it is the only way — which is the
        property that makes the report trustworthy. `scripts/self-index.sh` drops `packages`
        and keeps `bin`, so it can only ever be told about `bin`, and only in a repo whose root
        `package.json` declares one (trelix has no root `package.json`, so it stays silent).
        """
        try:
            rel = str(dir_path.relative_to(self.repo_root))
        except ValueError:
            rel = str(dir_path)
        if rel in self._reported_conditional:
            return
        self._reported_conditional.add(rel)
        logger.warning(
            "%s is NOT being indexed: %r is in this run's extra_ignore_dirs (one of "
            "trelix's .NET build-output defaults), but %s beside it declares this directory "
            "as first-party source. To index it, set TRELIX_WALKER_EXTRA_IGNORE_DIRS to the "
            "%d entries this run is using minus %r — the variable REPLACES the list rather "
            "than extending it, and a comma-separated value is rejected, so pass JSON "
            "(scripts/self-index.sh is a working reference). Run `trelix index --dry-run` "
            "for the file and token count before spending.",
            rel,
            dir_path.name,
            evidence,
            len(self.config.walker.extra_ignore_dirs),
            dir_path.name,
        )

    def _is_ignored_dir(self, dir_path: Path) -> bool:
        name = dir_path.name
        # The unconditional tier is byte-exact, as it has always been: `Bin/` and `OBJ/`
        # are NOT caught. Left alone deliberately — case-folding it would newly exclude
        # directories every existing index contains. The conditional tier below IS
        # case-insensitive, because `Packages/` on a case-insensitive filesystem is the
        # same directory as `packages/` and the probe answers for the real one.
        exact = name in self.config.walker.extra_ignore_dirs

        if name.lower() in self._conditional_names:
            evidence = self._conditional_evidence(dir_path)
            if self._index_conditional_dirs:
                if evidence is not None:
                    return self._is_gitignored(dir_path, is_dir=True)
                # Ambiguous: no workspace manifest, no `bin` declaration. Keeps whatever
                # verdict the UNCONDITIONAL tier already gave it — see
                # `_classify_conditional_dirs` for why proof is required.
                #
                # `if exact` rather than a bare `return True`, and that is the whole point:
                # the conditional tier is case-INSENSITIVE while the unconditional tier is
                # byte-exact, so `Bin/` and `Packages/` enter this block without ever having
                # been excluded. A bare `return True` therefore NARROWED the walk — it
                # dropped a capitalised `Bin/` full of first-party shell scripts (measured:
                # 7 files -> 1, six `Bin/deploy*.sh` lost) with no warning at all, because
                # the report below is guarded on `exact` too. This tier may only ever WIDEN
                # what the unconditional tier excluded, never narrow it; a silent
                # contraction is the exact defect class this whole change exists to remove.
                return True if exact else self._is_gitignored(dir_path, is_dir=True)
            if evidence is not None and exact:
                self._report_conditional(dir_path, evidence)

        if exact:
            return True
        return self._is_gitignored(dir_path, is_dir=True)

    def _is_ignored_file(self, file_path: Path) -> bool:
        """Return True if this file should be excluded by .gitignore patterns."""
        return self._is_gitignored(file_path, is_dir=False)

    def _detect_language(self, path: Path) -> Language:
        return detect_language(path)

    def _compute_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    def walk(self) -> Iterator[IndexedFile]:
        """Yield IndexedFile for every indexable file in the repo."""
        # Reset per walk, so a second walk does not inherit the first one's gaps.
        self._incomplete_paths = []

        allowed_languages = set(self.config.walker.languages)
        ignore_extensions = set(self.config.walker.extra_ignore_extensions)
        ignore_filenames = set(self.config.walker.extra_ignore_filenames)
        max_size = self.config.walker.max_file_size_bytes

        for path in self._iter_files(self.repo_root):
            # Gitignore file-level filter
            if self._is_ignored_file(path):
                continue

            # Exact filename filter (catches package-lock.json etc.)
            if path.name in ignore_filenames:
                continue

            # Extension filter
            if any(path.name.endswith(ext) for ext in ignore_extensions):
                continue

            # Language detection
            language = self._detect_language(path)
            if language not in allowed_languages:
                continue

            # Size filter. A stat() failure means we cannot judge the file at all, so
            # it is recorded rather than quietly dropped — unlike the size check below
            # it, which is a deliberate exclusion and not a gap.
            try:
                size = path.stat().st_size
            except OSError as exc:
                self._record_incomplete(path, exc)
                continue
            if size > max_size:
                continue

            # Compute hash
            try:
                file_hash = self._compute_hash(path)
            except OSError as exc:
                self._record_incomplete(path, exc)
                continue

            yield IndexedFile(
                path=str(path),
                rel_path=str(path.relative_to(self.repo_root)),
                language=language,
                hash=file_hash,
                size_bytes=size,
            )

    def _is_within_root(self, entry: Path) -> bool:
        """True unless containment is enabled and `entry` resolves outside the repo.

        With `follow_symlinks` at its default of True this returns True without
        touching the filesystem, so the traversal is bit-for-bit what it always
        was — no extra resolve() syscall per entry, and no behaviour change.

        When containment is enabled, the comparison is on RESOLVED paths.
        `Path.is_relative_to` is a lexical check, so comparing an unresolved entry
        against an unresolved root would let `repo/link -> /etc` through: the
        lexical path still starts with repo/. Resolving both sides is what makes
        the boundary real.

        Both failure modes exclude the entry — that is the safe reading of a
        containment setting — but only one of them is a GAP in the index. See the
        handlers below.
        """
        if self.config.walker.follow_symlinks:
            return True
        try:
            return entry.resolve().is_relative_to(self._resolved_root)
        except RuntimeError:
            # ELOOP. On 3.11/3.12 `resolve(strict=False)` calls
            # `os.path.realpath(strict=False)` — which never raises — then stat()s the
            # result and SWALLOWS every OSError except ELOOP, re-raising that one as
            # RuntimeError("Symlink loop from ..."). Measured on 3.11.14: broken,
            # nonexistent and permission-denied paths all resolve silently. So an
            # `except OSError` here was dead code, and a mutual loop (`la -> lb`,
            # `lb -> la`) aborted the ENTIRE walk with an uncaught RuntimeError, losing
            # every legitimate sibling file with it. (3.13+ raises nothing at all for a
            # loop — resolve() returns the path unchanged and is_dir()/is_file() are both
            # False — so this branch is 3.11/3.12-only, and both are supported.)
            #
            # NOT recorded as incomplete, matching the symlink-CYCLE precedent below in
            # _iter_files: ELOOP means resolution never terminates, so no file exists
            # behind the entry and nothing real is missed. Recording it would clear
            # walk_was_complete — and with it missing_is_trustworthy — for the lifetime
            # of a link the user created deliberately.
            return False
        except OSError as exc:
            # A resolve failure that is NOT a loop says nothing about whether real
            # content sits behind this path, and containment then drops it — so unlike
            # the branch above this IS a gap, recorded exactly like the stat() and
            # iterdir() failures elsewhere in this class. Unreachable on CPython today
            # (see above) but not decorative: resolve() is documented as raising OSError,
            # and 3.13+ reimplemented it on top of os.path.realpath, whose non-strict
            # contract is the only thing standing between here and a real errno.
            self._record_incomplete(entry, exc)
            return False

    def _iter_files(self, root: Path, _chain: frozenset[Path] | None = None) -> Iterator[Path]:
        """Recursive directory traversal, skipping ignored dirs and symlink cycles.

        `_chain` is the set of RESOLVED directory paths on the current recursion
        path — the ancestors of `root`, not every directory seen so far. That
        distinction is the whole design:

        A symlink pointing at one of its own ancestors makes the walk re-enter a
        directory it is already inside. `repo/loop -> repo` yielded the same file
        17 times at nesting depths up to 16 (measured), one distinct content hash,
        bounded only by the OS path limit — 17x the embedding cost and 17 duplicate
        results for one file. Note `follow_symlinks=False` does NOT help here: the
        loop target resolves INSIDE the root, so containment correctly permits it.

        A global visited-set would also fix the loop, and would be WRONG. Given
        `repo/a -> repo/shared` and `repo/b -> repo/shared`, it would descend into
        the first and silently drop the second, losing `b/...` from a legitimate
        layout. Only cycles are a defect; two aliases to the same target are not.
        Tracking the current chain fixes the former and leaves the latter intact.
        """
        try:
            entries = sorted(root.iterdir())
        except OSError as exc:
            # Returning here drops the ENTIRE subtree below `root`, which is why it is
            # recorded rather than merely tolerated.
            self._record_incomplete(root, exc)
            return

        if _chain is None:
            try:
                _chain = frozenset({root.resolve()})
            except (OSError, RuntimeError):
                # RuntimeError is the ELOOP case on 3.11/3.12 (see _is_within_root).
                # Nothing is dropped by starting with an empty chain — the repo root
                # simply is not a cycle candidate, and detection resumes one level
                # down — so this is not recorded as a gap.
                _chain = frozenset()

        # Classify this directory's conditional children before the entry loop asks about
        # them, so the probe reads `entries` instead of listing the directory a second time.
        self._prime_conditional_cache(root, entries)

        for entry in entries:
            if not self._is_within_root(entry):
                continue
            if entry.is_dir():
                if self._is_ignored_dir(entry):
                    continue
                try:
                    target = entry.resolve()
                except (OSError, RuntimeError) as exc:
                    # `entry.is_dir()` just succeeded, which means resolution through
                    # the entry worked a moment ago and found a directory — so this is
                    # neither a broken link nor a loop (both make is_dir() False), but a
                    # race or a filesystem error. The earlier comment here claimed
                    # "broken link or unreadable" and caught OSError only, which on
                    # 3.11/3.12 cannot fire at all (see _is_within_root).
                    #
                    # Unlike a dangling entry this drops an entire SUBTREE that existed,
                    # so it is recorded: walk_was_complete must not keep asserting the
                    # index mirrors the repository while a directory's worth of files is
                    # missing from it.
                    self._record_incomplete(entry, exc)
                    continue
                if target in _chain:
                    # Re-entering an ancestor. Skipping is silent by design — this
                    # is a link the user created, not an error, and the walk still
                    # yields every real file exactly once via its non-looping path.
                    continue
                yield from self._iter_files(entry, _chain | {target})
            elif entry.is_file():
                yield entry
