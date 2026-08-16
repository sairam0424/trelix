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
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path

import pathspec

from trelix.core.config import IndexConfig
from trelix.core.models import IndexedFile, Language

logger = logging.getLogger("trelix.indexing.walker")

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

    def __init__(self, config: IndexConfig) -> None:
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

    def _record_incomplete(self, path: Path, exc: OSError) -> None:
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

    def _is_ignored_dir(self, dir_path: Path) -> bool:
        if dir_path.name in self.config.walker.extra_ignore_dirs:
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
        """
        if self.config.walker.follow_symlinks:
            return True
        try:
            return entry.resolve().is_relative_to(self._resolved_root)
        except OSError:
            # Broken symlink, permission error, or a resolve loop the OS refused.
            # Excluding is the safe reading of a containment setting.
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
            except OSError:
                _chain = frozenset()

        for entry in entries:
            if not self._is_within_root(entry):
                continue
            if entry.is_dir():
                if self._is_ignored_dir(entry):
                    continue
                try:
                    target = entry.resolve()
                except OSError:
                    # Broken link or unreadable: nothing to descend into.
                    continue
                if target in _chain:
                    # Re-entering an ancestor. Skipping is silent by design — this
                    # is a link the user created, not an error, and the walk still
                    # yields every real file exactly once via its non-looping path.
                    continue
                yield from self._iter_files(entry, _chain | {target})
            elif entry.is_file():
                yield entry
