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
from collections.abc import Iterator
from pathlib import Path

import pathspec

from trelix.core.config import IndexConfig
from trelix.core.models import IndexedFile, Language

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
}


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

    def _gitignore_chain(
        self, path: Path
    ) -> list[tuple[Path, pathspec.PathSpec]]:  # type: ignore[type-arg]
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
        # Special case: .tsx must be checked before .ts
        suffix = path.suffix.lower()
        return EXTENSION_MAP.get(suffix, Language.UNKNOWN)

    def _compute_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    def walk(self) -> Iterator[IndexedFile]:
        """Yield IndexedFile for every indexable file in the repo."""
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

            # Size filter
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > max_size:
                continue

            # Compute hash
            try:
                file_hash = self._compute_hash(path)
            except OSError:
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

    def _iter_files(self, root: Path) -> Iterator[Path]:
        """Recursive directory traversal, skipping ignored dirs."""
        try:
            entries = sorted(root.iterdir())
        except PermissionError:
            return

        for entry in entries:
            if not self._is_within_root(entry):
                continue
            if entry.is_dir():
                if not self._is_ignored_dir(entry):
                    yield from self._iter_files(entry)
            elif entry.is_file():
                yield entry
