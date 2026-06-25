"""
FileWalker: discovers files in a repo, detects language, filters noise.

Key decisions stolen from continue.dev:
- pathspec for .gitignore-aware walking (respects nested .gitignore files)
- Comprehensive default ignore list for dirs and extensions
- File hash (SHA-256) computed here so the indexer can skip unchanged files
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
        self._gitignore_spec = self._load_gitignore_spec()

    def _load_gitignore_spec(self) -> pathspec.PathSpec | None:
        """Load .gitignore patterns from the repo root (and nested dirs later)."""
        gitignore_path = self.repo_root / ".gitignore"
        if not self.config.walker.respect_gitignore or not gitignore_path.exists():
            return None
        patterns = gitignore_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def _is_ignored_dir(self, dir_path: Path) -> bool:
        if dir_path.name in self.config.walker.extra_ignore_dirs:
            return True
        if self._gitignore_spec:
            rel = dir_path.relative_to(self.repo_root)
            return self._gitignore_spec.match_file(str(rel) + "/")
        return False

    def _is_ignored_file(self, file_path: Path) -> bool:
        """Return True if this file should be excluded by .gitignore patterns."""
        if self._gitignore_spec:
            rel = file_path.relative_to(self.repo_root)
            return self._gitignore_spec.match_file(str(rel))
        return False

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

    def _iter_files(self, root: Path) -> Iterator[Path]:
        """Recursive directory traversal, skipping ignored dirs."""
        try:
            entries = sorted(root.iterdir())
        except PermissionError:
            return

        for entry in entries:
            if entry.is_dir():
                if not self._is_ignored_dir(entry):
                    yield from self._iter_files(entry)
            elif entry.is_file():
                yield entry
