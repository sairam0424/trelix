"""On-disk cache of index-time document embeddings (``TRELIX_EMBEDDING_CACHE_*``).

WHY. Re-indexing text that has not changed pays the embedding provider again: a fresh
clone, a deleted ``.trelix/index.db``, a second worktree of the same repository, or a
CI runner with no index all re-embed every chunk. The chunk text is the same bytes, so
its vector is the same vector; this module remembers it.

WHAT IT IS. One SQLite file per *embedder fingerprint* (provider, model id, the width
knobs that change a vector for the same model id, and the declared width), named
``<fingerprint>.db`` inside the resolved cache directory. Each row maps
``sha256(chunk_text)`` to the float32 vector the embedder produced. The cache is
content-addressed, so two repositories that share a file share its rows, and a model
change is a different file rather than a mixed one.

WHERE IT SITS. ``CachedIndexEmbedder`` wraps the embedder that ``Indexer.__init__`` builds,
which is the one point every index-time embed call goes through: both Phase 3
strategies, the file-summary and sub-chunk side embeds, and ``index_file()``.
``embed_query`` passes straight through and is never cached here; that is
``TRELIX_RETRIEVAL_QUERY_CACHE_SIZE`` (``embedder/cache.py``), an in-memory LRU for
queries. The two are deliberately separate things.

WHAT IT DOES NOT DO. It never creates a directory except in ``prepare_cache_dir`` (the one
creator, mode ``0o700``); ``EmbeddingCache.open`` requires the parent to exist. It never
interprets the indexed text: the text is hashed, the vector stored, nothing else.
``TRELIX_EMBEDDING_CACHE_MAX_MB`` is a trim applied after a run, not a limit during one.

SENSITIVITY. The file holds a vector for every indexed chunk, so it carries the index's
sensitivity, plus membership inference: anyone who can read it can test whether a given
text was embedded. Hence ``0o700``/``0o600`` and SECURITY.md's section on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from trelix.core.config import EmbedderConfig, EmbeddingCacheConfig
from trelix.embedder.base import BaseEmbedder


class EmbeddingCacheError(ValueError):
    """The cache directory or file cannot be used.

    A ``ValueError`` for consistency with the other pre-load refusal raised beside it, the
    Batch API conflict in ``Indexer._prepare_embedding_cache_dir``: both mean "this
    configuration cannot run", and one handler covers both. The CLI needs neither base
    class for its one-line error: its ``Indexer(...)`` calls sit under ``except Exception``.
    """


# Which EmbedderConfig field holds the model id for each provider. There is no single
# `model` attribute; this is the same table `cli/main.py` prices with. A new provider
# MUST be added here (and to _KNOB_FIELDS if a knob changes its vectors), or two
# different models could share one cache file. CONTRIBUTING.md says so.
EMBED_MODEL_FIELDS: dict[str, str] = {
    "openai": "openai_model",
    "azure": "azure_embeddings_deployment",
    "voyage": "voyage_model",
    "local": "local_model",
    "local-code": "local_code_model",
    "bge-code": "bge_code_model",
    "nomic-code": "nomic_code_model",
    "bedrock-titan": "bedrock_titan_model",
    "bedrock-cohere": "bedrock_cohere_model",
    "cohere": "cohere_model",
}

# Config knobs that change the vector for the SAME model id. `azure_endpoint` is hashed
# into the fingerprint only; it is never stored or logged in clear.
_KNOB_FIELDS: dict[str, tuple[str, ...]] = {
    "openai": ("openai_dimensions",),
    "azure": ("azure_dimensions", "azure_endpoint"),
    "voyage": ("voyage_output_dimensions",),
    "bedrock-titan": ("bedrock_titan_dimensions", "bedrock_titan_normalize"),
}

_IDENTITY_SCHEMA = 1
_FLOAT32_BYTES = 4
_LOOKUP_GROUP = 500  # keys per IN (...) clause; well under SQLite's variable limit

_DIR_UNUSABLE = (
    "embedding cache directory cannot be used: {dir} ({exc}). Fix it, point "
    "TRELIX_EMBEDDING_CACHE_DIR at a writable directory, or set "
    "TRELIX_EMBEDDING_CACHE_ENABLED=false."
)
_WIDTH_MISMATCH = (
    "embedding cache {path} holds {stored}-dimensional vectors but the current embedder "
    "produces {current}: the model behind this fingerprint changed. Delete that file and "
    "index again."
)
_NOT_A_CACHE = "{path} is not a trelix embedding cache (no meta table). Move it away."
_WRONG_SCHEMA = (
    "{path} is a trelix embedding cache with schema {version}; this trelix reads schema 1. "
    "Delete that file and index again."
)
# SQLite could not read or write the file (locked past the timeout, read-only directory,
# I/O error): the file may well be a valid cache, so the advice is NOT "move it away".
_FILE_UNUSABLE = (
    "embedding cache {path} cannot be opened ({exc}). If another indexer holds it, retry; "
    "otherwise fix the directory or set TRELIX_EMBEDDING_CACHE_ENABLED=false."
)

# One statement each, so they can run inside the caller's transaction (``executescript``
# would commit it first and expose the tables before the meta rows exist).
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS embeddings ("
    "text_sha256 BLOB PRIMARY KEY, vector BLOB NOT NULL, last_used_at INTEGER NOT NULL"
    ") WITHOUT ROWID",
    "CREATE INDEX IF NOT EXISTS embeddings_last_used ON embeddings(last_used_at)",
)


def embedder_fingerprint(cfg: EmbedderConfig) -> str:
    """32 hex chars naming "this provider, this model, these width knobs".

    Canonical JSON (sorted keys, no spaces) of the identity, sha256, first 32 hex digits.
    ``declared_dimension`` is ``effective_dimension``, which is a *configured* number (384
    for every ``local`` model); the real width of the embedder that was built is checked
    separately against the file's ``meta.dimension`` in :meth:`EmbeddingCache.open`.
    """
    identity = {
        "schema": _IDENTITY_SCHEMA,
        "provider": cfg.provider,
        "model": getattr(cfg, EMBED_MODEL_FIELDS[cfg.provider]),
        "declared_dimension": cfg.effective_dimension,
        "knobs": {name: getattr(cfg, name) for name in _KNOB_FIELDS.get(cfg.provider, ())},
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def resolve_cache_dir(
    cfg: EmbeddingCacheConfig,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """The directory the cache files live in. Touches nothing on disk.

    ``cfg.dir`` wins (already absolute, see the config validator). Otherwise
    ``$XDG_CACHE_HOME/trelix/embeddings``, else ``~/.cache/trelix/embeddings`` — the same
    XDG handling as ``resolve_operator_env_file``, read from ``os.environ`` so a dotenv
    key can never relocate the cache. A relative ``XDG_CACHE_HOME`` is ignored, as the
    XDG Base Directory spec requires: honoured, it would resolve against the process cwd,
    which for ``trelix index .`` is the repository being indexed. ``environ`` and ``home``
    exist for tests.
    """
    if cfg.dir is not None:
        return cfg.dir
    env = os.environ if environ is None else environ
    xdg = Path(env.get("XDG_CACHE_HOME", "").strip()).expanduser()  # blank -> "."
    if xdg.is_absolute():
        return xdg / "trelix" / "embeddings"
    if home is None:
        try:
            home = Path.home()
        except RuntimeError as exc:  # no HOME and no passwd entry
            raise EmbeddingCacheError(
                _DIR_UNUSABLE.format(dir="~/.cache/trelix/embeddings", exc=exc)
            ) from exc
    return home / ".cache" / "trelix" / "embeddings"


def prepare_cache_dir(path: Path) -> Path:
    """Create ``path`` (mode ``0o700``) and prove it is a writable directory.

    The ONLY place that creates a cache directory. The mode applies to the leaf; parents
    such as ``~/.cache`` keep the umask default, which is acceptable because the leaf is
    ``0o700`` and every file inside it ``0o600``. A regular file at ``path`` fails
    ``mkdir`` with ``FileExistsError``; any ``OSError`` becomes ``EmbeddingCacheError``
    so the run stops before any model is loaded.
    """
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
        if not path.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")
        fd, probe = tempfile.mkstemp(dir=path)
        os.close(fd)
        os.unlink(probe)
    except OSError as exc:
        raise EmbeddingCacheError(_DIR_UNUSABLE.format(dir=path, exc=exc)) from exc
    return path


class EmbeddingCache:
    """One fingerprint's cache file. Every connection use is serialised by one lock,
    because the async Phase 3 runs four batches concurrently on one event loop.

    ``hits`` counts positions served from the file; ``misses`` counts distinct texts the
    inner embedder had to produce. The Indexer reads both.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: Path,
        conn: sqlite3.Connection,
        dimension: int,
        clock: Callable[[], float],
    ) -> None:
        self._path = path
        self._conn = conn
        self._dimension = dimension
        self._clock = clock
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dimension(self) -> int:
        return self._dimension

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        dimension: int,
        clock: Callable[[], float] = time.time,
    ) -> EmbeddingCache:
        """Open or initialise ``path`` for ``dimension``-wide vectors.

        Never creates directories (``path.parent`` must exist). A new file is created
        ``0o600``; SQLite's ``-journal`` sidecar copies that mode. An existing file must
        be a schema-1 trelix cache of exactly this width, otherwise
        ``EmbeddingCacheError`` names what it is instead.
        """
        if not path.parent.is_dir():
            raise EmbeddingCacheError(
                _DIR_UNUSABLE.format(dir=path.parent, exc="no such directory")
            )
        is_new = not path.exists() or path.stat().st_size == 0
        try:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)
            if is_new:
                os.chmod(path, 0o600)
        except OSError as exc:
            raise EmbeddingCacheError(_DIR_UNUSABLE.format(dir=path.parent, exc=exc)) from exc
        conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=NORMAL")
            cls._verify_or_initialise(conn, path, dimension)
        except sqlite3.OperationalError as exc:  # locked, read-only, I/O: says what SQLite said
            conn.close()
            raise EmbeddingCacheError(_FILE_UNUSABLE.format(path=path, exc=exc)) from exc
        except sqlite3.DatabaseError as exc:  # not a SQLite file at all, or corrupt
            conn.close()
            raise EmbeddingCacheError(_NOT_A_CACHE.format(path=path)) from exc
        except EmbeddingCacheError:
            conn.close()
            raise
        return cls(path, conn, dimension, clock)

    @classmethod
    def _verify_or_initialise(cls, conn: sqlite3.Connection, path: Path, dimension: int) -> None:
        """Initialise an empty file, or check that an existing one is a schema-1 cache of
        this width.

        ``BEGIN IMMEDIATE`` takes SQLite's write lock BEFORE the inspection, so two indexers
        first-opening the same new file are serialised: the second waits (up to the
        connection timeout), then sees the first's tables and meta rows in one piece — never
        a half-built file, never a duplicate ``meta`` insert. A SQLite file with tables but
        no ``meta`` is a foreign file.
        """
        conn.execute("BEGIN IMMEDIATE")
        tables = {
            name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not tables:
            cls._initialise(conn, dimension)
        elif "meta" not in tables:
            raise EmbeddingCacheError(_NOT_A_CACHE.format(path=path))
        conn.commit()
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        version = meta.get("schema_version")
        if version != str(cls.SCHEMA_VERSION):
            raise EmbeddingCacheError(_WRONG_SCHEMA.format(path=path, version=version))
        stored = meta.get("dimension")
        if stored != str(dimension):
            raise EmbeddingCacheError(
                _WIDTH_MISMATCH.format(path=path, stored=stored, current=dimension)
            )

    @classmethod
    def _initialise(cls, conn: sqlite3.Connection, dimension: int) -> None:
        """Schema and meta rows, inside the caller's transaction (no commit here).
        ``INSERT OR IGNORE`` makes a second initialiser of the same file a no-op instead
        of an ``IntegrityError``."""
        for statement in _SCHEMA:
            conn.execute(statement)
        conn.executemany(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            [("schema_version", str(cls.SCHEMA_VERSION)), ("dimension", str(dimension))],
        )

    def get_many(self, keys: Sequence[bytes]) -> dict[bytes, list[float]]:
        """Vectors for the keys that are present and the right width; absent keys are
        simply absent (never ``None``). Hits have their ``last_used_at`` refreshed."""
        found: dict[bytes, list[float]] = {}
        width = self._dimension * _FLOAT32_BYTES
        fmt = f"<{self._dimension}f"
        now = int(self._clock())
        with self._lock:
            for start in range(0, len(keys), _LOOKUP_GROUP):
                group = list(dict.fromkeys(keys[start : start + _LOOKUP_GROUP]))
                marks = ",".join("?" * len(group))
                rows = self._conn.execute(
                    f"SELECT text_sha256, vector FROM embeddings WHERE text_sha256 IN ({marks})",
                    group,
                ).fetchall()
                hit_keys: list[bytes] = []
                for key, blob in rows:
                    if len(blob) == width:
                        found[key] = list(struct.unpack(fmt, blob))
                        hit_keys.append(key)
                if hit_keys:
                    hit_marks = ",".join("?" * len(hit_keys))
                    self._conn.execute(
                        "UPDATE embeddings SET last_used_at = ? "
                        f"WHERE text_sha256 IN ({hit_marks})",
                        [now, *hit_keys],
                    )
            self._conn.commit()
        return found

    def put_many(self, items: Iterable[tuple[bytes, Sequence[float]]]) -> None:
        """Store vectors as little-endian float32, replacing any row with the same key."""
        fmt = f"<{self._dimension}f"
        now = int(self._clock())
        rows: list[tuple[bytes, bytes, int]] = []
        for key, vector in items:
            if len(vector) != self._dimension:
                raise ValueError(
                    f"embedding cache expects {self._dimension}-dimensional vectors, "
                    f"got one of length {len(vector)}"
                )
            rows.append((key, struct.pack(fmt, *vector), now))
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings(text_sha256, vector, last_used_at) "
                "VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def enforce_size_cap(self, max_bytes: int) -> int:
        """Trim the file to about ``max_bytes`` by evicting the least recently used rows.

        Returns the number of rows evicted (0 when the file is already within the cap).
        The rows are measured by their LIVE bytes, ``page_size * (page_count -
        freelist_count)``, not by the file size: a trim commits its ``DELETE`` before its
        ``VACUUM``, and when that ``VACUUM`` fails (another indexer's lock; the Indexer logs
        it) the file keeps the freed pages and stays over the cap. Measured by file size,
        every following run would evict the same fraction of an ever smaller row set.
        Live-page accounting leaves at most a small residual from partially emptied leaf
        pages (those stay live), then none.
        ``target = max_bytes * rows // live`` rows survive; the oldest by ``last_used_at``
        (ties by key, so the result is deterministic) go, then ``VACUUM`` returns the
        space, also when nothing is evicted and only free pages hold the file over the cap.
        Not a hard limit: one run can write past the cap and is trimmed after.
        """
        with self._lock:
            if self._path.stat().st_size <= max_bytes:
                return 0
            evicted = self._evict_over_cap_locked(max_bytes)
            self._conn.execute("VACUUM")
            return evicted

    def _evict_over_cap_locked(self, max_bytes: int) -> int:
        live = self._live_bytes_locked()
        if live <= max_bytes:
            return 0
        rows = self._row_count_locked()
        evicted = rows - max_bytes * rows // live
        self._conn.execute(
            "DELETE FROM embeddings WHERE text_sha256 IN ("
            "SELECT text_sha256 FROM embeddings ORDER BY last_used_at ASC, text_sha256 ASC "
            "LIMIT ?)",
            (evicted,),
        )
        self._conn.commit()
        return evicted

    def _live_bytes_locked(self) -> int:
        """Bytes in the pages that hold data: the file minus the freelist that a committed
        ``DELETE`` left behind for a ``VACUUM`` that has not run."""
        page_size, page_count, free_pages = (
            int(self._conn.execute(f"PRAGMA {name}").fetchone()[0])
            for name in ("page_size", "page_count", "freelist_count")
        )
        return page_size * (page_count - free_pages)

    def _row_count_locked(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def row_count(self) -> int:
        with self._lock:
            return self._row_count_locked()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _text_key(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


class CachedIndexEmbedder(BaseEmbedder):
    """``BaseEmbedder`` that answers ``embed``/``embed_async`` from the cache first.

    Misses are de-duplicated by key before the inner embedder sees them, so a batch of
    ``["x", "x"]`` costs one embedding; the result keeps the caller's order and length.
    An inner embedder returning the wrong number of vectors raises ``ValueError`` and
    caches nothing (``zip(strict=True)``). ``embed_query`` is delegated uncached.

    Accounting (read by the Indexer as ``chunks_from_cache``): ``hits`` is per position
    (a chunk is served from the cache or it is not), ``misses`` per distinct text.
    """

    def __init__(self, inner: BaseEmbedder, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def inner(self) -> BaseEmbedder:
        return self._inner

    @property
    def cache(self) -> EmbeddingCache:
        return self._cache

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def _lookup(
        self, texts: list[str]
    ) -> tuple[list[bytes], dict[bytes, list[float]], dict[bytes, str]]:
        keys = [_text_key(t) for t in texts]
        found = self._cache.get_many(keys)
        # A dict keyed by the hash, so a text that repeats in `texts` is one miss.
        missing: dict[bytes, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in found:
                missing[key] = text
        return keys, found, missing

    def _finish(
        self,
        keys: list[bytes],
        found: dict[bytes, list[float]],
        missing: dict[bytes, str],
        vectors: list[list[float]],
    ) -> list[list[float]]:
        if len(vectors) != len(missing):
            raise ValueError(
                f"embedder returned {len(vectors)} vector(s) for {len(missing)} text(s); "
                "nothing was cached"
            )
        self._cache.hits += sum(1 for key in keys if key in found)
        self._cache.misses += len(missing)
        fresh = list(zip(missing, vectors))  # lengths checked above
        self._cache.put_many(fresh)
        found.update(fresh)
        return [found[key] for key in keys]

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys, found, missing = self._lookup(texts)
        vectors = self._inner.embed(list(missing.values())) if missing else []
        return self._finish(keys, found, missing, vectors)

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        keys, found, missing = self._lookup(texts)
        vectors = await self._inner.embed_async(list(missing.values())) if missing else []
        return self._finish(keys, found, missing, vectors)
