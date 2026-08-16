#!/usr/bin/env bash
#
# Reproducible clean self-index of this repository.
#
# Why a script and not just `trelix index .`
# ------------------------------------------
# Two of the settings a correct self-index needs live in config groups that do NOT
# read `.env`. `WalkerConfig` and `ParserConfig` declare `env_prefix` without
# `env_file` (documented at docs/CONFIGURATION.md), so `TRELIX_WALKER_*` and
# `TRELIX_PARSER_*` entries in `.env` are silently ignored. They must be in the
# process environment, which is exactly what this file is for.
#
# Everything else — the embedder provider, Azure credentials, retrieval flags — IS
# read from `.env` by the config classes that declare `env_file`, so it is not
# duplicated here.
#
# Usage
# -----
#   scripts/self-index.sh              # index into .trelix/index.db
#   scripts/self-index.sh --dry-run    # report what would be indexed, embed nothing
#
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
TRELIX=".venv/bin/trelix"
DB=".trelix/index.db"

[[ -x "$PYTHON" ]] || { echo "error: $PYTHON not found — create the venv first" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Walker: which files get indexed
# ---------------------------------------------------------------------------
# TRELIX_WALKER_EXTRA_IGNORE_DIRS REPLACES the 30-entry default rather than adding
# to it, so every entry has to be restated — drop one and `.git`, `node_modules`,
# `.venv` or `.trelix` starts getting indexed.
#
# Delta from the shipped default, and why:
#   - "packages" REMOVED. It is in the default for .NET NuGet output, but it also
#     matches this repo's own packages/ monorepo directory, silently excluding the
#     three shipped sub-packages (trelix-mcp, trelix-langchain, trelix-llama-index).
#     A name-only filter cannot tell a NuGet cache from a workspace.
#   - ".claude-flow" ADDED. Untracked agent-tooling output that is not in
#     .gitignore, so nothing else would exclude it.
export TRELIX_WALKER_EXTRA_IGNORE_DIRS='[".git",".hg",".svn","node_modules","__pycache__",".mypy_cache",".ruff_cache","venv",".venv","env","dist","build","target","out",".next",".nuxt","coverage",".coverage","vendor","Pods",".gradle",".idea",".vscode",".angular","bin","obj",".vs",".rider",".trelix",".claude-flow"]'

# ---------------------------------------------------------------------------
# Parser: def_use_edges
# ---------------------------------------------------------------------------
# Dataflow extraction is index-time only and cannot be added later without a full
# re-index. Python-only in practice (DataFlowExtractor hardcodes the Python
# grammar), which is most of this repository.
export TRELIX_PARSER_DATAFLOW=true

# ---------------------------------------------------------------------------
# Deliberately NOT set
# ---------------------------------------------------------------------------
# TRELIX_CHUNKER_MULTI_GRANULARITY — Phase 2.6 embeds once per SYMBOL, bypassing
#   the token batching and TPM limiter, so enabling it here means thousands of
#   serial un-throttled embedding round-trips. sub_chunks also has no FK/CASCADE
#   and no delete path anywhere in src/, so rows orphan on every re-index. To
#   measure it, use a throwaway DB:
#     TRELIX_STORE_DB_PATH=/tmp/trelix-mgs3.db TRELIX_CHUNKER_MULTI_GRANULARITY=true \
#       TRELIX_FILE_SUMMARIES_ENABLED=false trelix index .
#
# TRELIX_RETRIEVAL_SPARSE — SparseConfig.model defaults to an unresolvable
#   HuggingFace id, so SparseEmbedder returns empty dicts and 0 rows are written
#   whether the flag is set or not.

if [[ "${1:-}" == "--dry-run" ]]; then
  exec "$PYTHON" - <<'PY'
import collections
from trelix.core.config import IndexConfig
from trelix.indexing.walker import FileWalker

config = IndexConfig(repo_path=".")
assert "packages" not in config.walker.extra_ignore_dirs, (
    "TRELIX_WALKER_EXTRA_IGNORE_DIRS did not reach the config — it must be in the "
    "process environment, not .env"
)
files = list(FileWalker(config).walk())
by_dir = collections.Counter(f.rel_path.split("/")[0] for f in files)
print(f"{len(files)} files would be indexed")
for name, count in by_dir.most_common(12):
    print(f"  {count:>4}  {name}")
for label, needle in ((".vscode-test", "/.vscode-test/"), ("node_modules", "node_modules")):
    stray = sum(1 for f in files if needle in f.path)
    print(f"  {label}: {stray}" + ("  <-- LEAK" if stray else ""))
PY
fi

# ---------------------------------------------------------------------------
# Refuse to index into an index built at a different embedding dimension
# ---------------------------------------------------------------------------
# The vec0 table's dimension is fixed by its CREATE statement and cannot be
# altered. `trelix migrate-vectors --reset` does NOT help: it deletes rows and the
# recorded dimension, leaving the FLOAT[n] declaration in place — and because the
# recorded dimension is gone, DimensionGuard then has nothing to compare against,
# so the next run pays for a full embedding pass before failing on the first
# insert. Deleting the file is the only working path, so say so up front.
if [[ -f "$DB" ]]; then
  "$PYTHON" - "$DB" <<'PY' || exit 1
import re
import sqlite3
import sys

from trelix.core.config import EmbedderConfig
from trelix.embedder import make_embedder

db_path = sys.argv[1]
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
try:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'chunk_embeddings'"
    ).fetchone()
finally:
    conn.close()

if not row:
    sys.exit(0)  # no vector table yet — nothing to conflict with

match = re.search(r"FLOAT\[(\d+)\]", row[0] or "")
if not match:
    sys.exit(0)

stored = int(match.group(1))
current = make_embedder(EmbedderConfig()).dimension
if stored != current:
    print(
        f"error: {db_path} was built with {stored}-dim vectors but the configured "
        f"embedder produces {current}-dim.\n"
        f"       The vec0 table's dimension cannot be changed in place, and\n"
        f"       `trelix migrate-vectors --reset` will not fix it.\n"
        f"       Delete the index and re-run:\n"
        f"         rm -f {db_path} {db_path}-wal {db_path}-shm",
        file=sys.stderr,
    )
    sys.exit(1)
PY
fi

exec "$TRELIX" index . --verbose
