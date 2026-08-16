#!/usr/bin/env bash
#
# Verify a self-index actually populated what it claims to.
#
# The point of this script is the second half: as well as asserting that the
# dimensions we DO populate are non-empty, it asserts that the ones we do NOT
# populate are still empty. A dimension that is empty for a known reason is a
# documented limitation; a dimension that is empty while something claims it works
# is a false claim. Both need to be checked, and only one of them is checked by
# "did the numbers go up".
#
# Usage:
#   scripts/verify-index.sh            # verify .trelix/index.db
#   scripts/verify-index.sh path.db
#
# Exit codes:
#   0  every gate passed
#   1  at least one gate failed
#   2  the index is missing or unreadable
#
set -uo pipefail
cd "$(dirname "$0")/.."

DB="${1:-.trelix/index.db}"
PYTHON=".venv/bin/python"

[[ -f "$DB" ]] || { echo "error: no index at $DB" >&2; exit 2; }

exec "$PYTHON" - "$DB" <<'PY'
import sqlite3
import sys

import sqlite_vec

db_path = sys.argv[1]
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)

failures: list[str] = []
GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def scalar(sql: str, params: tuple = ()) -> object:
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None
    except sqlite3.Error as exc:
        return f"<error: {exc}>"


def gate(name: str, actual: object, ok: bool, expected: str) -> None:
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {name:<44} {actual!s:>12}   {DIM}expect {expected}{RESET}")
    if not ok:
        failures.append(f"{name}: got {actual!r}, expected {expected}")


print("=" * 90)
print("POPULATED DIMENSIONS — these must be non-empty")
print("=" * 90)

# The vec0 dimension is fixed at CREATE time, so this is the one gate that cannot be
# satisfied by a later repair: if it is wrong, the index has to be rebuilt.
ddl = scalar("SELECT sql FROM sqlite_master WHERE name = 'chunk_embeddings'") or ""
gate("vec0 declared dimension", "3072" if "FLOAT[3072]" in str(ddl) else str(ddl)[:40],
     "FLOAT[3072]" in str(ddl), "FLOAT[3072]")
gate("index_metadata embedding_dimension",
     scalar("SELECT value FROM index_metadata WHERE key='embedding_dimension'"),
     str(scalar("SELECT value FROM index_metadata WHERE key='embedding_dimension'")) == "3072",
     "3072")

files = scalar("SELECT COUNT(*) FROM files")
gate("files indexed", files, isinstance(files, int) and files >= 450, ">= 450")
gate("packages/ sub-packages included",
     (pkg := scalar("SELECT COUNT(*) FROM files WHERE rel_path LIKE 'packages/%'")),
     isinstance(pkg, int) and pkg >= 30, ">= 30")

for label, pattern in (
    (".vscode-test junk", "%.vscode-test%"),
    ("node_modules junk", "%node_modules%"),
    ("scratch-pad junk", "%scratch-pad%"),
):
    n = scalar("SELECT COUNT(*) FROM files WHERE path LIKE ?", (pattern,))
    gate(label, n, n == 0, "0")

for label, sql, minimum in (
    ("symbols", "SELECT COUNT(*) FROM symbols", 5_000),
    ("chunks", "SELECT COUNT(*) FROM chunks", 5_000),
    ("call edges", "SELECT COUNT(*) FROM calls", 1_000),
    ("import edges", "SELECT COUNT(*) FROM imports", 500),
    ("def_use edges", "SELECT COUNT(*) FROM def_use_edges", 1_000),
    ("file summaries", "SELECT COUNT(*) FROM file_summaries", 300),
):
    n = scalar(sql)
    gate(label, n, isinstance(n, int) and n >= minimum, f">= {minimum:,}")

# The most important integrity gate. Chunks are committed in one transaction and
# embedded afterwards, so a failure between the two leaves chunks with no vector —
# and because the file's hash is committed too, a later run skips it as up to date
# and the gap never closes.
orphans = scalar("""
    SELECT COUNT(*) FROM chunks c
     WHERE NOT EXISTS (SELECT 1 FROM chunk_embeddings e WHERE e.chunk_id = c.id)
""")
gate("chunks with no embedding", orphans, orphans == 0, "0")

# Positive ids are real chunk vectors; negatives are file summaries.
real_vecs = scalar("SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id > 0")
chunks = scalar("SELECT COUNT(*) FROM chunks")
gate("chunk vectors == chunks", f"{real_vecs}/{chunks}", real_vecs == chunks, "equal")

summary_vecs = scalar("SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id < 0")
summaries = scalar("SELECT COUNT(*) FROM file_summaries")
gate("summary vectors == summaries", f"{summary_vecs}/{summaries}",
     summary_vecs == summaries, "equal")

blank = scalar("SELECT COUNT(*) FROM file_summaries WHERE summary IS NULL OR TRIM(summary) = ''")
gate("blank file summaries", blank, blank == 0, "0")

print()
print("=" * 90)
print("DIMENSIONS THAT MUST STILL BE EMPTY — each for a recorded reason")
print("=" * 90)

EXPECTED_EMPTY = (
    ("sub_chunks", "SELECT COUNT(*) FROM sub_chunks",
     "multi-granularity off: embeds once per symbol, no delete path, Python-only"),
    ("sparse_embeddings", "SELECT COUNT(*) FROM sparse_embeddings",
     "SparseConfig.model default is an unresolvable HuggingFace id"),
    ("artifacts", "SELECT COUNT(*) FROM artifacts",
     "needs a live Jira/Linear connector sync; no offline route exists"),
    ("generic_edges", "SELECT COUNT(*) FROM generic_edges",
     "this repo's git history contains no real ticket references"),
    ("taint_flows", "SELECT COUNT(*) FROM taint_flows",
     "`trelix index` never writes it; `trelix taint` is a separate command"),
)
for label, sql, reason in EXPECTED_EMPTY:
    n = scalar(sql)
    gate(label, n, n == 0, "0")
    print(f"         {DIM}{reason}{RESET}")

print()
print("=" * 90)
if failures:
    print(f"{RED}{len(failures)} gate(s) failed{RESET}")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print(f"{GREEN}all gates passed{RESET}")
sys.exit(0)
PY
