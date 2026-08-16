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

# type_edges records inheritance only, so it is two orders of magnitude smaller than the
# other edge tables — 130 rows live (129 `extends`, 1 `implements`) against 20,313 calls
# — and needs a floor of its own rather than any of the shared ones below. Being the odd
# size out is how it went ungated in the first place, and ungated it could fall to zero,
# taking Retriever's expand_with_type_edges leg and CodeGraph's EXTENDS/IMPLEMENTS
# labels with it, without moving a single counter this script prints.
for label, sql, minimum in (
    ("symbols", "SELECT COUNT(*) FROM symbols", 5_000),
    ("chunks", "SELECT COUNT(*) FROM chunks", 5_000),
    ("call edges", "SELECT COUNT(*) FROM calls", 1_000),
    ("import edges", "SELECT COUNT(*) FROM imports", 500),
    ("def_use edges", "SELECT COUNT(*) FROM def_use_edges", 1_000),
    ("type edges (inheritance)", "SELECT COUNT(*) FROM type_edges", 100),
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

# graph_metadata is written by the `trelix graph` pass (GraphBuilder ->
# save_graph_metadata), never by `trelix index`, so a zero here means the graph pass has
# not been run against THIS db — not that indexing broke. It is gated because every
# centrality-ranked path reads it and none of them complain when it is empty:
# top_symbols_by_centrality() returns [] on an empty table and Retriever swallows that
# at DEBUG, so a lost graph pass degrades ranking silently and invisibly.
#
# The floor is deliberately below the symbol count instead of equal to it. CodeGraph
# adds every symbol as a node, so a graph pass run right now would write a row per
# symbol — but the table is a snapshot of whenever it last ran and nothing re-syncs it,
# so it drifts: 10,714 rows against 10,991 symbols live, and 169 of those rows point at
# symbol ids that no longer exist (graph_metadata.symbol_id is a bare PK with no FK to
# cascade the delete). Gating equality would fail on that drift rather than on the loss
# this gate is here to catch.
graph_meta = scalar("SELECT COUNT(*) FROM graph_metadata")
gate("graph_metadata rows", graph_meta,
     isinstance(graph_meta, int) and graph_meta >= 10_000, ">= 10,000")

# A non-empty file with no symbol row is not merely sparse, it is unreachable: chunks
# hang off symbol_id, so no symbols means no chunks and every retrieval leg — vector,
# BM25, FTS, summary, graph — is blind to it. The true invariant is now 0, because
# Indexer._parse_one falls back to LineWindowParser for any non-empty file whose
# extractor returned nothing (indexer.py:701).
#
# It reads 11 anyway, and every one of the 11 is a stale row rather than a parse hole:
# their files.indexed_at all fall in the 11:39–11:53 run that predates the fallback,
# while helm/trelix/templates/deployment.yaml — a Go-templated manifest PyYAML rejects
# with the same ParserError as the other five — was touched afterwards, re-parsed at
# 17:01, and carries 5 line-window SECTION symbols. Incremental indexing will not close
# the gap on its own — indexer.py:385 skips a file whose stored hash still matches, and
# what changed for these 11 is not their content but the code that parses them.
#
# So the bound is 11 rather than 0 — an upper bound on a known-stale set, tight enough
# that a whole extractor going dark trips it (that failure moves nothing else here:
# symbols would shed a few hundred of 10,991 and still clear its floor). Force-reindex
# those 11 files and this should be tightened to 0. Raising it is not a fix.
unparsed = scalar("""
    SELECT COUNT(*) FROM files f
     WHERE f.size_bytes > 0
       AND NOT EXISTS (SELECT 1 FROM symbols s WHERE s.file_id = f.id)
""")
gate("non-empty files with 0 symbols", unparsed,
     isinstance(unparsed, int) and unparsed <= 11, "<= 11")

print()
print("=" * 90)
print("DIMENSIONS THAT MUST STILL BE EMPTY — each for a recorded reason")
print("=" * 90)

# Each reason states why the table is empty on a correct run. A reason that has stopped
# being true is worse than no reason: it tells whoever reads the output that a bug is
# still open when it has been closed, and it hides the real cause of the zero.
#
# generic_edges in particular used to be explained by "this repo's git history contains
# no real ticket references", which overstated it. Running GitLinker's own _walk_log
# against HEAD matches 13 distinct strings in 9 of 871 commits — PROJ-123, ENG-45,
# JDK-8301717, OSC-52, SA-4 and friends, every one of them a doc example, a test
# fixture, a JDK bug id or a licence fragment rather than a tracker key. So enabling
# the linker here would not be a no-op; it would write noise. The reason the table is
# empty is the one recorded below: nothing in the index pipeline calls the linker.
EXPECTED_EMPTY = (
    ("sub_chunks", "SELECT COUNT(*) FROM sub_chunks",
     "ChunkerConfig.multi_granularity_enabled=False; extractor is Python-grammar only"),
    ("sparse_embeddings", "SELECT COUNT(*) FROM sparse_embeddings",
     "RetrievalConfig.sparse_enabled=False — the model id loads now, the leg is opt-in"),
    ("artifacts", "SELECT COUNT(*) FROM artifacts",
     "needs a live Jira/Linear connector sync; no offline route exists"),
    ("generic_edges", "SELECT COUNT(*) FROM generic_edges",
     "`trelix index` never runs GitLinker; only `trelix link-tickets` writes this"),
    ("diff_chunks", "SELECT COUNT(*) FROM diff_chunks",
     "DiffEmbedder.store_pr_diff is the only writer and nothing in src/ constructs it"),
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
