"""The docs and store comments describe the sqlite store as the exact flat scan it is.

The sqlite store is a plain `vec0` table scanned flat (see src/trelix/store/vector.py), and
`trelix index` / `trelix stats` print a `Flat-scan advisory` past a measured threshold. The
sweep that made the prose say so removed three families of claim, and this file keeps each of
them from growing back:

- an HNSW index on the sqlite store ("sqlite-vec HNSW", "faster HNSW", "HNSW query", ...);
- the old chunk-count sizing rules (the advisory's measured thresholds replace them);
- "sqlite-vec has never shipped" an ANN index (what the repo pin supports is "no stable
  release has", because the `<0.1.10` pin keeps the ivf/diskann pre-releases out).

Every phrase below is a literal that stood in that file before the sweep (or in a first draft of
it) and is gone now, so each case fails when its claim is put back. The advisory itself is tested
in tests/unit/test_scale_advisory.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Each phrase below stood in its file before the sweep (or in a first draft of it): putting one
# back makes test_no_swept_claim_comes_back report that file.
_REMOVED_PHRASES: dict[str, tuple[str, ...]] = {
    "README.md": (
        "Repos up to ~100k chunks",
        "500k+ chunks, multi-repo",
        "100k+ chunks, ARM/Apple Silicon",
        "faster HNSW",
    ),
    "docs/FAQ.md": (
        "up to roughly **100,000 chunks**",
        "| 100k–500k chunks |",
        "LanceDB — 3–5x faster insert |",
        "faster HNSW",
        "#storage)). Past it:",
    ),
    "docs/GLOSSARY.md": (
        "existing HNSW index",
        "sqlite-vec has never shipped",
        "ARM-native HNSW",
        "approximate on Qdrant and LanceDB",
    ),
    "docs/INSTALLATION_GUIDE.md": (
        "Use an HNSW ANN index",
        "Large-scale vector store (>100k chunks)",
    ),
    "docs/PROVIDERS.md": ("faster HNSW",),
    "docs/USER_GUIDE.md": (
        "scales to 500k+ chunks",
        "faster HNSW",
        "slower HNSW",
        "(HNSW query)",
        "For repos under 100k chunks",
        "the widest vectors",
        "256 dims (very large repos only)",
    ),
    "docs/WHY_TRELIX.md": (
        "sqlite-vec HNSW",
        "HNSW ANN over embeddings",
        "combination of HNSW vector search",
    ),
    "docs/discoverability/AWESOME-LIST-SUBMISSIONS.md": ("sqlite-vec HNSW", "for >500k chunks"),
    "docs/CLI_REFERENCE.md": ("sqlite-vec has never shipped",),
    "docs/CONFIGURATION.md": ("sqlite-vec has never shipped",),
    "docs/architecture.md": ("sqlite-vec has never shipped", "ARM-native HNSW"),
    ".env.example": ("Use qdrant for >500k chunk scale",),
    "scripts/bench_vector_search.py": ("feat/flat-scan-advisory", "advisory planned for"),
    # trelix builds no index on a Lance table: the backend is an exact scan, not an HNSW one.
    "pyproject.toml": ("ARM-native HNSW",),
    "src/trelix/core/config.py": ("faster HNSW", "sqlite-vec has never shipped"),
    "src/trelix/store/dimension_guard.py": ("existing HNSW index",),
    "src/trelix/store/scale_advisory.py": ("sqlite-vec has never shipped",),
    "src/trelix/store/sparse_store.py": ("consider moving to Qdrant's native sparse support",),
    "src/trelix/store/vector.py": ("scales to 500k+ chunks", "sqlite-vec has never shipped"),
    "src/trelix/store/vector_lance.py": (
        "sqlite-vec HNSW becomes memory-constrained",
        "ARM-native HNSW",
    ),
    "src/trelix/store/vector_qdrant.py": ("deployments (>500k chunks)", "for >500k vectors"),
}

# Dropping any listed phrase from its file makes test_the_replacement_claims_stay_in fail.
# Refresh obligation: the user-guide sentence about the threshold table being non-monotonic is
# true of the committed placeholder table only. When a quiet benchmark run replaces the table
# and it is monotonic, that sentence and its pin below change together.
_REQUIRED_PHRASES: dict[str, tuple[str, ...]] = {
    ".env.example": (
        "sqlite | qdrant | lance",
        "# LANCE_URI=.trelix/lance",
        "# LANCE_TABLE=chunks",
    ),
    "docs/CLI_REFERENCE.md": ("no stable sqlite-vec release has shipped",),
    "docs/CONFIGURATION.md": ("no stable sqlite-vec release has shipped",),
    "docs/GLOSSARY.md": (
        "no stable sqlite-vec release has shipped",
        "trelix builds no index on its table",
    ),
    "docs/architecture.md": (
        "no stable sqlite-vec release has shipped",
        "exact scan (trelix builds no index)",
    ),
    # The insert multiplier was measured at 100k+ chunks, far above the advisory threshold, so
    # the row keeps its scale.
    "docs/FAQ.md": ("LanceDB — 3–5x faster insert at 100k+ chunks (vecdb-bench) |",),
    "docs/USER_GUIDE.md": (
        "the committed table is not monotonic, so a narrower width can fire the advisory earlier",
    ),
    "src/trelix/store/vector.py": ("LanceDB table, for indexes past the same threshold",),
    "src/trelix/store/vector_lance.py": ("trelix builds no index on the table",),
}


def _read(relative_path: str) -> str:
    return (_REPO_ROOT / relative_path).read_text(encoding="utf-8")


@pytest.mark.parametrize("relative_path", sorted(_REMOVED_PHRASES))
def test_no_swept_claim_comes_back(relative_path: str) -> None:
    text = _read(relative_path)

    came_back = [phrase for phrase in _REMOVED_PHRASES[relative_path] if phrase in text]

    assert came_back == []


@pytest.mark.parametrize("relative_path", sorted(_REQUIRED_PHRASES))
def test_the_replacement_claims_stay_in(relative_path: str) -> None:
    text = _read(relative_path)

    missing = [phrase for phrase in _REQUIRED_PHRASES[relative_path] if phrase not in text]

    assert missing == []
