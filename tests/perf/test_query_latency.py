"""
Manual performance test — NOT run in CI (no @pytest.mark, no conftest).

Usage:
    # Index a repo first:
    trelix index /path/to/repo

    # Then run (requires OPENAI_API_KEY or AZURE_API_KEY in .env):
    python tests/perf/test_query_latency.py /path/to/repo

Measures cold vs warm P50/P95 for 20 queries to validate the cache impact.
"""

from __future__ import annotations

import sys
import time

from dotenv import load_dotenv

load_dotenv()

QUERIES = [
    "how does authentication work",
    "database connection pooling",
    "error handling patterns",
    "how is the index built",
    "what parsers are supported",
    "chunking algorithm",
    "vector search implementation",
    "BM25 scoring",
    "call graph expansion",
    "LLM synthesis",
    "how does the file watcher work",
    "GraphRAG map reduce",
    "embedding providers",
    "test coverage",
    "config validation",
    "how to add a new language parser",
    "retrieval pipeline",
    "reranking implementation",
    "incremental indexing",
    "SQLite schema",
]


def run_queries(retriever: object, label: str) -> list[float]:
    latencies = []
    for q in QUERIES:
        t0 = time.perf_counter()
        retriever.retrieve(q)  # type: ignore[attr-defined]
        latencies.append((time.perf_counter() - t0) * 1000)
    lat = sorted(latencies)
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95)]
    print(f"{label}: P50={p50:.0f}ms  P95={p95:.0f}ms  Max={max(lat):.0f}ms")
    return latencies


if __name__ == "__main__":
    import os

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
    from trelix import IndexConfig, Retriever

    config = IndexConfig(repo_path=repo)
    retriever = Retriever(config)

    print("=== Query Embedding Cache Performance Test ===")
    print(f"Repo: {repo}")
    print(f"Cache size: {config.retrieval.query_cache_size}")
    print()

    cold = run_queries(retriever, "Cold (first pass)")
    warm = run_queries(retriever, "Warm (second pass, cached)")

    cold_p50 = sorted(cold)[len(cold) // 2]
    warm_p50 = sorted(warm)[len(warm) // 2]
    speedup = cold_p50 / max(warm_p50, 0.1)
    print(f"\nSpeedup: {speedup:.0f}x  (warm P50 {warm_p50:.0f}ms vs cold P50 {cold_p50:.0f}ms)")
    if warm_p50 < 50:
        print("✅ Cache working: warm P50 < 50ms")
    else:
        print(f"⚠️  Warm P50 {warm_p50:.0f}ms > 50ms — check if cache is enabled")
