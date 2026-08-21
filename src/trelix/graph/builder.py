"""GraphBuilder — orchestrates the full knowledge graph construction pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trelix.core.config import IndexConfig
from trelix.core.models import Symbol
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import (
    PartitionQuality,
    assess_partition_quality,
    assign_communities,
    detect_communities,
    get_community_summary,
)
from trelix.graph.concepts import ConceptExtractor, save_concepts
from trelix.graph.persistence import save_graph_metadata
from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.builder")


@dataclass
class GraphBuildResult:
    code_graph: CodeGraph
    community_count: int
    node_count: int
    edge_count: int
    concept_count: int
    elapsed_seconds: float
    community_summary: list[dict[str, Any]] = field(default_factory=list)
    # Defaulted so existing constructors (api/app.py, the MCP server's tests)
    # keep working; None means "not assessed", which is distinct from "healthy".
    partition_quality: PartitionQuality | None = None
    # Coverage of concept extraction. `concept_count` alone reads as a property of the
    # repository when it is really a property of a capped sample: extraction stops at
    # _MAX_CONCEPT_SYMBOLS, which on a 12,184-symbol index is 1.6% of it. Both default
    # to 0, which is also the honest value when extract_concepts was False.
    concept_symbols_considered: int = 0
    concept_symbols_total: int = 0


class GraphBuilder:
    """
    Orchestrates the full knowledge graph construction:
      1. Build CodeGraph from existing DB edges
      2. Run community detection
      3. (Optional) Extract semantic concepts via LLM
      4. Persist graph metadata to DB
    """

    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        db_path = config.db_path_absolute
        self._db = Database(Path(db_path) if not isinstance(db_path, Path) else db_path)

    def build(self, extract_concepts: bool = False) -> GraphBuildResult:
        start = time.perf_counter()
        logger.info("Building CodeGraph from %s", self._config.repo_path)

        # Step 1: build graph
        cg = CodeGraph(self._db)
        logger.info("CodeGraph: %d nodes, %d edges", cg.node_count, cg.edge_count)

        # Step 2: community detection
        communities = detect_communities(cg, algorithm="louvain")
        assign_communities(cg, communities)
        community_count = len(set(communities.values())) if communities else 0
        community_summary = get_community_summary(cg)
        logger.info("Detected %d communities", community_count)

        # A bare community count hid a degenerate partition for several releases:
        # trelix's own index reports 6640 communities, of which 6579 are single
        # nodes. WARNING (not INFO) because the CLI's default level is WARNING —
        # at INFO this stays invisible without -v, which is how it went unnoticed.
        quality = assess_partition_quality(cg, communities)
        if quality.is_degenerate:
            logger.warning("Degenerate community partition. %s", quality.describe())
        else:
            logger.info("Community partition: %s", quality.describe())

        # Step 3: persist metadata (community assignments)
        save_graph_metadata(self._db, cg)

        # Step 3b: Compute and persist PageRank centrality scores
        from trelix.graph.community import compute_pagerank

        pr_scores = compute_pagerank(
            cg, personalization_enabled=self._config.retrieval.pagerank_personalization_enabled
        )
        for node_id, score in pr_scores.items():
            if node_id in cg.nx.nodes:
                cg.nx.nodes[node_id]["centrality"] = score
        # Re-save metadata now that centrality attrs are set on nodes
        save_graph_metadata(self._db, cg)
        logger.info("Computed PageRank for %d nodes", len(pr_scores))

        # Step 4: optional concept extraction
        _MAX_CONCEPT_SYMBOLS = 200
        _CONCEPT_BATCH_SIZE = 20
        concept_count = 0
        concept_symbols_considered = 0
        concept_symbols_total = 0
        if extract_concepts:
            symbols_with_files = self._db.iter_all_symbols_with_files()
            symbols = [s for s, _ in symbols_with_files]
            concept_symbols_total = len(symbols)
            if symbols:
                extractor = ConceptExtractor(self._config.llm)

                # Rank by the PageRank centrality computed in step 3b before taking the
                # top _MAX_CONCEPT_SYMBOLS. Previously this sliced whatever order
                # iter_all_symbols_with_files() happened to return, which has no ORDER BY.
                # Measured on this repo's own index: the query plans as a plain `SCAN s`,
                # so the "first 200" were just the lowest symbol ids — 2..226, drawn from
                # 10 files, all of them .github/ and .devcontainer/ metadata (issue
                # templates, dependabot.yml, a workflow, SECURITY.md). Nothing from src/.
                # Every paid call went to repository boilerplate. Centrality was already
                # computed and sitting unused two steps above; the id tiebreak makes the
                # order total, so equal-centrality symbols cannot reshuffle if the DB
                # order ever changes.
                def _rank_key(symbol: Symbol) -> tuple[float, int]:
                    sid = symbol.id
                    if sid is None:
                        # Symbol.id is Optional because an unsaved Symbol has no id yet.
                        # These come from the DB so every one is saved, but the type
                        # permits None and a None in the tuple would raise on comparison
                        # against an int. Ranked last: centralities are positive, so
                        # -centrality is negative and 0.0 sorts after all of them.
                        return (0.0, 0)
                    return (-pr_scores.get(sid, 0.0), sid)

                ranked = sorted(symbols, key=_rank_key)
                concept_symbols_considered = min(len(ranked), _MAX_CONCEPT_SYMBOLS)
                if concept_symbols_total > _MAX_CONCEPT_SYMBOLS:
                    # WARNING, not INFO, for the reason given in step 2 above: the CLI's
                    # default level is WARNING, so an INFO line here is invisible without
                    # -v — which is how this truncation went unnoticed in the first place.
                    logger.warning(
                        "Concept extraction covered %d of %d symbols (%.1f%%), highest "
                        "centrality first. `concept_count` therefore describes that "
                        "sample, not the repository.",
                        concept_symbols_considered,
                        concept_symbols_total,
                        100.0 * concept_symbols_considered / concept_symbols_total,
                    )
                # Batch into groups of _CONCEPT_BATCH_SIZE, cap at _MAX_CONCEPT_SYMBOLS total
                concepts = []
                for i in range(0, concept_symbols_considered, _CONCEPT_BATCH_SIZE):
                    batch = ranked[i : i + _CONCEPT_BATCH_SIZE]
                    concepts.extend(extractor.extract_from_symbols(batch))
                if concepts:
                    save_concepts(self._db, concepts)
                    concept_count = len(concepts)
                    logger.info("Extracted %d semantic concepts", concept_count)

        elapsed = time.perf_counter() - start
        logger.info("Graph built in %.2fs", elapsed)

        return GraphBuildResult(
            code_graph=cg,
            community_count=community_count,
            node_count=cg.node_count,
            edge_count=cg.edge_count,
            concept_count=concept_count,
            elapsed_seconds=elapsed,
            community_summary=community_summary,
            partition_quality=quality,
            concept_symbols_considered=concept_symbols_considered,
            concept_symbols_total=concept_symbols_total,
        )
