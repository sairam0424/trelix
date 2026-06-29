"""GraphBuilder — orchestrates the full knowledge graph construction pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trelix.core.config import IndexConfig
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import assign_communities, detect_communities, get_community_summary
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

        # Step 3: persist metadata
        save_graph_metadata(self._db, cg)

        # Step 4: optional concept extraction
        concept_count = 0
        if extract_concepts:
            symbols_with_files = self._db.iter_all_symbols_with_files()
            symbols = [s for s, _ in symbols_with_files]
            if symbols:
                extractor = ConceptExtractor(self._config.llm)
                # Batch into groups of 20, cap at 200 symbols total
                concepts = []
                for i in range(0, min(len(symbols), 200), 20):
                    batch = symbols[i : i + 20]
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
        )
