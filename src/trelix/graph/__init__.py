"""trelix Knowledge Graph — unified code property graph over indexed codebases."""

from trelix.graph.builder import GraphBuildResult, GraphBuilder
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import assign_communities, detect_communities, get_community_summary
from trelix.graph.concepts import ConceptExtractor, SemanticConcept, load_concepts, save_concepts
from trelix.graph.persistence import load_graph_metadata, save_graph_metadata

__all__ = [
    "CodeGraph",
    "GraphBuilder",
    "GraphBuildResult",
    "ConceptExtractor",
    "SemanticConcept",
    "detect_communities",
    "assign_communities",
    "get_community_summary",
    "save_graph_metadata",
    "load_graph_metadata",
    "save_concepts",
    "load_concepts",
]
