"""LLM-powered semantic concept extraction for the knowledge graph."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from trelix.core.config import LLMConfig
from trelix.core.models import Symbol
from trelix.llm.client import ChatMessage
from trelix.llm.factory import build_chat_client
from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.concepts")

_EXTRACT_SYS = """\
You are a code analysis assistant. Extract the key technical concepts from the
provided code symbols. Focus on architectural concepts, design patterns, and
domain logic — NOT individual variable names.

Return ONLY a valid JSON array (no markdown, no prose) with objects:
[{"entity": "concept name", "importance": 1-5,
 "category": "security|concept|pattern|architecture|domain|misc"}]

Rules:
- Normalize names to lowercase
- importance 5 = core architectural concept, 1 = trivial implementation detail
- Maximum 8 concepts per batch
- Return [] if no meaningful concepts found
"""

_DDL_CONCEPTS = """
CREATE TABLE IF NOT EXISTS graph_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'concept',
    importance INTEGER NOT NULL DEFAULT 3,
    source_symbol_ids TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_graph_concepts_name ON graph_concepts(name);
"""


@dataclass
class SemanticConcept:
    name: str
    category: str
    importance: int
    source_symbol_ids: list[int] = field(default_factory=list)


class ConceptExtractor:
    """Extract semantic concepts from code symbols using an LLM."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._client = build_chat_client(llm_config)

    def extract_from_symbols(
        self,
        symbols: list[Symbol],
        max_symbols: int = 20,
    ) -> list[SemanticConcept]:
        """Extract concepts from a batch of symbols. Returns [] on any failure."""
        if not symbols:
            return []

        batch = symbols[:max_symbols]
        code_context = "\n\n".join(
            f"# {s.qualified_name} ({s.kind.value})\n{s.signature}\n{s.body[:300]}" for s in batch
        )
        source_ids = [s.id for s in batch if s.id is not None]

        try:
            response = self._client.complete(
                messages=[
                    ChatMessage(
                        role="user",
                        content=f"Extract concepts from these code symbols:\n\n{code_context}",
                    )
                ],
                system=_EXTRACT_SYS,
            )
            parsed = json.loads(response.content)
            if not isinstance(parsed, list):
                return []
            result = []
            for item in parsed:
                if not isinstance(item, dict) or "entity" not in item:
                    continue
                result.append(
                    SemanticConcept(
                        name=str(item["entity"]).lower().strip(),
                        category=str(item.get("category", "concept")),
                        importance=int(item.get("importance", 3)),
                        source_symbol_ids=list(source_ids),
                    )
                )
            return result
        except Exception as exc:
            logger.debug("ConceptExtractor failed: %s", exc)
            return []

    def extract_from_file_summary(
        self,
        summary: str,
        file_id: int,
    ) -> list[SemanticConcept]:
        """Extract concepts from a RAPTOR-style file summary."""
        if not summary.strip():
            return []
        try:
            response = self._client.complete(
                messages=[
                    ChatMessage(
                        role="user",
                        content=f"Extract concepts from this file summary:\n\n{summary}",
                    )
                ],
                system=_EXTRACT_SYS,
            )
            parsed = json.loads(response.content)
            if not isinstance(parsed, list):
                return []
            return [
                SemanticConcept(
                    name=str(item["entity"]).lower().strip(),
                    category=str(item.get("category", "concept")),
                    importance=int(item.get("importance", 3)),
                    source_symbol_ids=[file_id],
                )
                for item in parsed
                if isinstance(item, dict) and "entity" in item
            ]
        except Exception as exc:
            logger.debug("ConceptExtractor.extract_from_file_summary failed: %s", exc)
            return []


def save_concepts(db: Database, concepts: list[SemanticConcept]) -> None:
    """Persist concepts to graph_concepts table (insert-or-ignore by name)."""
    db._conn.executescript(_DDL_CONCEPTS)
    db._conn.executemany(
        """
        INSERT INTO graph_concepts (name, category, importance, source_symbol_ids)
        VALUES (?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        [(c.name, c.category, c.importance, json.dumps(c.source_symbol_ids)) for c in concepts],
    )
    db._conn.commit()


def load_concepts(db: Database) -> list[SemanticConcept]:
    """Load all persisted concepts from DB. Returns [] if table does not exist yet."""
    try:
        rows = db._conn.execute(
            "SELECT name, category, importance, source_symbol_ids FROM graph_concepts"
        ).fetchall()
        return [
            SemanticConcept(
                name=row[0],
                category=row[1],
                importance=row[2],
                source_symbol_ids=json.loads(row[3] or "[]"),
            )
            for row in rows
        ]
    except Exception:
        return []
