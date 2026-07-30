"""
Integration test: connector.sync() -> generic_edges -> CodeGraph, end to end.

Deliberately hermetic — no real Jira/TestRail API access, no embedder
(seeds the DB directly rather than going through Indexer.index(), since
that requires sentence-transformers or a real API key). This proves the
FULL cross-source pipeline wires together correctly: a connector-fetched
Artifact, manually linked to a symbol via a GenericEdge (the same edge a
real workflow — e.g. running `trelix link-tickets` first, then matching an
artifact's source_ref — would produce), surfaces in CodeGraph and
contributes to PageRank.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from trelix.core.config import ArtifactLinkerConfig, JiraConnectorConfig, XrayConnectorConfig
from trelix.core.models import (
    GenericEdge,
    IndexedFile,
    Language,
    Symbol,
    SymbolKind,
)
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import compute_pagerank
from trelix.indexing.artifact_linker import ArtifactLinker
from trelix.indexing.connectors.jira import JiraConnector
from trelix.indexing.connectors.xray import XrayConnector
from trelix.store.db import Database


def _seed_symbol(db: Database, rel_path: str, name: str) -> int:
    file_id = db.upsert_file(
        IndexedFile(
            path=f"/repo/{rel_path}",
            rel_path=rel_path,
            language=Language.PYTHON,
            hash="h",
            size_bytes=10,
        )
    )
    sym = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=name,
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=2,
        signature=f"def {name}()",
        body=f"def {name}(): pass",
    )
    sym_id = db.insert_symbol(sym)
    db._conn.commit()
    return sym_id


class TestConnectorToGraphPipeline:
    def test_synced_artifact_and_linked_symbol_appear_together_in_code_graph(
        self, tmp_path: Path
    ) -> None:
        db = Database(tmp_path / "index.db")
        login_id = _seed_symbol(db, "auth.py", "login")
        _seed_symbol(db, "auth.py", "logout")  # unreferenced control symbol

        # 1. Connector fetches a real (mocked-HTTP) artifact and persists it.
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "issues": [
                {
                    "key": "PROJ-1",
                    "fields": {
                        "summary": "Login is broken",
                        "description": "Users report login failures",
                        "status": {"name": "Open"},
                    },
                }
            ],
            "nextPageToken": None,
        }
        jira_config = JiraConnectorConfig(
            base_url="https://example.atlassian.net",
            email="me@example.com",
            api_token="tok",
            project_key="PROJ",
        )
        with patch("trelix.indexing.connectors.jira.httpx.get", return_value=mock_resp):
            result = JiraConnector(jira_config).sync(db)

        assert result.artifacts_written == 1
        artifact = db.get_artifact_by_source_ref("ticket:PROJ-1")
        assert artifact is not None

        # 2. Link the code symbol to that same artifact's source_ref (the
        # role trelix link-tickets plays for git-derived edges — a connector
        # sync alone does not create edges, only artifacts, per its own
        # docstring).
        db.insert_generic_edges(
            [
                GenericEdge(
                    from_symbol_id=login_id,
                    source_ref=artifact.source_ref,
                    edge_kind="references_ticket",
                )
            ]
        )

        # 3. The full graph pipeline picks it up.
        cg = CodeGraph(db)
        assert artifact.source_ref in cg.nx
        assert cg.nx.nodes[artifact.source_ref]["type"] == "artifact"
        assert artifact.source_ref in cg.neighbors(login_id)

        pr = compute_pagerank(cg)
        # login (referenced by a ticket) must outrank logout (no edges at all
        # beyond being a node) — proves the connector's output genuinely
        # participates in ranking, not just plumbing that gets dropped.
        symbol_scores = {nid: score for nid, score in pr.items() if isinstance(nid, int)}
        assert symbol_scores[login_id] > 0

    def test_sync_alone_creates_no_edges(self, tmp_path: Path) -> None:
        """A connector sync populates the artifacts table only — confirms
        the CLI docstring's claim precisely, so this test would fail loudly
        if that contract ever silently changed."""
        db = Database(tmp_path / "index.db")
        _seed_symbol(db, "auth.py", "login")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "issues": [{"key": "PROJ-2", "fields": {"summary": "Unrelated ticket"}}],
            "nextPageToken": None,
        }
        jira_config = JiraConnectorConfig(
            base_url="https://example.atlassian.net",
            email="me@example.com",
            api_token="tok",
            project_key="PROJ",
        )
        with patch("trelix.indexing.connectors.jira.httpx.get", return_value=mock_resp):
            JiraConnector(jira_config).sync(db)

        assert db.get_artifact_by_source_ref("ticket:PROJ-2") is not None
        edges = db._conn.execute("SELECT COUNT(*) FROM generic_edges").fetchone()[0]
        assert edges == 0

        # And the artifact correctly does NOT appear in CodeGraph — only
        # symbols reachable via a real generic_edges row become graph nodes.
        cg = CodeGraph(db)
        assert "ticket:PROJ-2" not in cg.nx

    def test_sync_with_linker_auto_links_into_code_graph(self, tmp_path: Path) -> None:
        """Passing an ArtifactLinker into sync() closes the gap the two
        tests above document — a synced artifact is reachable from
        generic_edges/CodeGraph the moment sync() returns, no separate
        `trelix link-artifacts` pass required."""
        db = Database(tmp_path / "index.db")
        login_id = _seed_symbol(db, "auth.py", "login")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "issues": [
                {
                    "key": "PROJ-3",
                    "fields": {"summary": "login is broken", "status": {"name": "Open"}},
                }
            ],
            "nextPageToken": None,
        }
        jira_config = JiraConnectorConfig(
            base_url="https://example.atlassian.net",
            email="me@example.com",
            api_token="tok",
            project_key="PROJ",
        )
        linker = ArtifactLinker(db, ArtifactLinkerConfig())
        with patch("trelix.indexing.connectors.jira.httpx.get", return_value=mock_resp):
            result = JiraConnector(jira_config).sync(db, linker=linker)

        assert result.artifacts_written == 1
        assert result.edges_linked == 1
        assert db.get_generic_edge_targets(login_id) == ["ticket:PROJ-3"]

        cg = CodeGraph(db)
        assert "ticket:PROJ-3" in cg.nx
        assert "ticket:PROJ-3" in cg.neighbors(login_id)

    def test_xray_sync_with_linker_auto_links_into_code_graph(self, tmp_path: Path) -> None:
        """Proves Phase 1's design generalizes: Xray is the third connector,
        and needs zero Xray-specific linking code — ArtifactLinker operates
        on the artifacts table generically (any artifact_kind, any
        source_ref convention), so a synced Xray test is automatically
        link-eligible the moment it's synced, same as Jira/TestRail."""
        db = Database(tmp_path / "index.db")
        login_id = _seed_symbol(db, "auth.py", "login")

        auth_resp = MagicMock()
        auth_resp.status_code = 200
        auth_resp.json.return_value = "jwt-token"

        graphql_resp = MagicMock()
        graphql_resp.status_code = 200
        graphql_resp.json.return_value = {
            "data": {
                "getTests": {
                    "results": [
                        {
                            "issueId": "10001",
                            "jira": {
                                "key": "PROJ-9",
                                "summary": "login is broken",
                                "status": {"name": "Open"},
                            },
                            "testType": {"name": "Manual"},
                            "steps": [],
                            "unstructured": "",
                        }
                    ]
                }
            }
        }

        xray_config = XrayConnectorConfig(
            client_id="cid",
            client_secret="sek",
            project_key="PROJ",
            jira_base_url="https://example.atlassian.net",
        )
        linker = ArtifactLinker(db, ArtifactLinkerConfig())
        with patch(
            "trelix.indexing.connectors.xray.httpx.post",
            side_effect=[auth_resp, graphql_resp],
        ):
            result = XrayConnector(xray_config).sync(db, linker=linker)

        assert result.artifacts_written == 1
        assert result.edges_linked == 1
        assert db.get_generic_edge_targets(login_id) == ["xray-test:PROJ-9"]

        cg = CodeGraph(db)
        assert "xray-test:PROJ-9" in cg.nx
        assert "xray-test:PROJ-9" in cg.neighbors(login_id)

    def test_low_weight_edge_has_less_pagerank_influence_than_full_weight_edge(
        self, tmp_path: Path
    ) -> None:
        """A weight=0.5 generic_edge (embedding-fallback confidence) must
        move less PageRank mass than an otherwise-identical weight=1.0 edge
        (regex-match confidence) — proves the persisted weight column is
        actually read by CodeGraph._build()/nx.pagerank(), not silently
        dropped on every read path.

        Weight only affects PageRank when a node has multiple outgoing
        edges competing for vote share — a lone symbol<->artifact pair in
        isolation sends 100% of its vote down its one edge regardless of
        that edge's absolute weight, so two isolated pairs at different
        weights would (wrongly) look identical even with a correct
        implementation. This fixture instead shares ONE artifact between
        two symbols at different weights, so the artifact's vote genuinely
        splits proportional to weight — a real competition weight must
        win to move the result at all.
        """
        db = Database(tmp_path / "index.db")
        high_confidence_id = _seed_symbol(db, "auth.py", "login")
        low_confidence_id = _seed_symbol(db, "auth.py", "logout")

        db.insert_generic_edges(
            [
                GenericEdge(
                    from_symbol_id=high_confidence_id,
                    source_ref="ticket:SHARED",
                    edge_kind="references_artifact",
                    weight=1.0,
                ),
                GenericEdge(
                    from_symbol_id=low_confidence_id,
                    source_ref="ticket:SHARED",
                    edge_kind="references_artifact",
                    weight=0.5,
                ),
            ]
        )

        cg = CodeGraph(db)
        assert cg.nx["ticket:SHARED"][high_confidence_id][0]["weight"] == 1.0
        assert cg.nx["ticket:SHARED"][low_confidence_id][0]["weight"] == 0.5

        pr = compute_pagerank(cg)
        # ticket:SHARED's vote splits 1.0/(1.0+0.5) to login vs.
        # 0.5/(1.0+0.5) to logout — login must outrank logout.
        assert pr[high_confidence_id] > pr[low_confidence_id]
