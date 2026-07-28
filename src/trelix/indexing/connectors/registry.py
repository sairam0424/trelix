"""
Connector registry — get_artifact_source(name) resolves a connector name to
an instantiated ArtifactSource, mirroring the match-statement factory idiom
already used by make_embedder() (embedder/base.py) and make_vector_store().
"""

from __future__ import annotations

from typing import Literal

from trelix.core.config import JiraConnectorConfig, TestRailConnectorConfig
from trelix.indexing.connectors.base import ArtifactSource

ConnectorName = Literal["jira", "testrail"]


def get_artifact_source(name: ConnectorName) -> ArtifactSource:
    """Instantiate the named connector, reading its config from env/`.env`
    (each connector's Config class handles that itself — see
    core/config.py's JiraConnectorConfig/TestRailConnectorConfig)."""
    match name:
        case "jira":
            from trelix.indexing.connectors.jira import JiraConnector

            return JiraConnector(JiraConnectorConfig())
        case "testrail":
            from trelix.indexing.connectors.testrail import TestRailConnector

            return TestRailConnector(TestRailConnectorConfig())
        case _:
            raise ValueError(f"Unknown connector: {name!r}. Expected 'jira' or 'testrail'.")
