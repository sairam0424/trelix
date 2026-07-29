"""
Connector registry — get_artifact_source(name) resolves a connector name to
an instantiated ArtifactSource, mirroring the match-statement factory idiom
already used by make_embedder() (embedder/base.py) and make_vector_store().
"""

from __future__ import annotations

from typing import Literal

from trelix.core.config import JiraConnectorConfig, TestRailConnectorConfig, XrayConnectorConfig
from trelix.indexing.connectors.base import ArtifactSource

ConnectorName = Literal["jira", "testrail", "xray"]


def get_artifact_source(name: ConnectorName) -> ArtifactSource:
    """Instantiate the named connector, reading its config from env/`.env`
    (each connector's Config class handles that itself — see
    core/config.py's JiraConnectorConfig/TestRailConnectorConfig/
    XrayConnectorConfig)."""
    match name:
        case "jira":
            from trelix.indexing.connectors.jira import JiraConnector

            return JiraConnector(JiraConnectorConfig())
        case "testrail":
            from trelix.indexing.connectors.testrail import TestRailConnector

            return TestRailConnector(TestRailConnectorConfig())
        case "xray":
            from trelix.indexing.connectors.xray import XrayConnector

            return XrayConnector(XrayConnectorConfig())
        case _:
            raise ValueError(
                f"Unknown connector: {name!r}. Expected 'jira', 'testrail', or 'xray'."
            )
