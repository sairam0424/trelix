"""
Connector registry — get_artifact_source(name) resolves a connector name to
an instantiated ArtifactSource, mirroring the match-statement factory idiom
already used by make_embedder() (embedder/base.py) and make_vector_store().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from trelix.core.config import (
    JiraConnectorConfig,
    LinearConnectorConfig,
    TestRailConnectorConfig,
    XrayConnectorConfig,
)
from trelix.indexing.connectors.base import ArtifactSource

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig

ConnectorName = Literal["jira", "testrail", "xray", "linear", "diagram"]


def get_artifact_source(
    name: ConnectorName, index_config: IndexConfig | None = None
) -> ArtifactSource:
    """Instantiate the named connector, reading its config from env/`.env`
    (each connector's Config class handles that itself — see
    core/config.py's JiraConnectorConfig/TestRailConnectorConfig/
    XrayConnectorConfig/LinearConnectorConfig).

    ``index_config`` is only consulted by "diagram" — the other four are
    remote-API connectors fully configured by their own env vars and never
    need the indexed repo's path or LLM provider; "diagram" reads local
    files under ``index_config.repo_path`` and captions them via
    ``index_config.llm``, so it is required for that one case."""
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
        case "linear":
            from trelix.indexing.connectors.linear import LinearConnector

            return LinearConnector(LinearConnectorConfig())
        case "diagram":
            from trelix.indexing.connectors.diagram import DiagramConnector

            if index_config is None:
                raise ValueError("get_artifact_source('diagram') requires index_config")
            return DiagramConnector(index_config)
        case _:
            raise ValueError(
                f"Unknown connector: {name!r}. Expected 'jira', 'testrail', 'xray', "
                "'linear', or 'diagram'."
            )
