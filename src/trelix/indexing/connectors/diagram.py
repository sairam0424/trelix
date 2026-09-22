"""
DiagramConnector — indexes local `.drawio` (diagrams.net/draw.io) files as
Artifacts, captioned via the existing text-only TrelixChatClient.

Scoped v1 (see docs/ROADMAP.md's "Multi-modal" entry): `.drawio` is plain XML
(the mxGraph format) — every shape's label lives in a `value="..."`
attribute and every connection is an `<mxCell edge="1" source="..."
target="...">`, so the diagram's structure is fully describable as TEXT.
This needs zero new vision/multimodal provider code; it reuses the same
TrelixChatClient/build_chat_client path every other LLM call site in trelix
already uses. Raster images (.png/.jpg) genuinely need a vision-capable
call and are deliberately OUT of scope here — a separate, larger change with
its own new provider code to review, not a natural extension of this one.

Deliberately mirrors the connector pattern (ArtifactSource, jira.py/
testrail.py) rather than the full Symbol/Chunk/vector pipeline: a captioned
diagram becomes an Artifact linked into generic_edges via ArtifactLinker,
exactly like a Jira ticket or TestRail case — never a Chunk, so
chunks.symbol_id's hard NOT NULL FK and the walker's language allow-list are
both completely untouched. See Artifact's docstring in core/models.py for
why that boundary exists.
"""

from __future__ import annotations

import logging
from pathlib import Path

from trelix.core.config import IndexConfig
from trelix.core.models import Artifact
from trelix.indexing.connectors.base import ArtifactSource
from trelix.llm.client import ChatMessage
from trelix.llm.factory import build_chat_client

logger = logging.getLogger("trelix.indexing.connectors.diagram")

# Directories never worth walking into for diagram files — mirrors the spirit
# of WalkerConfig's own ignore list without pulling in the full FileWalker
# (which would require .drawio to become a recognized Language, exactly what
# this connector's whole design is meant to avoid).
_SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "venv", "__pycache__", ".trelix"})

_CAPTION_PROMPT = """\
Describe this diagram (draw.io/diagrams.net XML source below) for a code
search index. Focus on: what the diagram represents (architecture, flow,
sequence, etc.), the named shapes/components, and how they connect to each
other. Output ONLY the description, 2-4 sentences, no preamble, no markdown.

Diagram XML:
{xml}
"""

# Cap on how much raw XML goes into the prompt -- a pathological/huge .drawio
# export (embedded images as base64 data URIs are a known drawio pattern)
# should not blow the LLM's context window silently.
_MAX_XML_CHARS = 20_000


class DiagramConnector(ArtifactSource):
    """See module docstring. Constructed with the same IndexConfig every
    other pipeline stage uses -- repo_path for discovery, .llm for
    captioning, no separate connector config class needed."""

    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        self._repo_path = Path(config.repo_path).resolve()

    def validate_config(self) -> None:
        if not self._repo_path.is_dir():
            raise ValueError(f"DiagramConnector: repo_path is not a directory: {self._repo_path}")

    def fetch(self) -> list[Artifact]:
        self.validate_config()
        chat_client = build_chat_client(self._config.llm)

        artifacts: list[Artifact] = []
        for path in self._discover():
            try:
                xml = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("DiagramConnector: could not read %s (%s); skipping", path, exc)
                continue
            artifacts.append(self._file_to_artifact(path, xml, chat_client))
        return artifacts

    def _discover(self) -> list[Path]:
        found: list[Path] = []
        for path in self._repo_path.rglob("*.drawio"):
            if any(part in _SKIP_DIRS for part in path.relative_to(self._repo_path).parts):
                continue
            found.append(path)
        return sorted(found)

    def _file_to_artifact(self, path: Path, xml: str, chat_client: object) -> Artifact:
        rel_path = path.relative_to(self._repo_path).as_posix()
        caption = self._caption(xml, chat_client)
        return Artifact(
            source_ref=f"diagram:{rel_path}",
            artifact_kind="diagram",
            title=path.stem,
            body=caption,
            url=None,
            metadata={"path": rel_path},
        )

    def _caption(self, xml: str, chat_client: object) -> str:
        """One LLM call per diagram. Same graceful-degradation contract as
        the compressors: a captioning failure never drops the diagram
        entirely -- it falls back to a mechanical description so the
        artifact is still indexed and linkable, just without a synthesized
        summary (never worse-without-a-trace)."""
        truncated = xml[:_MAX_XML_CHARS]
        try:
            response = chat_client.complete(  # type: ignore[attr-defined]
                [ChatMessage(role="user", content=_CAPTION_PROMPT.format(xml=truncated))],
                max_tokens=256,
                temperature=0.0,
            )
            caption = (response.content or "").strip()
            if caption:
                return caption
        except Exception as exc:  # noqa: BLE001 — graceful degradation (compressor contract)
            logger.warning("DiagramConnector: captioning failed (%s); using raw XML fallback", exc)
        return f"[trelix: captioning unavailable] raw diagram source, {len(xml)} chars of XML."
