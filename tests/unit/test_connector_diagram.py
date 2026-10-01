"""
Unit tests for DiagramConnector (ArtifactSource for local .drawio files).

Covers: discovery (skip-dirs, non-.drawio files ignored, sorted/deterministic
order), the happy captioning path (via a hand-written fake TrelixChatClient,
never a real LLM call), graceful degradation on a captioning failure or an
empty LLM response, validate_config on a missing/non-existent repo path, and
the artifact shape (source_ref/artifact_kind/metadata) ArtifactLinker and
generic_edges expect.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from trelix.core.config import IndexConfig
from trelix.indexing.connectors.diagram import DiagramConnector
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient

_DRAWIO_XML = """\
<mxfile>
  <diagram name="Architecture">
    <mxGraphModel>
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="api" value="API Gateway" vertex="1" parent="1">
          <mxGeometry x="40" y="40" width="120" height="60" as="geometry" />
        </mxCell>
        <mxCell id="db" value="Postgres Database" vertex="1" parent="1">
          <mxGeometry x="240" y="40" width="120" height="60" as="geometry" />
        </mxCell>
        <mxCell id="edge1" edge="1" source="api" target="db" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


class _StubChatClient(TrelixChatClient):
    """Hand-written fake, matching the established convention (never
    Mock — see tests/unit/test_chunker_contextual_llm_gaps.py)."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
    ) -> ChatResponse:
        self.calls.append({"messages": messages})
        return ChatResponse(content=self._content, model="stub-model", finish_reason="stop")

    def stream(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("DiagramConnector never calls stream()")

    def tool_call(self, *args: object, **kwargs: object) -> ToolCallResponse:
        raise AssertionError("DiagramConnector never calls tool_call()")


class _RaisingChatClient(TrelixChatClient):
    def complete(self, *args: object, **kwargs: object) -> ChatResponse:
        raise RuntimeError("simulated provider outage")

    def stream(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("not called in these tests")

    def tool_call(self, *args: object, **kwargs: object) -> ToolCallResponse:
        raise AssertionError("not called in these tests")


def _write_drawio(repo: Path, rel_path: str, xml: str = _DRAWIO_XML) -> Path:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    return path


def _connector(repo: Path) -> DiagramConnector:
    return DiagramConnector(IndexConfig(repo_path=str(repo)))


def test_validate_config_raises_when_repo_path_is_a_file_not_a_directory(tmp_path: Path) -> None:
    """IndexConfig's own validator only checks Path.exists(), not is_dir() --
    a repo_path pointing at an existing FILE passes IndexConfig construction
    but must still be caught here before DiagramConnector tries to rglob it."""
    a_file = tmp_path / "not-a-directory.txt"
    a_file.write_text("x")
    connector = _connector(a_file)
    with pytest.raises(ValueError, match="not a directory"):
        connector.validate_config()


def test_validate_config_passes_for_a_real_directory(tmp_path: Path) -> None:
    _connector(tmp_path).validate_config()  # must not raise


def test_discover_finds_drawio_files_recursively(tmp_path: Path) -> None:
    _write_drawio(tmp_path, "docs/a.drawio")
    _write_drawio(tmp_path, "docs/nested/b.drawio")
    connector = _connector(tmp_path)

    found = connector._discover()

    assert {p.name for p in found} == {"a.drawio", "b.drawio"}


def test_discover_ignores_non_drawio_files(tmp_path: Path) -> None:
    _write_drawio(tmp_path, "docs/a.drawio")
    (tmp_path / "docs" / "readme.md").write_text("not a diagram")
    connector = _connector(tmp_path)

    found = connector._discover()

    assert len(found) == 1
    assert found[0].name == "a.drawio"


@pytest.mark.parametrize("skip_dir", [".git", "node_modules", ".venv", ".trelix"])
def test_discover_skips_ignored_directories(tmp_path: Path, skip_dir: str) -> None:
    _write_drawio(tmp_path, f"{skip_dir}/should-not-be-found.drawio")
    _write_drawio(tmp_path, "docs/should-be-found.drawio")
    connector = _connector(tmp_path)

    found = connector._discover()

    assert [p.name for p in found] == ["should-be-found.drawio"]


def test_discover_returns_a_deterministic_sorted_order(tmp_path: Path) -> None:
    _write_drawio(tmp_path, "z.drawio")
    _write_drawio(tmp_path, "a.drawio")
    connector = _connector(tmp_path)

    found = connector._discover()

    assert [p.name for p in found] == ["a.drawio", "z.drawio"]


def test_fetch_with_no_drawio_files_returns_empty_list(tmp_path: Path) -> None:
    connector = _connector(tmp_path)
    with patch("trelix.indexing.connectors.diagram.build_chat_client") as mock_build:
        artifacts = connector.fetch()
    assert artifacts == []
    mock_build.assert_called_once()  # constructed even with nothing to caption


def test_fetch_produces_an_artifact_with_the_real_llm_caption(tmp_path: Path) -> None:
    _write_drawio(tmp_path, "docs/architecture.drawio")
    connector = _connector(tmp_path)
    client = _StubChatClient("An API gateway connected to a Postgres database.")

    with patch("trelix.indexing.connectors.diagram.build_chat_client", return_value=client):
        artifacts = connector.fetch()

    assert len(artifacts) == 1
    a = artifacts[0]
    assert a.source_ref == "diagram:docs/architecture.drawio"
    assert a.artifact_kind == "diagram"
    assert a.title == "architecture"
    assert a.body == "An API gateway connected to a Postgres database."
    assert a.url is None
    assert a.metadata == {"path": "docs/architecture.drawio"}
    # The raw XML actually reached the prompt sent to the chat client.
    assert "API Gateway" in client.calls[0]["messages"][0].content
    assert "Postgres Database" in client.calls[0]["messages"][0].content


def test_fetch_degrades_gracefully_when_captioning_raises(tmp_path: Path) -> None:
    _write_drawio(tmp_path, "docs/architecture.drawio")
    connector = _connector(tmp_path)

    with patch(
        "trelix.indexing.connectors.diagram.build_chat_client",
        return_value=_RaisingChatClient(),
    ):
        artifacts = connector.fetch()

    assert len(artifacts) == 1  # still produced -- never dropped
    assert "captioning unavailable" in artifacts[0].body
    # The docstring's "still indexed and linkable" promise requires the
    # fallback to carry real content, not just a length -- ArtifactLinker's
    # regex name-matching has nothing to match against otherwise.
    assert "API Gateway" in artifacts[0].body
    assert "Postgres Database" in artifacts[0].body
    assert artifacts[0].source_ref == "diagram:docs/architecture.drawio"


def test_fetch_degrades_gracefully_on_empty_llm_response(tmp_path: Path) -> None:
    _write_drawio(tmp_path, "docs/architecture.drawio")
    connector = _connector(tmp_path)
    client = _StubChatClient("   ")  # whitespace-only

    with patch("trelix.indexing.connectors.diagram.build_chat_client", return_value=client):
        artifacts = connector.fetch()

    assert "captioning unavailable" in artifacts[0].body
    assert "API Gateway" in artifacts[0].body
    assert "Postgres Database" in artifacts[0].body


def test_mechanical_fallback_strips_inline_html_and_entities_from_labels() -> None:
    """drawio commonly stores rich-text labels as inline HTML with escaped
    entities (e.g. value="&lt;b&gt;Auth Service&lt;/b&gt;"). The fallback
    must surface the readable text, not the markup."""
    from trelix.indexing.connectors.diagram import _mechanical_description

    xml = '<mxCell value="&lt;b&gt;Auth Service&lt;/b&gt; &amp; friends" />'

    description = _mechanical_description(xml)

    assert "Auth Service & friends" in description
    assert "<b>" not in description
    assert "&lt;" not in description


def test_mechanical_fallback_deduplicates_labels_in_first_seen_order() -> None:
    from trelix.indexing.connectors.diagram import _mechanical_description

    xml = '<mxCell value="API Gateway" /><mxCell value="Database" /><mxCell value="API Gateway" />'

    description = _mechanical_description(xml)

    assert description.count("API Gateway") == 1
    assert description.index("API Gateway") < description.index("Database")


def test_mechanical_fallback_reports_length_when_no_labels_found() -> None:
    from trelix.indexing.connectors.diagram import _mechanical_description

    xml = "<mxfile><diagram></diagram></mxfile>"

    description = _mechanical_description(xml)

    assert "captioning unavailable" in description
    assert str(len(xml)) in description


def test_fetch_handles_multiple_diagrams_independently(tmp_path: Path) -> None:
    """One diagram's captioning failure must not affect another's success."""
    _write_drawio(tmp_path, "a.drawio")
    _write_drawio(tmp_path, "b.drawio")
    connector = _connector(tmp_path)

    call_count = 0

    class _FlakyChatClient(TrelixChatClient):
        def complete(self, *args: object, **kwargs: object) -> ChatResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated outage on the first diagram only")
            return ChatResponse(content="caption for b", model="stub", finish_reason="stop")

        def stream(self, *args: object, **kwargs: object) -> Any:
            raise AssertionError("not used")

        def tool_call(self, *args: object, **kwargs: object) -> ToolCallResponse:
            raise AssertionError("not used")

    with patch(
        "trelix.indexing.connectors.diagram.build_chat_client",
        return_value=_FlakyChatClient(),
    ):
        artifacts = connector.fetch()

    assert len(artifacts) == 2
    bodies = {a.source_ref: a.body for a in artifacts}
    assert "captioning unavailable" in bodies["diagram:a.drawio"]
    assert bodies["diagram:b.drawio"] == "caption for b"


def test_unreadable_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A file that disappears (or is permission-denied) between discovery
    and read must not abort the whole sync -- mirrors the connector base
    class's own "one bad item doesn't sink the batch" posture."""
    good = _write_drawio(tmp_path, "good.drawio")
    connector = _connector(tmp_path)
    client = _StubChatClient("caption")

    original_read_text = Path.read_text

    def _flaky_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "bad.drawio":
            raise OSError("simulated permission denied")
        return original_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    bad = tmp_path / "bad.drawio"
    bad.write_text(_DRAWIO_XML, encoding="utf-8")

    with (
        patch("trelix.indexing.connectors.diagram.build_chat_client", return_value=client),
        patch.object(Path, "read_text", _flaky_read_text),
    ):
        artifacts = connector.fetch()

    assert {a.source_ref for a in artifacts} == {f"diagram:{good.name}"}


# ---------------------------------------------------------------------------
# symlink containment -- mirrors ImageConnector (image.py `_is_within_repo`)
# ---------------------------------------------------------------------------

_SECRET_MARKER = "FAKE-SECRET-MARKER-for-tests"


def test_symlink_escaping_the_repo_is_not_synced_and_never_reaches_the_llm(
    tmp_path: Path,
) -> None:
    """CRITICAL regression: read_text() follows a symlink, so an arch.drawio
    -> /some/outside/file link used to have its target's first 20000 chars
    sent to the configured chat client inside the caption prompt."""
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text(_SECRET_MARKER, encoding="utf-8")
    (repo / "arch.drawio").symlink_to(outside)
    _write_drawio(repo, "real.drawio")
    client = _StubChatClient("caption")

    with patch("trelix.indexing.connectors.diagram.build_chat_client", return_value=client):
        artifacts = _connector(repo).fetch()

    assert [a.source_ref for a in artifacts] == ["diagram:real.drawio"]
    assert len(client.calls) == 1
    assert all(_SECRET_MARKER not in call["messages"][0].content for call in client.calls)


def test_discover_skips_a_symlink_that_escapes_the_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text(_SECRET_MARKER, encoding="utf-8")
    (repo / "arch.drawio").symlink_to(outside)

    assert _connector(repo)._discover() == []


def test_discover_still_finds_a_symlink_that_resolves_inside_the_repo(tmp_path: Path) -> None:
    """Containment, not a blanket symlink ban."""
    repo = tmp_path / "repo"
    _write_drawio(repo, "real/diagram.drawio")
    (repo / "alias.drawio").symlink_to(repo / "real" / "diagram.drawio")

    found = _connector(repo)._discover()

    assert repo / "alias.drawio" in found
    assert repo / "real" / "diagram.drawio" in found


def test_in_repo_symlink_is_synced_with_its_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_drawio(repo, "real/diagram.drawio")
    (repo / "alias.drawio").symlink_to(repo / "real" / "diagram.drawio")
    client = _StubChatClient("caption")

    with patch("trelix.indexing.connectors.diagram.build_chat_client", return_value=client):
        artifacts = _connector(repo).fetch()

    assert {a.source_ref for a in artifacts} == {
        "diagram:alias.drawio",
        "diagram:real/diagram.drawio",
    }
    assert all("API Gateway" in call["messages"][0].content for call in client.calls)


def test_discover_does_not_crash_on_a_symlink_loop(tmp_path: Path) -> None:
    """resolve() on a loop raises RuntimeError on some platforms/Pythons and
    returns a path on others, so only the no-crash property is asserted."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_drawio(repo, "real.drawio")
    (repo / "loop_a.drawio").symlink_to(repo / "loop_b.drawio")
    (repo / "loop_b.drawio").symlink_to(repo / "loop_a.drawio")

    found = _connector(repo)._discover()  # must not raise

    assert repo / "real.drawio" in found


def test_discover_skips_a_dangling_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_drawio(repo, "real.drawio")
    (repo / "dangling.drawio").symlink_to(tmp_path / "does-not-exist")

    assert _connector(repo)._discover() == [repo / "real.drawio"]


def test_discover_skips_a_directory_named_like_a_drawio_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "folder.drawio").mkdir(parents=True)
    _write_drawio(repo, "real.drawio")

    assert _connector(repo)._discover() == [repo / "real.drawio"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX named pipes required")
def test_discover_skips_a_non_regular_file(tmp_path: Path) -> None:
    """A FIFO named *.drawio would block read_text() forever; discovery only
    (never fetch) is exercised so a regression fails instead of hanging."""
    repo = tmp_path / "repo"
    repo.mkdir()
    os.mkfifo(repo / "pipe.drawio")  # type: ignore[attr-defined]
    _write_drawio(repo, "real.drawio")

    assert _connector(repo)._discover() == [repo / "real.drawio"]


def test_discover_logs_one_warning_with_the_skipped_symlink_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text(_SECRET_MARKER, encoding="utf-8")
    (repo / "a.drawio").symlink_to(outside)
    (repo / "b.drawio").symlink_to(outside)
    (repo / "dangling.drawio").symlink_to(tmp_path / "does-not-exist")
    _write_drawio(repo, "real.drawio")

    with caplog.at_level(logging.WARNING, logger="trelix.indexing.connectors.diagram"):
        _connector(repo)._discover()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "3" in warnings[0].getMessage()
    assert _SECRET_MARKER not in caplog.text


def test_discover_logs_no_warning_when_nothing_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_drawio(tmp_path, "real.drawio")

    with caplog.at_level(logging.WARNING, logger="trelix.indexing.connectors.diagram"):
        _connector(tmp_path)._discover()

    assert caplog.records == []


def test_containment_does_not_depend_on_walker_follow_symlinks(tmp_path: Path) -> None:
    """Containment is unconditional (unlike WalkerConfig.follow_symlinks)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text(_SECRET_MARKER, encoding="utf-8")
    (repo / "arch.drawio").symlink_to(outside)
    config = IndexConfig(repo_path=str(repo))
    config.walker.follow_symlinks = True

    assert DiagramConnector(config)._discover() == []


def test_discover_skips_a_relative_symlink_that_escapes_with_dotdot(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "outside_secret.txt").write_text(_SECRET_MARKER, encoding="utf-8")
    (repo / "arch.drawio").symlink_to("../outside_secret.txt")

    assert _connector(repo)._discover() == []


def test_discover_skips_a_symlink_chain_that_ends_outside_the_repo(tmp_path: Path) -> None:
    """Only the fully resolved target counts: hop1 sits and points inside the
    repo, so it must not pass on the strength of its own location."""
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text(_SECRET_MARKER, encoding="utf-8")
    (repo / "hop2.drawio").symlink_to(outside)
    (repo / "hop1.drawio").symlink_to(repo / "hop2.drawio")
    _write_drawio(repo, "real.drawio")

    assert _connector(repo)._discover() == [repo / "real.drawio"]


def test_discover_skips_a_symlink_into_a_sibling_sharing_the_repo_name_prefix(
    tmp_path: Path,
) -> None:
    """`repo-evil` starts with the string `repo`; containment must compare path
    components, not string prefixes."""
    repo = tmp_path / "repo"
    repo.mkdir()
    secret = _write_drawio(tmp_path / "repo-evil", "secret.drawio")
    (repo / "arch.drawio").symlink_to(secret)

    assert _connector(repo)._discover() == []


def test_discover_does_not_descend_into_a_symlinked_directory(tmp_path: Path) -> None:
    """A .drawio reached through a symlinked parent directory is not itself a
    symlink, so the per-file containment check would not catch it; this pins
    the assumption that rglob() does not descend into symlinked directories."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_drawio(tmp_path / "outside_dir", "secret.drawio")
    (repo / "linked_dir").symlink_to(tmp_path / "outside_dir", target_is_directory=True)
    _write_drawio(repo, "real.drawio")

    assert _connector(repo)._discover() == [repo / "real.drawio"]
