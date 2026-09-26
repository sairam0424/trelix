"""
Unit tests for ImageConnector (ArtifactSource for local raster images).

Covers: discovery (gitignore-based skipping via the shared
`indexing.gitignore` helper, case-insensitive extension matching, sorted/
deterministic order), the happy captioning path (via a hand-written fake
TrelixChatClient, never a real LLM call) including that the image bytes
actually reach the ChatMessage sent to the client, graceful degradation on a
captioning failure or an empty LLM response, the max_images_per_sync cap,
the vision-provider/model override in `_vision_llm_config`, the dimension/
byte-size downscaling safety caps, validate_config on a missing/non-existent
repo path, and the artifact shape (source_ref/artifact_kind/metadata)
ArtifactLinker and generic_edges expect.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

from trelix.core.config import ImageConnectorConfig, IndexConfig, LLMConfig, WalkerConfig
from trelix.indexing.connectors.image import ImageConnector
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient


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
        raise AssertionError("ImageConnector never calls stream()")

    def tool_call(self, *args: object, **kwargs: object) -> ToolCallResponse:
        raise AssertionError("ImageConnector never calls tool_call()")


class _RaisingChatClient(TrelixChatClient):
    def complete(self, *args: object, **kwargs: object) -> ChatResponse:
        raise RuntimeError("simulated provider outage")

    def stream(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("not called in these tests")

    def tool_call(self, *args: object, **kwargs: object) -> ToolCallResponse:
        raise AssertionError("not called in these tests")


def _write_image(
    repo: Path, rel_path: str, *, size: tuple[int, int] = (8, 4), fmt: str = "PNG"
) -> Path:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(200, 50, 50)).save(path, format=fmt)
    return path


def _connector(repo: Path, **image_kwargs: object) -> ImageConnector:
    """Default `llm` to a provider matching `ImageConnectorConfig`'s own
    default `vision_provider` ("anthropic") -- a real deployment must
    configure this to actually caption anything (validate_config() fails
    fast otherwise, see the validate_config/_vision_llm_config tests
    below), so these tests represent that correctly-configured case rather
    than the mismatched default that motivated the fail-fast check."""
    config = IndexConfig(
        repo_path=str(repo),
        llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-6", _env_file=None),  # type: ignore[call-arg]
        image=ImageConnectorConfig(**image_kwargs),  # type: ignore[arg-type]
    )
    return ImageConnector(config)


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


def test_validate_config_raises_when_repo_path_is_a_file_not_a_directory(tmp_path: Path) -> None:
    a_file = tmp_path / "not-a-directory.txt"
    a_file.write_text("x")
    connector = _connector(a_file)
    with pytest.raises(ValueError, match="not a directory"):
        connector.validate_config()


def test_validate_config_passes_for_a_real_directory(tmp_path: Path) -> None:
    _connector(tmp_path).validate_config()  # must not raise


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discover_finds_images_recursively(tmp_path: Path) -> None:
    _write_image(tmp_path, "docs/a.png")
    _write_image(tmp_path, "docs/nested/b.jpg", fmt="JPEG")
    connector = _connector(tmp_path)

    found = connector._discover()

    assert {p.name for p in found} == {"a.png", "b.jpg"}


def test_discover_matches_extensions_case_insensitively(tmp_path: Path) -> None:
    _write_image(tmp_path, "A.PNG")
    _write_image(tmp_path, "b.JpEg", fmt="JPEG")
    connector = _connector(tmp_path)

    found = connector._discover()

    assert {p.name for p in found} == {"A.PNG", "b.JpEg"}


def test_discover_ignores_non_image_files(tmp_path: Path) -> None:
    _write_image(tmp_path, "docs/a.png")
    (tmp_path / "docs" / "readme.md").write_text("not an image")
    connector = _connector(tmp_path)

    found = connector._discover()

    assert len(found) == 1
    assert found[0].name == "a.png"


def test_discover_respects_gitignore(tmp_path: Path) -> None:
    """No hand-maintained skip-dirs list -- discovery goes through the same
    shared `indexing.gitignore` helper FileWalker uses."""
    (tmp_path / ".gitignore").write_text("ignored/\n")
    _write_image(tmp_path, "ignored/should-not-be-found.png")
    _write_image(tmp_path, "docs/should-be-found.png")
    connector = _connector(tmp_path)

    found = connector._discover()

    assert [p.name for p in found] == ["should-be-found.png"]


def test_discover_can_disable_gitignore_via_walker_config(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored/\n")
    _write_image(tmp_path, "ignored/should-be-found-when-disabled.png")
    config = IndexConfig(repo_path=str(tmp_path), walker=WalkerConfig(respect_gitignore=False))
    connector = ImageConnector(config)

    found = connector._discover()

    assert [p.name for p in found] == ["should-be-found-when-disabled.png"]


def test_discover_returns_a_deterministic_sorted_order(tmp_path: Path) -> None:
    _write_image(tmp_path, "z.png")
    _write_image(tmp_path, "a.png")
    connector = _connector(tmp_path)

    found = connector._discover()

    assert [p.name for p in found] == ["a.png", "z.png"]


# ---------------------------------------------------------------------------
# fetch — happy path / artifact shape
# ---------------------------------------------------------------------------


def test_fetch_with_no_images_returns_empty_list(tmp_path: Path) -> None:
    connector = _connector(tmp_path)
    with patch("trelix.indexing.connectors.image.build_chat_client") as mock_build:
        artifacts = connector.fetch()
    assert artifacts == []
    mock_build.assert_called_once()  # constructed even with nothing to caption


def test_fetch_produces_an_artifact_with_the_real_llm_caption_and_sends_image_bytes(
    tmp_path: Path,
) -> None:
    _write_image(tmp_path, "docs/screenshot.png", size=(8, 4))
    connector = _connector(tmp_path)
    client = _StubChatClient("A red rectangle screenshot.")

    with patch("trelix.indexing.connectors.image.build_chat_client", return_value=client):
        artifacts = connector.fetch()

    assert len(artifacts) == 1
    a = artifacts[0]
    assert a.source_ref == "image:docs/screenshot.png"
    assert a.artifact_kind == "image"
    assert a.title == "screenshot"
    assert a.body == "A red rectangle screenshot."
    assert a.url is None
    assert a.metadata == {"path": "docs/screenshot.png"}

    sent = client.calls[0]["messages"][0]
    assert sent.images is not None
    assert len(sent.images) == 1
    assert sent.images[0].media_type == "image/png"
    assert len(sent.images[0].data) > 0


def test_fetch_degrades_gracefully_when_captioning_raises(tmp_path: Path) -> None:
    _write_image(tmp_path, "docs/screenshot.png", size=(10, 20))
    connector = _connector(tmp_path)

    with patch(
        "trelix.indexing.connectors.image.build_chat_client",
        return_value=_RaisingChatClient(),
    ):
        artifacts = connector.fetch()

    assert len(artifacts) == 1  # still produced -- never dropped
    assert "captioning unavailable" in artifacts[0].body
    assert "screenshot.png" in artifacts[0].body
    assert "10x20px" in artifacts[0].body
    assert artifacts[0].source_ref == "image:docs/screenshot.png"


def test_fetch_degrades_gracefully_on_empty_llm_response(tmp_path: Path) -> None:
    _write_image(tmp_path, "docs/screenshot.png", size=(10, 20))
    connector = _connector(tmp_path)
    client = _StubChatClient("   ")  # whitespace-only

    with patch("trelix.indexing.connectors.image.build_chat_client", return_value=client):
        artifacts = connector.fetch()

    assert "captioning unavailable" in artifacts[0].body
    assert "10x20px" in artifacts[0].body


def test_fetch_handles_multiple_images_independently(tmp_path: Path) -> None:
    """One image's captioning failure must not affect another's success."""
    _write_image(tmp_path, "a.png")
    _write_image(tmp_path, "b.png")
    connector = _connector(tmp_path)

    call_count = 0

    class _FlakyChatClient(TrelixChatClient):
        def complete(self, *args: object, **kwargs: object) -> ChatResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated outage on the first image only")
            return ChatResponse(content="caption for b", model="stub", finish_reason="stop")

        def stream(self, *args: object, **kwargs: object) -> Any:
            raise AssertionError("not used")

        def tool_call(self, *args: object, **kwargs: object) -> ToolCallResponse:
            raise AssertionError("not used")

    with patch(
        "trelix.indexing.connectors.image.build_chat_client",
        return_value=_FlakyChatClient(),
    ):
        artifacts = connector.fetch()

    assert len(artifacts) == 2
    bodies = {a.source_ref: a.body for a in artifacts}
    assert "captioning unavailable" in bodies["image:a.png"]
    assert bodies["image:b.png"] == "caption for b"


def test_unreadable_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A file that disappears (or is permission-denied) between discovery
    and read must not abort the whole sync -- mirrors the connector base
    class's own "one bad item doesn't sink the batch" posture."""
    good = _write_image(tmp_path, "good.png")
    _write_image(tmp_path, "bad.png")
    connector = _connector(tmp_path)
    client = _StubChatClient("caption")

    original_read_bytes = Path.read_bytes

    def _flaky_read_bytes(self: Path, *args: object, **kwargs: object) -> bytes:
        if self.name == "bad.png":
            raise OSError("simulated permission denied")
        return original_read_bytes(self, *args, **kwargs)  # type: ignore[arg-type]

    with (
        patch("trelix.indexing.connectors.image.build_chat_client", return_value=client),
        patch.object(Path, "read_bytes", _flaky_read_bytes),
    ):
        artifacts = connector.fetch()

    assert {a.source_ref for a in artifacts} == {f"image:{good.name}"}


# ---------------------------------------------------------------------------
# max_images_per_sync cap
# ---------------------------------------------------------------------------


def test_fetch_respects_max_images_per_sync_cap(tmp_path: Path) -> None:
    _write_image(tmp_path, "a.png")
    _write_image(tmp_path, "b.png")
    _write_image(tmp_path, "c.png")
    connector = _connector(tmp_path, max_images_per_sync=2)
    client = _StubChatClient("caption")

    with patch("trelix.indexing.connectors.image.build_chat_client", return_value=client):
        artifacts = connector.fetch()

    # Sorted discovery order means the first two (a, b) win the cap.
    assert {a.title for a in artifacts} == {"a", "b"}


# ---------------------------------------------------------------------------
# _vision_llm_config
# ---------------------------------------------------------------------------


def test_vision_llm_config_overrides_provider_and_model_but_preserves_other_llm_settings(
    tmp_path: Path,
) -> None:
    config = IndexConfig(
        repo_path=str(tmp_path),
        llm=LLMConfig(provider="openai", model="gpt-4o", max_tokens=999),
        image=ImageConnectorConfig(vision_model="claude-opus-4"),
    )
    connector = ImageConnector(config)

    vision_config = connector._vision_llm_config()

    assert vision_config.provider == "anthropic"
    assert vision_config.model == "claude-opus-4"
    assert vision_config.max_tokens == 999  # preserved from the base llm config


def test_vision_llm_config_falls_back_to_the_text_model_when_provider_already_matches(
    tmp_path: Path,
) -> None:
    """When config.llm already targets the vision provider, its own model
    is a real model name for that provider, so it's safe to reuse when no
    explicit vision_model override is set."""
    config = IndexConfig(
        repo_path=str(tmp_path),
        llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-6", _env_file=None),  # type: ignore[call-arg]
    )
    connector = ImageConnector(config)

    vision_config = connector._vision_llm_config()

    assert vision_config.provider == "anthropic"
    assert vision_config.model == "claude-sonnet-4-6"


def test_validate_config_raises_when_provider_mismatched_and_no_vision_model_override(
    tmp_path: Path,
) -> None:
    """Regression: silently reusing config.llm.model across a provider
    mismatch (e.g. an OpenAI model name sent to Anthropic) produced a
    request that would always fail, silently, since _caption() catches
    every exception. Now fails fast at validate_config() with a clear,
    actionable error instead."""
    config = IndexConfig(
        repo_path=str(tmp_path),
        llm=LLMConfig(provider="openai", model="gpt-4o"),
    )
    connector = ImageConnector(config)

    with pytest.raises(ValueError, match="TRELIX_IMAGE_VISION_MODEL"):
        connector.validate_config()


def test_validate_config_passes_when_provider_mismatched_but_vision_model_is_set(
    tmp_path: Path,
) -> None:
    """An explicit vision_model override is enough to proceed even when the
    main pipeline's LLM provider is something else entirely."""
    config = IndexConfig(
        repo_path=str(tmp_path),
        llm=LLMConfig(provider="openai", model="gpt-4o"),
        image=ImageConnectorConfig(vision_model="claude-opus-4"),
    )
    connector = ImageConnector(config)

    connector.validate_config()  # must not raise


# ---------------------------------------------------------------------------
# _prepare_image_bytes — safety caps
# ---------------------------------------------------------------------------


def test_prepare_image_bytes_returns_original_when_under_caps(tmp_path: Path) -> None:
    path = _write_image(tmp_path, "small.png", size=(8, 4))
    connector = _connector(tmp_path)
    raw_bytes = path.read_bytes()

    media_type, out_bytes = connector._prepare_image_bytes(path, raw_bytes)

    assert out_bytes == raw_bytes
    assert media_type == "image/png"


def test_prepare_image_bytes_downscales_when_over_dimension_cap(tmp_path: Path) -> None:
    path = _write_image(tmp_path, "big.png", size=(400, 100))
    connector = _connector(tmp_path, max_image_dimension_px=100)
    raw_bytes = path.read_bytes()

    media_type, out_bytes = connector._prepare_image_bytes(path, raw_bytes)

    assert media_type == "image/png"
    with Image.open(io.BytesIO(out_bytes)) as img:
        assert max(img.size) <= 100


def test_prepare_image_bytes_falls_back_to_jpeg_when_still_over_byte_cap(tmp_path: Path) -> None:
    """A PNG has no quality knob -- once resized it must re-encode as JPEG
    to have any chance of fitting a tiny byte cap."""
    path = _write_image(tmp_path, "big.png", size=(400, 100))
    connector = _connector(tmp_path, max_image_dimension_px=100, max_image_bytes=50)
    raw_bytes = path.read_bytes()

    media_type, out_bytes = connector._prepare_image_bytes(path, raw_bytes)

    assert media_type == "image/jpeg"
    assert len(out_bytes) > 0


def test_prepare_image_bytes_falls_back_to_original_bytes_on_corrupt_image(
    tmp_path: Path,
) -> None:
    """A corrupt/undecodable file that is already under the byte cap has a
    safe, bounded payload to send -- returns the original bytes unchanged
    rather than blocking captioning outright."""
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not a real png")
    connector = _connector(tmp_path)

    result = connector._prepare_image_bytes(path, b"not a real png")

    assert result is not None
    media_type, out_bytes = result
    assert media_type == "image/png"
    assert out_bytes == b"not a real png"


def test_prepare_image_bytes_returns_none_when_corrupt_and_over_byte_cap(
    tmp_path: Path,
) -> None:
    """Regression: an undecodable file that ALSO exceeds max_image_bytes
    has no safe, bounded payload -- must signal 'skip the vision call
    entirely' (None) rather than transmitting arbitrarily large, unverified
    bytes to the vision API."""
    path = tmp_path / "corrupt.png"
    garbage = b"not a real png" * 10
    path.write_bytes(garbage)
    connector = _connector(tmp_path, max_image_bytes=10)

    result = connector._prepare_image_bytes(path, garbage)

    assert result is None


# ---------------------------------------------------------------------------
# _mechanical_description
# ---------------------------------------------------------------------------


def test_mechanical_description_reports_filename_dimensions_and_size(tmp_path: Path) -> None:
    path = _write_image(tmp_path, "shot.png", size=(10, 20))
    connector = _connector(tmp_path)
    raw_bytes = path.read_bytes()

    description = connector._mechanical_description(path, raw_bytes)

    assert "shot.png" in description
    assert "10x20px" in description
    assert "captioning unavailable" in description


def test_mechanical_description_handles_undecodable_bytes(tmp_path: Path) -> None:
    connector = _connector(tmp_path)

    description = connector._mechanical_description(Path("broken.png"), b"garbage")

    assert "broken.png" in description
    assert "unknown dimensions" in description
