"""
ImageConnector — indexes local raster images (`.png`/`.jpg`/`.jpeg`) as
Artifacts, captioned via the Anthropic vision backend added in Phase 1
(trelix.llm.client.ChatMessage.images / ImageContent).

Phase 2 of docs/ROADMAP.md's "Multi-modal" entry: DiagramConnector's module
docstring explicitly scoped raster images OUT of that change — "a separate,
larger change with its own new provider code to review" — because unlike
`.drawio` XML, a raster image has no text form an existing text-only LLM call
can describe. This connector is that separate change.

Mirrors DiagramConnector's overall shape (ArtifactSource, one Artifact per
file, graceful degradation to a mechanical fallback on captioning failure)
with two deliberate differences:

1. Discovery does NOT hand-maintain its own skip-dirs list the way
   DiagramConnector's `_SKIP_DIRS` does. It reuses the same nested-
   `.gitignore` matching FileWalker itself uses, via `indexing.gitignore`
   (extracted from FileWalker for exactly this reuse) — with its own,
   separate cache, since this connector's discovery walk is independent of
   any FileWalker instance.
2. A captioning failure/empty response falls back to a MECHANICAL
   description built from the file's name, pixel dimensions, and size on
   disk — not from any extracted content, since (unlike `.drawio`'s XML
   labels) a raster image has no text to fall back to.

Same connector-not-Chunk boundary as DiagramConnector: an image becomes an
Artifact linked into generic_edges via ArtifactLinker, never a Chunk, so
chunks.symbol_id's hard NOT NULL FK stays untouched. See Artifact's
docstring in core/models.py.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import pathspec

from trelix.core.config import IndexConfig, LLMConfig
from trelix.core.models import Artifact
from trelix.indexing.connectors.base import ArtifactSource
from trelix.indexing.gitignore import is_path_gitignored
from trelix.llm.client import ChatMessage, ImageContent
from trelix.llm.factory import build_chat_client

logger = logging.getLogger("trelix.indexing.connectors.image")

_CAPTION_PROMPT = """\
Describe this image for a code search index. Focus on what it depicts \
(a UI screenshot, an architecture diagram, a photo, a chart, etc.), any \
visible text or labels, and any structure or relationships it shows. \
Output ONLY the description, 2-4 sentences, no preamble, no markdown.
"""

_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class ImageConnector(ArtifactSource):
    """See module docstring. Constructed with the same IndexConfig every
    other pipeline stage uses -- repo_path for discovery, .llm/.image for
    captioning, no separate call signature from DiagramConnector's."""

    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        self._repo_path = Path(config.repo_path).resolve()

    def validate_config(self) -> None:
        if not self._repo_path.is_dir():
            raise ValueError(f"ImageConnector: repo_path is not a directory: {self._repo_path}")
        image_cfg = self._config.image
        if (
            self._config.llm.provider != image_cfg.vision_provider
            and image_cfg.vision_model is None
        ):
            raise ValueError(
                "ImageConnector: config.llm.provider is "
                f"{self._config.llm.provider!r}, not {image_cfg.vision_provider!r} "
                "(the only vision-capable backend today), and no "
                "TRELIX_IMAGE_VISION_MODEL override is set -- there is no known-valid "
                "model name to send. Set TRELIX_IMAGE_VISION_MODEL to a real "
                f"{image_cfg.vision_provider} model name, or configure "
                f"TRELIX_LLM_PROVIDER={image_cfg.vision_provider} with a matching model."
            )

    def fetch(self) -> list[Artifact]:
        self.validate_config()
        chat_client = build_chat_client(self._vision_llm_config())

        image_cfg = self._config.image
        discovered = self._discover()
        capped = discovered[: image_cfg.max_images_per_sync]
        if len(capped) < len(discovered):
            logger.warning(
                "ImageConnector: found %d image(s) under %s, capping to %d (max_images_per_sync)",
                len(discovered),
                self._repo_path,
                len(capped),
            )

        artifacts: list[Artifact] = []
        for path in capped:
            try:
                raw_bytes = path.read_bytes()
            except OSError as exc:
                logger.warning("ImageConnector: could not read %s (%s); skipping", path, exc)
                continue
            artifacts.append(self._file_to_artifact(path, raw_bytes, chat_client))
        return artifacts

    def _vision_llm_config(self) -> LLMConfig:
        """Build the LLMConfig used for captioning: a copy of `config.llm`
        (api keys, timeouts, etc. all preserved) with provider/model
        overridden to `config.image`'s vision choice -- captioning may use a
        different provider/model than the pipeline's main text LLM, since
        only Anthropic supports vision today.

        `validate_config()` has already confirmed a known-valid model
        exists: either `vision_model` was explicitly set, or `config.llm`
        already targets `vision_provider` (so its own `.model` is a real
        model name for that provider, not just reused blindly across a
        provider mismatch -- see the bug this guards against in
        validate_config()'s docstring/comment above)."""
        image_cfg = self._config.image
        model = image_cfg.vision_model
        if model is None:
            if self._config.llm.provider != image_cfg.vision_provider:
                raise ValueError(
                    "ImageConnector._vision_llm_config called without validate_config() "
                    "having run first -- provider/model mismatch would produce an "
                    "invalid request."
                )
            model = self._config.llm.model
        return self._config.llm.model_copy(
            update={"provider": image_cfg.vision_provider, "model": model}
        )

    def _discover(self) -> list[Path]:
        """Walk `repo_path`, pruning `.gitignore`d directories as we go
        (rather than post-filtering a full `rglob`) and matching filenames
        against `config.image.extensions` case-insensitively. Uses its own,
        call-scoped gitignore-spec cache -- deliberately not shared with any
        FileWalker instance, mirroring how DiagramConnector's `_discover()`
        does its own independent traversal with zero shared state."""
        image_cfg = self._config.image
        respect_gitignore = self._config.walker.respect_gitignore
        extensions = {ext.lower() for ext in image_cfg.extensions}
        cache: dict[Path, pathspec.PathSpec | None] = {}  # type: ignore[type-arg]

        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self._repo_path):
            current_dir = Path(dirpath)
            # Prune ignored subdirectories in place so os.walk never descends
            # into them (e.g. a gitignored node_modules/ with thousands of
            # files never gets listed at all, not merely filtered after).
            dirnames[:] = [
                d
                for d in dirnames
                if not is_path_gitignored(
                    self._repo_path,
                    current_dir / d,
                    is_dir=True,
                    respect_gitignore=respect_gitignore,
                    cache=cache,
                )
            ]
            for filename in filenames:
                if Path(filename).suffix.lower() not in extensions:
                    continue
                path = current_dir / filename
                if is_path_gitignored(
                    self._repo_path,
                    path,
                    is_dir=False,
                    respect_gitignore=respect_gitignore,
                    cache=cache,
                ):
                    continue
                found.append(path)
        return sorted(found)

    def _file_to_artifact(self, path: Path, raw_bytes: bytes, chat_client: object) -> Artifact:
        rel_path = path.relative_to(self._repo_path).as_posix()
        caption = self._caption(path, raw_bytes, chat_client)
        return Artifact(
            source_ref=f"image:{rel_path}",
            artifact_kind="image",
            title=path.stem,
            body=caption,
            url=None,
            metadata={"path": rel_path},
        )

    def _caption(self, path: Path, raw_bytes: bytes, chat_client: object) -> str:
        """One vision LLM call per image. Same graceful-degradation contract
        as DiagramConnector's `_caption`: a captioning failure never drops
        the image entirely -- it falls back to a mechanical description so
        the artifact is still indexed and linkable, just without a
        synthesized summary (never worse-without-a-trace)."""
        prepared = self._prepare_image_bytes(path, raw_bytes)
        if prepared is None:
            return self._mechanical_description(path, raw_bytes)
        media_type, capped_bytes = prepared
        try:
            response = chat_client.complete(  # type: ignore[attr-defined]
                [
                    ChatMessage(
                        role="user",
                        content=_CAPTION_PROMPT,
                        images=[ImageContent(data=capped_bytes, media_type=media_type)],
                    )
                ],
                max_tokens=256,
                temperature=0.0,
            )
            caption = (response.content or "").strip()
            if caption:
                return caption
        except Exception as exc:  # noqa: BLE001 — graceful degradation (compressor contract)
            logger.warning(
                "ImageConnector: captioning failed for %s (%s); using fallback", path, exc
            )
        return self._mechanical_description(path, raw_bytes)

    def _prepare_image_bytes(self, path: Path, raw_bytes: bytes) -> tuple[str, bytes] | None:
        """Cap dimensions/byte-size before sending to the vision API --
        downscale via Pillow rather than skip, so an oversized image still
        gets captioned and indexed, just from a smaller copy.

        Returns None when Pillow cannot even open the image (corrupt file,
        unsupported variant) AND the raw bytes exceed `max_image_bytes` --
        there is no safe, bounded payload to send in that case, so the
        caller must skip the vision call entirely rather than transmit
        arbitrarily large, unverified bytes. When Pillow can't open the
        image but it's already under the byte cap, the original bytes are
        returned unchanged (bounded size; if they're not actually a valid
        image, `_caption`'s own exception handling around the API call is
        the real safety net for that)."""
        image_cfg = self._config.image
        media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
        try:
            from PIL import Image

            with Image.open(io.BytesIO(raw_bytes)) as img:
                # Trust the decoded format over the file extension -- a
                # mismatched extension (e.g. a PNG saved as .jpg) would
                # otherwise send a media_type header that doesn't match the
                # actual bytes.
                sniffed = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(img.format or "")
                if sniffed:
                    media_type = sniffed

                width, height = img.size
                over_dimension = max(width, height) > image_cfg.max_image_dimension_px
                over_bytes = len(raw_bytes) > image_cfg.max_image_bytes
                if not over_dimension and not over_bytes:
                    return media_type, raw_bytes

                resized: Image.Image = img
                if over_dimension:
                    scale = image_cfg.max_image_dimension_px / max(width, height)
                    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
                    resized = img.resize(new_size, Image.Resampling.LANCZOS)

                fmt = "JPEG" if media_type == "image/jpeg" else "PNG"
                buffer = io.BytesIO()
                to_save = resized.convert("RGB") if fmt == "JPEG" else resized
                to_save.save(buffer, format=fmt)
                out_bytes = buffer.getvalue()

                # A dimension-only resize does not guarantee the byte cap is
                # also satisfied (a highly detailed image can still be large
                # at a smaller size); PNG has no quality knob, so fall back
                # to a JPEG re-encode, which compresses far better for
                # photographic content.
                if len(out_bytes) > image_cfg.max_image_bytes and fmt == "PNG":
                    buffer = io.BytesIO()
                    resized.convert("RGB").save(buffer, format="JPEG", quality=85)
                    out_bytes = buffer.getvalue()
                    media_type = "image/jpeg"

                return media_type, out_bytes
        except Exception as exc:  # noqa: BLE001 — never block captioning on a resize failure
            if len(raw_bytes) > image_cfg.max_image_bytes:
                logger.warning(
                    "ImageConnector: %s could not be decoded (%s) and exceeds "
                    "max_image_bytes (%d > %d); skipping vision call",
                    path,
                    exc,
                    len(raw_bytes),
                    image_cfg.max_image_bytes,
                )
                return None
            logger.warning(
                "ImageConnector: could not inspect/resize %s (%s); sending original bytes",
                path,
                exc,
            )
            return media_type, raw_bytes

    def _mechanical_description(self, path: Path, raw_bytes: bytes) -> str:
        """The fallback for a captioning failure or empty response: filename
        + pixel dimensions + file size. Unlike DiagramConnector's XML-label
        fallback, a raster image has no text content to extract -- this is
        honestly labeled as unavailable rather than fabricating a caption
        from the filename alone."""
        try:
            from PIL import Image

            with Image.open(io.BytesIO(raw_bytes)) as img:
                dimensions = f"{img.width}x{img.height}px"
        except Exception:  # noqa: BLE001 — dimensions are best-effort in a fallback path
            dimensions = "unknown dimensions"
        size_kb = len(raw_bytes) / 1024
        return f"[trelix: captioning unavailable] {path.name}, {dimensions}, {size_kb:.1f} KB."
