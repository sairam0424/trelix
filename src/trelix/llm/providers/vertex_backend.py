"""Google Vertex AI / Gemini backend for TrelixChatClient."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.vertex_backend")


class VertexBackend(TrelixChatClient):
    """
    TrelixChatClient backed by Google Vertex AI / Gemini via google-genai SDK.

    Key differences from OpenAI:
    - max_output_tokens in generation_config
    - system_instruction= separate param
    - google.genai.types.Tool for function definitions
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._model = config.model
        self._client = self._build_client(config)

    def _build_client(self, config: LLMConfig) -> Any:
        try:
            from google import genai
        except (ImportError, TypeError) as exc:
            raise ImportError(
                "Vertex backend requires google-genai. "
                "Install it with: pip install 'trelix[vertex]'"
            ) from exc
        if config.google_api_key:
            return genai.Client(api_key=config.google_api_key)
        if config.google_project_id:
            return genai.Client(
                vertexai=True,
                project=config.google_project_id,
                location=config.google_location,
            )
        logger.debug("VertexBackend: no Google credentials set.")
        return None

    def _build_contents(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        return [
            {"role": "user" if m.role == "user" else "model", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> ChatResponse:
        if self._client is None:
            return ChatResponse(
                content="[trelix] Vertex AI not configured.",
                model="none",
                finish_reason="stop",
            )
        from google.genai import types

        effective_system = system or next((m.content for m in messages if m.role == "system"), None)
        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            system_instruction=effective_system,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=self._build_contents(messages),
            config=gen_config,
        )
        finish = (
            response.candidates[0].finish_reason.name.lower() if response.candidates else "stop"
        )
        normalized = (
            "stop"
            if finish in ("stop", "1")
            else "length"
            if finish in ("max_tokens", "2")
            else "stop"
        )
        return ChatResponse(
            content=response.text or "",
            model=self._model,
            finish_reason=normalized,
            input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
            output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
        )

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> Iterator[str]:
        if self._client is None:
            yield "[trelix] Vertex AI not configured."
            return
        from google.genai import types

        effective_system = system or next((m.content for m in messages if m.role == "system"), None)
        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens or self._config.max_tokens,
            temperature=temperature if temperature is not None else self._config.temperature,
            system_instruction=effective_system,
        )
        for chunk in self._client.models.generate_content_stream(
            model=self._model,
            contents=self._build_contents(messages),
            config=gen_config,
        ):
            if chunk.text:
                yield chunk.text

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
        max_tokens: int | None = None,
    ) -> ToolCallResponse:
        if self._client is None:
            raise RuntimeError("Vertex AI not configured.")
        from google.genai import types

        vertex_tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t["function"]["name"],
                        description=t["function"].get("description", ""),
                        parameters=t["function"].get("parameters", {}),
                    )
                ]
            )
            for t in tools
        ]
        gen_config = types.GenerateContentConfig(
            tools=vertex_tools,
            max_output_tokens=max_tokens or self._config.max_tokens,
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=self._build_contents(messages),
            config=gen_config,
        )
        for part in response.candidates[0].content.parts if response.candidates else []:
            if part.function_call:
                return ToolCallResponse(
                    tool_name=part.function_call.name,
                    tool_arguments=dict(part.function_call.args),
                    raw_response=response,
                )
        raise RuntimeError("Vertex AI did not return a function call.")
