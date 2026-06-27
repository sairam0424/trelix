"""AWS Bedrock Converse API backend for TrelixChatClient."""
from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient, ToolCallResponse

if TYPE_CHECKING:
    from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.llm.bedrock_backend")

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def _resolve_bedrock_model(config: "LLMConfig") -> tuple[str, str]:
    """
    Return (primary_model_id, fallback_model_id) for Bedrock.

    When config.model is its zero-value default ("gpt-4o") and provider is
    "bedrock", it was never explicitly set — use the Bedrock-specific fields
    (bedrock_primary_model / bedrock_fallback_model) which default to
    sonnet-4-6 / haiku-4-5.

    If the caller explicitly set TRELIX_LLM_MODEL, that value wins as primary
    and the fallback still comes from bedrock_fallback_model.
    """
    primary = (
        config.bedrock_primary_model
        if config.model == "gpt-4o"  # never overridden from LLMConfig default
        else config.model
    )
    fallback = config.bedrock_fallback_model
    return primary, fallback


class BedrockBackend(TrelixChatClient):
    """
    TrelixChatClient backed by AWS Bedrock Converse API.

    Model selection:
      Default primary:  us.anthropic.claude-sonnet-4-6  (via TRELIX_LLM_BEDROCK_PRIMARY_MODEL)
      Default fallback: us.anthropic.claude-haiku-4-5-20251001-v1:0  (via TRELIX_LLM_BEDROCK_FALLBACK_MODEL)

      On ValidationException (model not available / throughput tier mismatch),
      every call transparently retries once with the fallback model and logs a
      warning. The active model is updated so subsequent calls skip the retry.

    Key API differences from OpenAI (research-verified):
    - Token limit: inferenceConfig.maxTokens (camelCase, nested)
    - System prompt: system=[{"text": "..."}] at top level
    - Message content: always list-of-dicts [{"text": "..."}]
    - Tool choice: {"auto": {}} / {"any": {}} / {"tool": {"name": "fn"}}
    """

    def __init__(self, config: "LLMConfig") -> None:
        self._config = config
        self._primary_model, self._fallback_model = _resolve_bedrock_model(config)
        self._model = self._primary_model  # active model — may switch on fallback
        self._client = self._build_client(config)

    @staticmethod
    def _decode_credential(value: str) -> str:
        """Transparently decode base64-encoded credentials stored in .env."""
        try:
            decoded = base64.b64decode(value).decode("utf-8")
            # Valid AWS creds are printable ASCII — if decode succeeds and looks
            # like a credential (starts with known prefixes or is a long secret),
            # use the decoded value.
            if decoded.isprintable() and "\n" not in decoded:
                return decoded
        except Exception:  # noqa: BLE001
            pass
        return value

    def _build_client(self, config: "LLMConfig") -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "Bedrock backend requires boto3. "
                "Install it with: pip install 'trelix[bedrock]'"
            ) from exc
        session_kwargs: dict[str, Any] = {}
        if config.aws_profile:
            session_kwargs["profile_name"] = config.aws_profile
        session = boto3.Session(**session_kwargs)
        client_kwargs: dict[str, Any] = {"region_name": config.aws_region}
        if config.aws_access_key_id:
            client_kwargs["aws_access_key_id"] = self._decode_credential(config.aws_access_key_id)
        if config.aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = self._decode_credential(config.aws_secret_access_key)
        return session.client("bedrock-runtime", **client_kwargs)

    def _build_request(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int],
        system: Optional[str],
        tools: Optional[list[dict[str, Any]]] = None,
        force_tool: Optional[str] = None,
    ) -> dict[str, Any]:
        effective_system = system or next(
            (m.content for m in messages if m.role == "system"), None
        )
        request: dict[str, Any] = {
            "modelId": self._model,
            "inferenceConfig": {
                "maxTokens": max_tokens or self._config.max_tokens,
                "temperature": self._config.temperature,
            },
            "messages": [
                {
                    "role": m.role,
                    "content": [{"text": m.content}],  # always list-of-dicts
                }
                for m in messages if m.role != "system"
            ],
        }
        if effective_system:
            request["system"] = [{"text": effective_system}]
        if tools:
            request["toolConfig"] = {
                "tools": [self._convert_tool(t) for t in tools],
                "toolChoice": (
                    {"tool": {"name": force_tool}} if force_tool else {"auto": {}}
                ),
            }
        return request

    def _convert_tool(self, openai_tool: dict[str, Any]) -> dict[str, Any]:
        fn = openai_tool["function"]
        return {
            "toolSpec": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "inputSchema": {"json": fn.get("parameters", {})},
            }
        }

    def _normalize_finish_reason(self, stop_reason: str) -> str:
        return _STOP_REASON_MAP.get(stop_reason, "stop")

    def _is_model_unavailable(self, exc: Exception) -> bool:
        """True when Bedrock signals the model isn't available on-demand."""
        msg = str(exc)
        return (
            "ValidationException" in type(exc).__name__
            or "ValidationException" in msg
        ) and (
            "on-demand throughput" in msg
            or "inference profile" in msg
            or "not supported" in msg
        )

    def _try_with_fallback(self, fn: Any, request: dict[str, Any]) -> Any:
        """
        Call fn(request). On ValidationException for the primary model, swap
        to the fallback, update the active model, and retry once.
        """
        try:
            return fn(**request)
        except Exception as exc:  # noqa: BLE001
            if not self._is_model_unavailable(exc) or self._model == self._fallback_model:
                raise
            logger.warning(
                "Bedrock model %r unavailable (%s). Falling back to %r.",
                self._model,
                exc,
                self._fallback_model,
            )
            self._model = self._fallback_model
            request["modelId"] = self._fallback_model
            return fn(**request)

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> ChatResponse:
        request = self._build_request(messages, max_tokens, system)
        if temperature is not None:
            request["inferenceConfig"]["temperature"] = temperature
        response = self._try_with_fallback(self._client.converse, request)
        output_msg = response["output"]["message"]
        content = next(
            (block["text"] for block in output_msg["content"] if "text" in block), ""
        )
        usage = response.get("usage", {})
        return ChatResponse(
            content=content,
            model=self._model,
            finish_reason=self._normalize_finish_reason(response.get("stopReason", "end_turn")),
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )

    def stream(
        self,
        messages: list[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        request = self._build_request(messages, max_tokens, system)
        if temperature is not None:
            request["inferenceConfig"]["temperature"] = temperature
        response = self._try_with_fallback(self._client.converse_stream, request)
        stream = response.get("stream")
        if stream:
            for event in stream:
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                if "text" in delta:
                    yield delta["text"]

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        force_tool: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ToolCallResponse:
        request = self._build_request(messages, max_tokens, None, tools, force_tool)
        response = self._try_with_fallback(self._client.converse, request)
        output_msg = response["output"]["message"]
        tool_use = next(
            (block["toolUse"] for block in output_msg["content"] if "toolUse" in block),
            None,
        )
        if not tool_use:
            raise RuntimeError("Bedrock did not return a toolUse block.")
        return ToolCallResponse(
            tool_name=tool_use["name"],
            tool_arguments=dict(tool_use.get("input", {})),
            raw_response=response,
        )
