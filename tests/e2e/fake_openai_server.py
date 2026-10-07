"""A stdlib OpenAI-compatible fake server for the e2e tests; tests only, never production.

It speaks the one route `trelix review` and `OpenAIBackend` use, `POST /v1/chat/completions`,
and answers each request with the next scripted `Reply`: a `chat.completion` object rendered as
JSON, or as server-sent events (`data: <chunk>` lines ending in `data: [DONE]`) when the request
body carries `"stream": true`. A completion carries the `usage.prompt_tokens` the PR2
prompt-truncation guard reads, or no `usage` at all when the script says so.

`http.server.ThreadingHTTPServer` plus `threading`, bound to `127.0.0.1:0` so the OS picks a
free port; every request is recorded (path, lower-cased headers, parsed JSON body) for the test
to assert on at the process boundary that `CliRunner` and `httpx.MockTransport` cannot see.

Loud-failure rules: HTTP/1.1 with an explicit `Content-Length` on every reply; a path other than
the chat route answers 404; a request past the end of the script answers 400 (the SDK raises
`BadRequestError`, which `core/retry.py` does not retry), so a seventh request is a visible
failure, never a hang.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_CHAT_PATH = "/v1/chat/completions"
_JSON = "application/json"
_SSE = "text/event-stream"


@dataclass(frozen=True)
class Reply:
    """One scripted answer: a raw body sent verbatim, or a chat completion rendered per request."""

    status: int = 200
    content_type: str = _JSON
    body: bytes = b""
    chat: Mapping[str, Any] | None = None


def completion(
    content: str,
    *,
    prompt_tokens: int | None = 10_000,
    finish_reason: str = "stop",
    model: str = "qwen2.5-coder:7b",
) -> Reply:
    """A `chat.completion` reply; `prompt_tokens=None` omits the `usage` object entirely."""
    chat: dict[str, Any] = {
        "id": "c",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if prompt_tokens is not None:
        chat["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 1,
            "total_tokens": prompt_tokens + 1,
        }
    return Reply(chat=chat)


def garbage_json() -> Reply:
    """A 200 `application/json` reply whose body is not JSON (the SDK raises `JSONDecodeError`)."""
    return Reply(status=200, content_type=_JSON, body=b"{not json")


@dataclass(frozen=True)
class RecordedRequest:
    """One request as the server saw it; `body` is the parsed JSON, `None` when it was not JSON."""

    path: str
    headers: Mapping[str, str]
    body: Any


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _error_body(message: str) -> bytes:
    return _json_bytes(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        }
    )


def _sse_bytes(chat: Mapping[str, Any]) -> bytes:
    """Render a completion as the two-chunk stream the SDK's streaming parser expects."""
    choice = chat["choices"][0]
    head = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": chat["model"]}
    first = {
        **head,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": choice["message"]["content"]},
                "finish_reason": None,
            }
        ],
    }
    last = {
        **head,
        "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
    }
    events = b"".join(b"data: " + _json_bytes(chunk) + b"\n\n" for chunk in (first, last))
    return events + b"data: [DONE]\n\n"


def _parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


class _Handler(BaseHTTPRequestHandler):
    """Reaches the script and the record through `self.server.fake` (set by `start()`)."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Silence the default access log on stderr so the pytest output stays clean."""

    def do_GET(self) -> None:
        self._send(404, _JSON, _error_body(f"fake server: no route for {self.path}"))

    def do_POST(self) -> None:
        fake: FakeOpenAIServer = self.server.fake  # type: ignore[attr-defined]
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = _parse_json(raw)
        headers = {name.lower(): value for name, value in self.headers.items()}
        count = fake._record(RecordedRequest(self.path, headers, body))
        if self.path != _CHAT_PATH:
            self._send(404, _JSON, _error_body(f"fake server: no route for {self.path}"))
            return
        reply = fake._next_reply()
        if reply is None:
            message = f"fake server: no reply scripted for request {count}"
            self._send(400, _JSON, _error_body(message))
            return
        if reply.chat is None:
            self._send(reply.status, reply.content_type, reply.body)
            return
        if isinstance(body, dict) and body.get("stream") is True:
            self._send(200, _SSE, _sse_bytes(reply.chat))
            return
        self._send(200, _JSON, _json_bytes(reply.chat))

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeOpenAIServer:
    """The scripted server; `with FakeOpenAIServer(replies) as server:` starts and stops it.

    `ThreadingHTTPServer` is used as shipped, with its `daemon_threads = True`: that is what keeps
    `stop()`'s `server_close()` from joining a handler thread left parked in `rfile.readline()` on
    a kept-alive HTTP/1.1 connection the SDK client never closes.
    """

    def __init__(self, replies: Sequence[Reply]) -> None:
        self._replies: deque[Reply] = deque(replies)
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.requests: list[RecordedRequest] = []

    def start(self) -> FakeOpenAIServer:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.fake = self  # type: ignore[attr-defined] - read by _Handler.do_POST
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        self._server = server
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None

    def __enter__(self) -> FakeOpenAIServer:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("FakeOpenAIServer is not started")
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def bodies(self) -> list[Any]:
        return [request.body for request in self.requests]

    def _record(self, request: RecordedRequest) -> int:
        """Append under the lock; returns the request's 1-based ordinal."""
        with self._lock:
            self.requests.append(request)
            return len(self.requests)

    def _next_reply(self) -> Reply | None:
        with self._lock:
            return self._replies.popleft() if self._replies else None
