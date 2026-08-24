"""
Unit tests for the unified retry contract (src/trelix/core/retry.py) —
real exception objects (httpx/requests/openai; anthropic/litellm are
optional extras and skip via importorskip when absent), no mocking of the
classification logic itself, plus a real tenacity retry-loop integration
test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import openai
import pytest
import requests

from trelix.core.retry import (
    RETRYABLE_STATUS_CODES,
    _extract_retry_after_seconds,
    _wait_retry_after_or_exponential,
    is_retryable_http_error,
    with_retry,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ERROR_CODE_TO_OPENAI_EXC: dict[int, type[openai.APIStatusError]] = {
    429: openai.RateLimitError,
    500: openai.InternalServerError,
    404: openai.NotFoundError,
    401: openai.AuthenticationError,
}


def _httpx_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"{status_code} error", request=request, response=response)


def _httpx_status_error_with_retry_after(
    status_code: int, retry_after: str
) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status_code, headers={"Retry-After": retry_after}, request=request)
    return httpx.HTTPStatusError(f"{status_code} error", request=request, response=response)


def _requests_http_error(status_code: int) -> requests.exceptions.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    return requests.exceptions.HTTPError(response=response)


def _openai_status_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    exc_cls = _ERROR_CODE_TO_OPENAI_EXC.get(status_code, openai.APIStatusError)
    return exc_cls(f"{status_code} error", response=response, body=None)


def _anthropic_status_error(status_code: int) -> Exception:
    anthropic = pytest.importorskip("anthropic")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return anthropic.APIStatusError(f"{status_code} error", response=response, body=None)


def _botocore_client_error(status_code: int) -> Exception:
    """A real botocore.exceptions.ClientError shaped like what boto3's
    bedrock-runtime client raises on a throttled/server-error InvokeModel
    call: no HTTPStatusError, no .response.status_code — the status lives
    at .response["ResponseMetadata"]["HTTPStatusCode"], which is why
    _extract_status_code() has a dedicated branch for it distinct from the
    httpx/requests/openai/anthropic branches."""
    botocore_exceptions = pytest.importorskip("botocore.exceptions")
    return botocore_exceptions.ClientError(
        error_response={
            "Error": {"Code": "ThrottlingException", "Message": "rate limited"},
            "ResponseMetadata": {"HTTPStatusCode": status_code},
        },
        operation_name="InvokeModel",
    )


class TestIsRetryableHttpError:
    @pytest.mark.parametrize("status_code", sorted(RETRYABLE_STATUS_CODES))
    def test_httpx_retryable_status_codes(self, status_code: int) -> None:
        assert is_retryable_http_error(_httpx_status_error(status_code)) is True

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
    def test_httpx_non_retryable_status_codes(self, status_code: int) -> None:
        assert is_retryable_http_error(_httpx_status_error(status_code)) is False

    @pytest.mark.parametrize("status_code", sorted(RETRYABLE_STATUS_CODES))
    def test_requests_retryable_status_codes(self, status_code: int) -> None:
        assert is_retryable_http_error(_requests_http_error(status_code)) is True

    @pytest.mark.parametrize("status_code", [400, 401, 404])
    def test_requests_non_retryable_status_codes(self, status_code: int) -> None:
        assert is_retryable_http_error(_requests_http_error(status_code)) is False

    def test_httpx_connection_error_is_retryable(self) -> None:
        exc = httpx.ConnectError("connection refused")
        assert is_retryable_http_error(exc) is True

    def test_httpx_timeout_is_retryable(self) -> None:
        exc = httpx.TimeoutException("timed out")
        assert is_retryable_http_error(exc) is True

    def test_requests_connection_error_is_retryable(self) -> None:
        exc = requests.exceptions.ConnectionError("connection refused")
        assert is_retryable_http_error(exc) is True

    def test_requests_timeout_is_retryable(self) -> None:
        exc = requests.exceptions.Timeout("timed out")
        assert is_retryable_http_error(exc) is True

    def test_unrelated_exception_is_not_retryable(self) -> None:
        assert is_retryable_http_error(ValueError("not a network error")) is False

    def test_botocore_client_error_missing_is_gracefully_not_retryable(self) -> None:
        """botocore isn't installed in every trelix extra — an exception
        that isn't any recognized HTTP-library type must degrade to
        'not retryable' rather than raising an import error."""
        assert is_retryable_http_error(RuntimeError("some boto3-shaped error")) is False

    def test_malformed_sdk_exception_degrades_to_not_retryable_instead_of_crashing(
        self,
    ) -> None:
        """is_retryable_http_error()'s entire contract is 'never raise' —
        a real openai.APIStatusError instance whose .response attribute
        raises on access (a torn-down mock, a broken shim, an SDK version
        bump that changes the attribute's shape) must degrade to False,
        not propagate AttributeError out of tenacity's retry predicate
        mid-retry-loop."""

        class _BrokenResponse:
            @property
            def status_code(self) -> int:
                raise AttributeError("response was torn down")

        exc = openai.APIStatusError.__new__(openai.APIStatusError)
        exc.response = _BrokenResponse()  # type: ignore[assignment]

        assert is_retryable_http_error(exc) is False

    @pytest.mark.parametrize(
        "exc_class_name", ["EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError"]
    )
    def test_botocore_connection_level_errors_are_retryable(self, exc_class_name: str) -> None:
        """BotoCoreError subclasses (EndpointConnectionError,
        ConnectTimeoutError, ReadTimeoutError — raised by boto3 on DNS
        failure, connect timeout, or read timeout talking to
        bedrock-runtime) are a disjoint hierarchy from ClientError (which
        carries a real HTTP status via ResponseMetadata and is already
        handled by status-code extraction) — without a dedicated
        connection-level check, a network-level Bedrock failure was never
        retried, unlike every other backend/connector wired into this
        same contract."""
        botocore_exceptions = pytest.importorskip("botocore.exceptions")
        exc_class = getattr(botocore_exceptions, exc_class_name)
        exc = exc_class(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com")
        assert is_retryable_http_error(exc) is True

    @pytest.mark.parametrize("status_code", sorted(RETRYABLE_STATUS_CODES))
    def test_botocore_client_error_retryable_status_codes(self, status_code: int) -> None:
        """The bedrock-runtime shape: a real HTTP status carried in
        ClientError.response["ResponseMetadata"]["HTTPStatusCode"], not on
        an httpx/requests/openai-style .response.status_code attribute —
        this is the ClientError branch of _extract_status_code(), disjoint
        from the BotoCoreError connection-level branch covered above."""
        assert is_retryable_http_error(_botocore_client_error(status_code)) is True

    @pytest.mark.parametrize("status_code", [400, 403, 404])
    def test_botocore_client_error_non_retryable_status_codes(self, status_code: int) -> None:
        assert is_retryable_http_error(_botocore_client_error(status_code)) is False

    @pytest.mark.parametrize("status_code", sorted(RETRYABLE_STATUS_CODES))
    def test_openai_retryable_status_codes(self, status_code: int) -> None:
        """openai.APIStatusError wraps an httpx.Response but is NOT itself
        an httpx.HTTPStatusError — every LLM backend's complete()/stream()/
        tool_call() raises this shape, not a raw httpx exception."""
        assert is_retryable_http_error(_openai_status_error(status_code)) is True

    @pytest.mark.parametrize("status_code", [400, 401, 404])
    def test_openai_non_retryable_status_codes(self, status_code: int) -> None:
        assert is_retryable_http_error(_openai_status_error(status_code)) is False

    def test_openai_connection_error_is_retryable(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        exc = openai.APIConnectionError(request=request)
        assert is_retryable_http_error(exc) is True

    @pytest.mark.parametrize("status_code", sorted(RETRYABLE_STATUS_CODES))
    def test_anthropic_retryable_status_codes(self, status_code: int) -> None:
        assert is_retryable_http_error(_anthropic_status_error(status_code)) is True

    @pytest.mark.parametrize("status_code", [400, 401, 404])
    def test_anthropic_non_retryable_status_codes(self, status_code: int) -> None:
        assert is_retryable_http_error(_anthropic_status_error(status_code)) is False

    def test_anthropic_connection_error_is_retryable(self) -> None:
        anthropic = pytest.importorskip("anthropic")
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        exc = anthropic.APIConnectionError(message="connection failed", request=request)
        assert is_retryable_http_error(exc) is True

    def test_litellm_ratelimit_error_is_retryable_via_openai_subclass(self) -> None:
        """litellm's exceptions subclass openai's (RateLimitError ->
        openai.RateLimitError -> openai.APIStatusError) — no litellm-specific
        branch is needed in the classifier, this proves it."""
        litellm = pytest.importorskip("litellm")
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(429, request=request)
        exc = litellm.RateLimitError(
            "rate limited", llm_provider="openai", model="gpt-4o", response=response
        )
        assert is_retryable_http_error(exc) is True


class TestWithRetry:
    def test_retries_until_success_within_max_attempts(self) -> None:
        attempts = {"count": 0}

        @with_retry(max_attempts=5, min_wait_seconds=0.001, max_wait_seconds=0.01)
        def flaky() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise _httpx_status_error(503)
            return "ok"

        result = flaky()
        assert result == "ok"
        assert attempts["count"] == 3

    def test_gives_up_after_max_attempts_and_reraises(self) -> None:
        attempts = {"count": 0}

        @with_retry(max_attempts=3, min_wait_seconds=0.001, max_wait_seconds=0.01)
        def always_fails() -> str:
            attempts["count"] += 1
            raise _httpx_status_error(500)

        with pytest.raises(httpx.HTTPStatusError):
            always_fails()
        assert attempts["count"] == 3

    def test_non_retryable_error_fails_immediately_without_retrying(self) -> None:
        attempts = {"count": 0}

        @with_retry(max_attempts=5, min_wait_seconds=0.001, max_wait_seconds=0.01)
        def bad_request() -> str:
            attempts["count"] += 1
            raise _httpx_status_error(400)

        with pytest.raises(httpx.HTTPStatusError):
            bad_request()
        assert attempts["count"] == 1

    @pytest.mark.asyncio
    async def test_works_on_async_functions(self) -> None:
        attempts = {"count": 0}

        @with_retry(max_attempts=5, min_wait_seconds=0.001, max_wait_seconds=0.01)
        async def flaky_async() -> str:
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise _httpx_status_error(429)
            return "ok"

        result = await flaky_async()
        assert result == "ok"
        assert attempts["count"] == 2


class TestExtractRetryAfterSeconds:
    def test_seconds_form_is_parsed(self) -> None:
        exc = _httpx_status_error_with_retry_after(429, "5")
        assert _extract_retry_after_seconds(exc) == 5.0

    def test_http_date_form_is_parsed(self) -> None:
        import email.utils
        import time

        future = time.time() + 10
        http_date = email.utils.formatdate(future, usegmt=True)
        exc = _httpx_status_error_with_retry_after(429, http_date)
        delay = _extract_retry_after_seconds(exc)
        assert delay is not None
        # formatdate() truncates to whole seconds — allow a small window.
        assert 8.0 <= delay <= 11.0

    def test_missing_header_returns_none(self) -> None:
        exc = _httpx_status_error(429)
        assert _extract_retry_after_seconds(exc) is None

    def test_unparseable_header_returns_none(self) -> None:
        exc = _httpx_status_error_with_retry_after(429, "not-a-valid-value")
        assert _extract_retry_after_seconds(exc) is None

    def test_no_response_attribute_returns_none(self) -> None:
        assert _extract_retry_after_seconds(ValueError("no response at all")) is None


class TestWithRetryHonorsRetryAfter:
    def test_retry_after_header_is_honored_over_exponential_backoff(self) -> None:
        """A server-supplied Retry-After must override the default
        exponential backoff — before this fix, migrating Jira/TestRail's
        hand-rolled backoff (which read this header explicitly) onto the
        shared tenacity contract silently dropped it, always using blind
        exponential backoff regardless of what the server requested."""
        sleep_calls: list[float] = []

        def _fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        attempts = {"count": 0}

        @with_retry(max_attempts=3, min_wait_seconds=0.001, max_wait_seconds=60.0)
        def flaky() -> str:
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise _httpx_status_error_with_retry_after(429, "17")
            return "ok"

        with patch("tenacity.nap.time.sleep", side_effect=_fake_sleep):
            result = flaky()

        assert result == "ok"
        assert sleep_calls == [17.0]


class TestWaitRetryAfterOrExponentialClampsOverflow:
    """A hostile or malformed Retry-After value must be clamped to
    max_wait_seconds — without this, time.sleep() raises OverflowError on a
    sufficiently large float, crashing the retry loop instead of retrying
    or failing cleanly, bypassing stop_after_attempt entirely."""

    class _FakeOutcome:
        def __init__(self, exc: BaseException) -> None:
            self._exc = exc
            self.failed = True

        def exception(self) -> BaseException:
            return self._exc

    class _FakeRetryState:
        def __init__(self, exc: BaseException, attempt_number: int = 1) -> None:
            self.outcome = TestWaitRetryAfterOrExponentialClampsOverflow._FakeOutcome(exc)
            self.attempt_number = attempt_number

    def test_absurdly_large_retry_after_is_clamped_to_max_wait_seconds(self) -> None:
        waiter = _wait_retry_after_or_exponential(min_wait_seconds=1.0, max_wait_seconds=60.0)
        exc = _httpx_status_error_with_retry_after(429, "99999999999999999999999999999999")
        wait_seconds = waiter(self._FakeRetryState(exc))
        assert wait_seconds == 60.0

    def test_clamped_value_never_overflows_time_sleep(self) -> None:
        """End-to-end proof: the clamped value is small enough that
        time.sleep() (unmocked) never raises OverflowError."""
        waiter = _wait_retry_after_or_exponential(min_wait_seconds=1.0, max_wait_seconds=60.0)
        exc = _httpx_status_error_with_retry_after(429, "99999999999999999999999999999999")
        wait_seconds = waiter(self._FakeRetryState(exc))
        # Not actually sleeping 60s in a test — just confirming the value
        # itself is sleep()-safe (the OverflowError happens at call time,
        # not at argument-construction time).
        import time

        try:
            time.sleep(0.0)  # sanity: sleep() itself works in this env
            float(wait_seconds)  # the real regression: this used to be 1e+32
        except OverflowError:
            pytest.fail("wait_seconds was not clamped — time.sleep() would overflow")
        assert wait_seconds <= 60.0

    def test_plausible_large_value_within_max_wait_is_not_clamped(self) -> None:
        """A real, non-hostile large value (e.g. several hours) under
        max_wait_seconds must still be honored exactly — clamping should
        only kick in when the value exceeds the ceiling."""
        waiter = _wait_retry_after_or_exponential(min_wait_seconds=1.0, max_wait_seconds=3600.0)
        exc = _httpx_status_error_with_retry_after(429, "1800")
        wait_seconds = waiter(self._FakeRetryState(exc))
        assert wait_seconds == 1800.0

    def test_no_retry_after_header_falls_back_to_exponential_unclamped_by_this_path(
        self,
    ) -> None:
        """When there's no Retry-After header, the exponential branch runs
        unaffected by this change — its own max_wait_seconds ceiling
        (wait_random_exponential's max=) already bounds it independently."""
        waiter = _wait_retry_after_or_exponential(min_wait_seconds=0.001, max_wait_seconds=0.01)
        exc = _httpx_status_error(429)  # no Retry-After header
        fake_state = self._FakeRetryState(exc)
        wait_seconds = waiter(fake_state)
        assert 0.0 <= wait_seconds <= 0.01


class TestExtractStatusCodeNeverImportsAnSdk:
    """_extract_status_code()/_is_connection_level_error() used to run
    `import httpx`, `import requests`, `from botocore.exceptions import
    ClientError`, `import openai`, `import anthropic`, `from google.genai
    import errors`, and `import voyageai.error` unconditionally on every
    call, regardless of what exception was being classified. `import
    voyageai.error` alone transitively pulls in torch plus the
    transformers/datasets/accelerate/peft/safetensors stack (well over a
    thousand modules, measured) — on trelix's hottest retry path, on every
    retryable-failure check, even for a plain httpx/ValueError that has
    nothing to do with voyageai.

    The fix (_loaded_module() in src/trelix/core/retry.py) replaces every
    `import X` in these two functions with a `sys.modules.get(X)` lookup:
    an exception cannot be an instance of a class from a module that was
    never imported, so checking the cache is a correct substitute for a
    fresh import and never imports anything itself.

    THE FALSIFYING INPUT: reverting either function's guarded
    `_loaded_module("voyageai.error")` branch back to an unconditional
    `import voyageai.error as voyage_errors` (the pre-fix shape) makes
    test_calling_the_predicate_on_a_plain_exception_never_imports_torch
    fail — confirmed by hand: with that revert, the subprocess prints
    "voyageai,torch" on its second line instead of the empty string this
    test asserts. (Verified via `git stash`/`git stash pop` on the fix
    during development — see this task's proof protocol output — rather
    than re-deriving a live mutation harness inside the test itself.)
    """

    def _run_in_fresh_subprocess(self, exc_expr: str) -> subprocess.CompletedProcess[str]:
        """Classify *exc_expr* (a Python expression, e.g. "ValueError('x')")
        in a brand-new interpreter process that has never imported
        trelix or any SDK, so any import triggered by classification is
        observable in that process's own sys.modules — not contaminated by
        whatever this pytest process happened to import already.

        PYTHONPATH is pointed at THIS repo's own src/ explicitly (not
        inherited) because the shared venv this suite runs under has an
        unrelated editable trelix install; without this override the child
        would import a different tree's retry.py than the one this test
        run is meant to prove."""
        code = (
            "import sys\n"
            "from trelix.core.retry import is_retryable_http_error\n"
            f"result = is_retryable_http_error({exc_expr})\n"
            "watched = ['torch', 'voyageai', 'openai', 'anthropic', 'httpx',"
            " 'requests', 'botocore', 'google.genai']\n"
            "leaked = [m for m in watched if m in sys.modules]\n"
            "print(result)\n"
            "print(','.join(leaked))\n"
        )
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env={
                **os.environ,
                "PYTHONPATH": str(_REPO_ROOT / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=False,
        )

    def test_calling_the_predicate_on_a_plain_exception_never_imports_torch(self) -> None:
        proc = self._run_in_fresh_subprocess("ValueError('not a network error')")
        assert proc.returncode == 0, (
            f"subprocess crashed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
        lines = proc.stdout.splitlines()
        assert lines[0] == "False", f"expected a plain ValueError to be non-retryable: {lines}"
        leaked = lines[1] if len(lines) > 1 else "<missing>"
        assert leaked == "", (
            f"is_retryable_http_error() imported {leaked!r} as a side effect of "
            "classifying a plain ValueError that has nothing to do with any SDK"
        )

    def test_calling_the_predicate_on_a_real_httpx_error_still_does_not_import_torch(
        self,
    ) -> None:
        """Same control, but with a real retryable httpx.HTTPStatusError
        instead of a ValueError — proves the fix holds even on the path
        that DOES match a branch (httpx's) and return early, not just on
        the always-falls-through ValueError case above."""
        exc_expr = (
            "__import__('httpx').HTTPStatusError("
            "'503 error', "
            "request=__import__('httpx').Request('GET', 'https://example.com'), "
            "response=__import__('httpx').Response(503, "
            "request=__import__('httpx').Request('GET', 'https://example.com')))"
        )
        proc = self._run_in_fresh_subprocess(exc_expr)
        assert proc.returncode == 0, (
            f"subprocess crashed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
        lines = proc.stdout.splitlines()
        assert lines[0] == "True", f"expected httpx 503 to be retryable: {lines}"
        # httpx itself is expected to be loaded (the exception is one of its
        # classes) -- the control is that nothing ELSE leaked alongside it.
        leaked = set((lines[1] if len(lines) > 1 else "").split(",")) - {""}
        assert leaked == {"httpx"}, (
            f"classifying an httpx error imported more than just httpx: {leaked}"
        )
