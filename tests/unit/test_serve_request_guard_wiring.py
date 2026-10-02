"""`trelix serve` -> `create_app(allowed_hosts=...)` wiring for the Host/Origin guard.

The enabling rule lives in `serve` (the only place that knows the bind host): the guard is
ON when the API is open (no static token, no OIDC) AND the bind host is loopback. A
configured credential already stops the DNS-rebinding read, and failing closed for a
non-loopback bind is a bigger decision than this fix, so those cases keep today's
behavior unless `TRELIX_API_ALLOWED_HOSTS` opts in.

`serve` ends in `uvicorn.run()`, which blocks, so it is patched (no socket is ever bound;
the unit suite runs under `--disable-socket`). Two styles are used: `create_app` mocked to
read the argument `serve` passed, and the real `create_app` with the app handed to the
patched `uvicorn.run` probed over `TestClient`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from trelix.cli.main import _api_auth_is_open, serve

LOOPBACK = ("localhost", "127.0.0.1", "::1")
EVIL = "evil-rebind-canary.example"
_OPEN_BIND_ALL = "0.0.0.0"  # noqa: S104 - the literal IS the subject: the non-loopback bind under test


def _serve_with_mocked_create_app(repo: Path, host: str) -> Any:
    """Run `serve`; return the `allowed_hosts` keyword it handed to `create_app`."""
    with (
        patch("trelix.api.app.create_app") as create_app,
        patch("uvicorn.run"),
        patch("trelix.core.logging_setup.setup_json_logging"),
    ):
        serve(repo_path=str(repo), host=host, port=8765)
    assert create_app.call_count == 1
    assert create_app.call_args.kwargs["served_root"] == str(repo)
    return create_app.call_args.kwargs["allowed_hosts"]


def _serve_and_capture_app(repo: Path, host: str) -> Any:
    """Run `serve` with the REAL create_app; return the app given to uvicorn.run."""
    with (
        patch("uvicorn.run") as run,
        patch("trelix.core.logging_setup.setup_json_logging"),
    ):
        serve(repo_path=str(repo), host=host, port=8765)
    return run.call_args.args[0]


@pytest.fixture(autouse=True)
def _clean_guard_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRELIX_API_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("TRELIX_API_AUTH_TOKEN", raising=False)


def test_the_auth_precondition_is_really_unconfigured() -> None:
    """Without this every 'open' case below could pass for the wrong reason."""
    assert _api_auth_is_open() is True


class TestWhichHostsServePassesToCreateApp:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]"])
    def test_open_api_on_loopback_turns_the_guard_on_with_the_loopback_names(
        self, tmp_path: Path, host: str
    ) -> None:
        assert _serve_with_mocked_create_app(tmp_path, host) == LOOPBACK

    def test_a_specific_loopback_address_is_added_to_the_list(self, tmp_path: Path) -> None:
        got = _serve_with_mocked_create_app(tmp_path, "127.0.0.2")
        assert got == (*LOOPBACK, "127.0.0.2")

    @pytest.mark.parametrize("host", [_OPEN_BIND_ALL, "::", "10.0.0.7", "example.internal"])
    def test_a_non_loopback_bind_leaves_the_guard_to_the_env_var(
        self, tmp_path: Path, host: str
    ) -> None:
        assert _serve_with_mocked_create_app(tmp_path, host) is None

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
    def test_a_configured_token_on_loopback_keeps_todays_behavior(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str
    ) -> None:
        monkeypatch.setenv("TRELIX_API_AUTH_TOKEN", "fake-token-for-tests")
        assert _serve_with_mocked_create_app(tmp_path, host) is None

    def test_a_configured_oidc_verifier_on_loopback_keeps_todays_behavior(
        self, tmp_path: Path
    ) -> None:
        with patch("trelix.api.app._build_oidc_verifier", return_value=object()):
            assert _api_auth_is_open() is False
            assert _serve_with_mocked_create_app(tmp_path, "127.0.0.1") is None

    def test_env_hosts_on_a_non_loopback_bind_are_left_to_create_app(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_API_ALLOWED_HOSTS", "trelix.internal")
        assert _serve_with_mocked_create_app(tmp_path, _OPEN_BIND_ALL) is None

    def test_the_serve_level_rule_does_not_decide_the_env_star_opt_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """serve still passes the tuple; create_app owns the `*` opt-out (next class)."""
        monkeypatch.setenv("TRELIX_API_ALLOWED_HOSTS", "*")
        assert _serve_with_mocked_create_app(tmp_path, "127.0.0.1") == LOOPBACK


class TestEndToEndThroughServe:
    """Real `create_app`; probe the app `serve` hands to uvicorn."""

    @staticmethod
    def _status(app: Any, host: str, headers: dict[str, str] | None = None) -> int:
        client = TestClient(app, base_url="http://localhost:8765")
        return client.get("/stats", headers={"Host": host, **(headers or {})}).status_code

    def test_open_loopback_serve_refuses_a_rebound_host_and_a_foreign_origin(
        self, tmp_path: Path
    ) -> None:
        app = _serve_and_capture_app(tmp_path, "127.0.0.1")
        assert self._status(app, f"{EVIL}:8765") == 403
        assert self._status(app, "localhost:8765", {"Origin": f"http://{EVIL}"}) == 403
        assert self._status(app, "127.0.0.1:8765") != 403
        assert self._status(app, "[::1]:8765") != 403

    def test_open_loopback_serve_still_answers_health_for_any_host(self, tmp_path: Path) -> None:
        app = _serve_and_capture_app(tmp_path, "127.0.0.1")
        client = TestClient(app, base_url="http://localhost:8765")
        assert client.get("/health", headers={"Host": "10.0.0.9:8765"}).status_code == 200

    def test_env_star_turns_the_guard_off_even_on_an_open_loopback_bind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_API_ALLOWED_HOSTS", "*")
        app = _serve_and_capture_app(tmp_path, "127.0.0.1")
        assert self._status(app, EVIL) != 403

    def test_env_hosts_are_joined_to_the_loopback_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_API_ALLOWED_HOSTS", "dev.trelix.test")
        app = _serve_and_capture_app(tmp_path, "127.0.0.1")
        assert self._status(app, "dev.trelix.test:8765") != 403
        assert self._status(app, "localhost") != 403
        assert self._status(app, EVIL) == 403

    def test_a_token_on_loopback_leaves_the_api_unguarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRELIX_API_AUTH_TOKEN", "fake-token-for-tests")
        app = _serve_and_capture_app(tmp_path, "127.0.0.1")
        # Not 403: the Host is not checked; the request reaches auth and gets a 401.
        assert self._status(app, EVIL) == 401

    def test_a_non_loopback_bind_without_env_hosts_is_unguarded_as_before(
        self, tmp_path: Path
    ) -> None:
        app = _serve_and_capture_app(tmp_path, _OPEN_BIND_ALL)
        assert self._status(app, EVIL) != 403

    def test_a_non_loopback_bind_with_env_hosts_is_guarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The compose shape: `--host 0.0.0.0` plus TRELIX_API_ALLOWED_HOSTS."""
        monkeypatch.setenv("TRELIX_API_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]")
        app = _serve_and_capture_app(tmp_path, _OPEN_BIND_ALL)
        assert self._status(app, "localhost:8765") != 403
        assert self._status(app, "[::1]:8765") != 403
        assert self._status(app, EVIL) == 403

    def test_serve_start_logs_one_line_naming_the_opt_out_when_the_guard_is_on(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="trelix.api"):
            _serve_and_capture_app(tmp_path, "127.0.0.1")
        lines = [
            r.getMessage() for r in caplog.records if "TRELIX_API_ALLOWED_HOSTS=*" in r.getMessage()
        ]
        assert len(lines) == 1

    def test_serve_start_logs_nothing_about_the_guard_when_it_is_off(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="trelix.api"):
            _serve_and_capture_app(tmp_path, _OPEN_BIND_ALL)
        assert [r for r in caplog.records if "TRELIX_API_ALLOWED_HOSTS" in r.getMessage()] == []


def test_the_exposure_warning_still_fires_for_a_non_loopback_open_bind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The guard did not touch the existing warning (same open-mode test, shared helper)."""
    with patch("uvicorn.run"), patch("trelix.core.logging_setup.setup_json_logging"):
        serve(repo_path=str(tmp_path), host=_OPEN_BIND_ALL, port=8765)
    assert "no authentication configured" in " ".join(capsys.readouterr().err.split())


def test_a_loopback_open_serve_prints_no_exposure_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("uvicorn.run"), patch("trelix.core.logging_setup.setup_json_logging"):
        serve(repo_path=str(tmp_path), host="127.0.0.1", port=8765)
    assert capsys.readouterr().err == ""


def test_mocked_create_app_fixture_sanity(tmp_path: Path) -> None:
    """`_serve_with_mocked_create_app` must not be vacuous: a mock app is what uvicorn gets."""
    with (
        patch("trelix.api.app.create_app", return_value=MagicMock(name="app")) as create_app,
        patch("uvicorn.run") as run,
        patch("trelix.core.logging_setup.setup_json_logging"),
    ):
        serve(repo_path=str(tmp_path), host="127.0.0.1", port=8765)
    assert run.call_args.args[0] is create_app.return_value
